"""Trace-driven prefix-cache simulator that mirrors vLLM V1 semantics.

Modelled after ``vllm/v1/core/{block_pool,kv_cache_manager}.py`` (v0.29.0):

* only *full* blocks are hashed; hashes chain through the parent hash and
  include the block's extra keys (our scope barriers);
* the longest cached prefix is looked up but capped at ``num_tokens - 1``
  (vLLM always recomputes at least one token to obtain logits);
* cache-hit blocks are "touched" (removed from the free queue);
* new blocks are taken from the head of the free queue; a popped block that
  still carries a hash is an **eviction**;
* when a request finishes its blocks are freed in *reverse* order: cached
  blocks go to the tail (LRU), never-cached blocks to the head;
* full blocks are registered in the hash table on allocation, so concurrent
  requests with a common prefix share it (as in vLLM V1).

Concurrency is modelled as a sliding window of ``max_concurrency`` in-flight
requests, each holding its prompt + ``output_tokens`` blocks until retired.

On top of the vLLM model the simulator keeps, for every cached block, the
**ground-truth effective ACL** of the prefix it ends, and audits each hit
against the requester (the cache-hit gate).  Under a secure policy the number
of unauthorized hits must be exactly zero.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass, field

from .acl import ACL, PUBLIC, Directory
from .context import BuiltPrompt
from .scope import Barrier, scope_keys_for_block


def _hash(parent: bytes | None, tokens: tuple[int, ...], extra: tuple | None) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    h.update(parent or b"\x00" * 16)
    h.update(repr(tokens).encode())
    h.update(repr(extra).encode())
    return h.digest()


class _Block:
    __slots__ = ("bid", "hash", "ref_cnt", "acl", "from_probe", "contributors")

    def __init__(self, bid: int):
        self.bid = bid
        self.hash: bytes | None = None
        self.ref_cnt = 0
        self.acl: ACL = PUBLIC
        self.from_probe = False     # computed by an adversarial probe, not an honest request
        self.contributors: set[str] = set()   # users whose requests computed or reused this block


class _FreeQueue:
    """Eviction-ordered free list (head == evict first) with O(1) remove."""

    def __init__(self, bids):
        self._od: OrderedDict[int, None] = OrderedDict((b, None) for b in bids)

    def __len__(self):
        return len(self._od)

    def __contains__(self, bid):
        return bid in self._od

    def popleft(self) -> int:
        bid, _ = self._od.popitem(last=False)
        return bid

    def remove(self, bid: int) -> None:
        del self._od[bid]

    def append(self, bid: int) -> None:
        self._od[bid] = None

    def prepend(self, bid: int) -> None:
        self._od[bid] = None
        self._od.move_to_end(bid, last=False)


@dataclass
class SimRecord:
    req_id: str
    user: str
    policy: str
    prompt_tokens: int
    cached_tokens: int
    hit_blocks: int
    unauthorized_hits: int
    unauthorized_acls: list[str]
    evictions: int
    new_blocks: int
    stalls: int
    table_size: int
    free_blocks: int
    probe: bool = False
    informative_positions: list = field(default_factory=list)   # hit block indices computed by *other* users
    honest_hit_blocks: int = 0      # hits on blocks computed by honest (non-probe) requests

    @property
    def recomputed_tokens(self) -> int:
        return self.prompt_tokens - self.cached_tokens


@dataclass
class SimSummary:
    policy: str
    num_blocks: int
    block_size: int
    max_concurrency: int
    output_tokens: int
    requests: int
    prompt_tokens: int
    cached_tokens: int
    recomputed_tokens: int
    cached_pct: float
    hit_blocks: int
    unauthorized_hits: int
    evictions: int
    stalls: int
    max_table_size: int
    mean_table_size: float
    unique_hashes_seen: int
    probe_requests: int = 0
    probe_unauthorized_hits: int = 0      # must be 0 for every secure policy
    probe_cached_tokens: int = 0
    mode: str = "prefix"
    extra: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)


class PrefixCacheSimulator:
    def __init__(
        self,
        num_blocks: int,
        directory: Directory,
        block_size: int = 16,
        max_concurrency: int = 8,
        output_tokens: int = 32,
        mode: str = "prefix",
        pic_recompute: float = 0.15,
        pic_scoping: str = "acl",
    ):
        """``mode="prefix"`` mirrors vLLM automatic prefix caching.  ``mode="pic"``
        models *position-independent* chunk caching (CacheBlend-style): each
        evidence chunk's KV is cached independently of what precedes it and
        reused at any position, with a fraction ``pic_recompute`` of its tokens
        recomputed to restore cross-chunk attention.  ``pic_scoping`` keys chunk
        entries by the chunk's *own* ACL ("acl"), not at all ("global"), or per
        user ("user")."""
        if num_blocks < 2:
            raise ValueError("need at least 2 blocks")
        if mode not in ("prefix", "pic"):
            raise ValueError(mode)
        self.num_blocks = num_blocks
        self.bs = block_size
        self.max_concurrency = max_concurrency
        self.output_tokens = output_tokens
        self.directory = directory
        self.mode = mode
        self.pic_recompute = pic_recompute
        self.pic_scoping = pic_scoping
        self.reset()

    # ----------------------------------------------------------- state
    def reset(self) -> None:
        self.blocks = [_Block(i) for i in range(self.num_blocks)]
        self.free = _FreeQueue(range(self.num_blocks))
        self.table: dict[bytes, int] = {}
        self.in_flight: deque[tuple[str, list[int]]] = deque()
        self.records: list[SimRecord] = []
        self._seen_hashes: set[bytes] = set()
        self._table_sizes: list[int] = []

    # --------------------------------------------------------- hashing
    def block_hashes(self, prompt: BuiltPrompt) -> list[bytes]:
        barriers = tuple((b.token_offset, b.scope_id) for b in prompt.barriers)
        out: list[bytes] = []
        parent: bytes | None = None
        ids = prompt.token_ids
        for i in range(len(ids) // self.bs):
            s, e = i * self.bs, (i + 1) * self.bs
            keys = scope_keys_for_block(barriers, s, e)
            extra = tuple(keys) if keys else None
            h = _hash(parent, tuple(ids[s:e]), extra)
            out.append(h)
            parent = h
        return out

    # ------------------------------------------------------- lifecycle
    def _retire_oldest(self) -> None:
        _, bids = self.in_flight.popleft()
        # vLLM frees in reverse order so tail blocks are evicted first
        to_first, to_last = [], []
        for bid in reversed(bids):
            b = self.blocks[bid]
            b.ref_cnt -= 1
            if b.ref_cnt == 0:
                (to_last if b.hash is not None else to_first).append(bid)
        for bid in reversed(to_first):
            self.free.prepend(bid)
        for bid in to_last:
            self.free.append(bid)

    def drain(self) -> None:
        while self.in_flight:
            self._retire_oldest()

    def probe(self, victim: BuiltPrompt, attacker: str, mode: str = "no_salt", rng: random.Random | None = None) -> SimRecord:
        """Adversarial replay (the timing-probe experiment).

        ``attacker`` submits the victim's *exact token sequence* straight to
        the engine — i.e. bypassing the trusted middleware — but without the
        server-keyed scope ids: ``mode="no_salt"`` sends no salt at all,
        ``mode="forged"`` keeps the victim's barrier offsets but with random
        scope ids.  Under global sharing (UB) the attacker hits every cached
        block of the victim's prefix; under any scoped policy the attacker's
        blocks live in a disjoint namespace and must hit nothing restricted.
        Probe requests occupy cache like any other request.
        """
        rng = rng or random
        if mode == "no_salt":
            barriers = []
        elif mode == "forged":
            barriers = [Barrier(b.token_offset, "%016x" % rng.getrandbits(64), b.acl_canonical) for b in victim.barriers]
        else:
            raise ValueError(mode)
        fake = BuiltPrompt(
            req_id=f"probe-{mode}-{victim.req_id}", user=attacker, token_ids=victim.token_ids,
            segments=victim.segments, barriers=barriers, cache_salt=None,
            ordered_chunk_ids=victim.ordered_chunk_ids, policy=victim.policy,
            meta={"probe": True, "victim": victim.req_id, "probe_mode": mode},
        )
        rec = self.process(fake)
        rec.probe = True
        return rec

    def _pic_hashes(self, prompt: BuiltPrompt) -> tuple[list[bytes], list[float], list[bool]]:
        """Per-block hashes for position-independent caching: the hash chain
        restarts at every segment, chunks are keyed by their own ACL, and the
        question tail is never shareable.  Also returns, per block, the
        fraction of its tokens saved on a hit and whether it may hit at all."""
        bs = self.bs
        ids = prompt.token_ids
        hashes, saved, can_hit = [], [], []
        seg_of = [prompt.segment_at(i * bs) for i in range(len(ids) // bs)]
        parent, cur_seg = None, None
        for i, seg in enumerate(seg_of):
            if seg is not cur_seg:
                parent, cur_seg = None, seg
            if seg.name == "preamble":
                extra = ("system",)
                frac, ok = 1.0, True
            elif seg.name in ("doc", "pinned"):
                if self.pic_scoping == "acl":
                    extra = ("acl", (seg.own_acl or seg.acl).canonical())
                elif self.pic_scoping == "user":
                    extra = ("user", prompt.user)
                else:
                    extra = None
                frac, ok = 1.0 - self.pic_recompute, True
            else:                                     # tail / markers: request-private
                extra = ("req", prompt.req_id)
                frac, ok = 0.0, False
            h = _hash(parent, tuple(ids[i * bs:(i + 1) * bs]), extra)
            hashes.append(h)
            saved.append(frac)
            can_hit.append(ok)
            parent = h
        return hashes, saved, can_hit

    def process(self, prompt: BuiltPrompt, output_tokens: int | None = None) -> SimRecord:
        bs = self.bs
        n = prompt.num_tokens
        out_tokens = self.output_tokens if output_tokens is None else output_tokens
        is_probe = bool(prompt.meta.get("probe"))

        if self.mode == "prefix":
            hashes = self.block_hashes(prompt)
            # longest cached prefix, capped at num_tokens - 1
            max_hits = (n - 1) // bs
            hit_pos: list[int] = []
            for i, h in enumerate(hashes[:max_hits]):
                if h not in self.table:
                    break
                hit_pos.append(i)
            saved = [1.0] * len(hashes)
        else:
            hashes, saved, can_hit = self._pic_hashes(prompt)
            hit_pos = [i for i, h in enumerate(hashes) if can_hit[i] and h in self.table]
        self._seen_hashes.update(hashes)
        hits = [self.table[hashes[i]] for i in hit_pos]
        hit_set = set(hit_pos)

        # cache-hit gate audit (ground truth): a leak is a hit on state computed by an
        # *honest* request whose effective ACL does not include the requester
        unauthorized = 0
        bad_acls: list[str] = []
        informative: list[int] = []
        honest_hits = 0
        for i, bid in zip(hit_pos, hits):
            blk = self.blocks[bid]
            if prompt.user not in blk.contributors:
                informative.append(i)
            if blk.from_probe:
                continue
            honest_hits += 1
            if not blk.acl.allows(prompt.user, self.directory):
                unauthorized += 1
                bad_acls.append(blk.acl.canonical())

        total_tokens = n + out_tokens
        n_total_blocks = math.ceil(total_tokens / bs)
        need = n_total_blocks - len(hits)
        # hit blocks that are currently free will be removed from the free queue by
        # ``touch`` — vLLM subtracts them too (``num_evictable_computed_blocks``)
        evictable_hits = sum(1 for bid in hits if self.blocks[bid].ref_cnt == 0)
        stalls = 0
        while len(self.free) - evictable_hits < need and self.in_flight:
            self._retire_oldest()
            stalls += 1
            evictable_hits = sum(1 for bid in hits if self.blocks[bid].ref_cnt == 0)
        if len(self.free) - evictable_hits < need:
            raise RuntimeError(f"KV budget of {self.num_blocks} blocks cannot hold one request ({need} blocks)")

        for bid in hits:                       # touch
            b = self.blocks[bid]
            if b.ref_cnt == 0:
                self.free.remove(bid)
            b.ref_cnt += 1

        new_bids: list[int] = []
        evictions = 0
        for _ in range(need):
            bid = self.free.popleft()
            b = self.blocks[bid]
            if b.hash is not None:
                if self.table.get(b.hash) == bid:
                    del self.table[b.hash]
                b.hash = None
                b.from_probe = False
                b.contributors = set()
                evictions += 1
            b.ref_cnt = 1
            new_bids.append(bid)

        # block position -> physical block
        assign: list[int] = []
        it = iter(new_bids)
        hit_iter = dict(zip(hit_pos, hits))
        for i in range(n_total_blocks):
            assign.append(hit_iter[i] if i in hit_set else next(it))
        for i in range(len(hashes)):                       # newly computed full prompt blocks
            if i in hit_set:
                continue
            b = self.blocks[assign[i]]
            b.hash = hashes[i]
            if self.mode == "pic":
                # a position-independent chunk is encoded in isolation: its true label is its own ACL
                seg = prompt.segment_at(i * bs)
                b.acl = seg.own_acl if seg.own_acl is not None else (PUBLIC if seg.name == "preamble" else seg.acl)
            else:
                b.acl = prompt.acl_at(i * bs + bs - 1)
            b.from_probe = is_probe
            self.table.setdefault(hashes[i], assign[i])
        final_acl = prompt.segments[-1].acl
        for i in range(len(hashes), total_tokens // bs):  # full decode blocks (unique)
            h = _hash(None, (i,), ("out", prompt.req_id))
            b = self.blocks[assign[i]]
            b.hash = h
            b.acl = final_acl
            b.from_probe = is_probe
            self.table[h] = assign[i]
        for i in range(len(hashes)):
            self.blocks[assign[i]].contributors.add(prompt.user)

        self.in_flight.append((prompt.req_id, assign))
        while len(self.in_flight) > self.max_concurrency:
            self._retire_oldest()

        cached = int(round(sum(bs * saved[i] for i in hit_pos)))
        rec = SimRecord(
            req_id=prompt.req_id, user=prompt.user, policy=prompt.policy,
            prompt_tokens=n, cached_tokens=cached, hit_blocks=len(hits),
            unauthorized_hits=unauthorized, unauthorized_acls=bad_acls,
            evictions=evictions, new_blocks=len(new_bids), stalls=stalls,
            table_size=len(self.table), free_blocks=len(self.free),
            informative_positions=informative if (is_probe or prompt.meta.get("record_positions")) else [],
            honest_hit_blocks=honest_hits,
        )
        self.records.append(rec)
        self._table_sizes.append(len(self.table))
        return rec

    # --------------------------------------------------------- summary
    def summary(self, policy: str = "") -> SimSummary:
        probes = [r for r in self.records if r.probe]
        recs = [r for r in self.records if not r.probe]
        pt = sum(r.prompt_tokens for r in recs)
        ct = sum(r.cached_tokens for r in recs)
        return SimSummary(
            policy=policy or (recs[0].policy if recs else ""),
            num_blocks=self.num_blocks, block_size=self.bs,
            max_concurrency=self.max_concurrency, output_tokens=self.output_tokens,
            requests=len(recs), prompt_tokens=pt, cached_tokens=ct,
            recomputed_tokens=pt - ct, cached_pct=(100.0 * ct / pt) if pt else 0.0,
            hit_blocks=sum(r.hit_blocks for r in recs),
            unauthorized_hits=sum(r.unauthorized_hits for r in recs),
            evictions=sum(r.evictions for r in recs),
            stalls=sum(r.stalls for r in recs),
            max_table_size=max(self._table_sizes, default=0),
            mean_table_size=(sum(self._table_sizes) / len(self._table_sizes)) if self._table_sizes else 0.0,
            unique_hashes_seen=len(self._seen_hashes),
            probe_requests=len(probes),
            probe_unauthorized_hits=sum(r.unauthorized_hits for r in probes),
            probe_cached_tokens=sum(r.cached_tokens for r in probes),
            mode=self.mode,
        )


def simulate(prompts, num_blocks: int, directory: Directory, block_size: int = 16,
             max_concurrency: int = 8, output_tokens: int = 32,
             output_lengths: dict[str, int] | None = None,
             probe_rate: float = 0.0, probe_mode: str = "no_salt", seed: int = 0,
             mode: str = "prefix", pic_recompute: float = 0.15, pic_scoping: str = "acl") -> tuple[list[SimRecord], SimSummary]:
    """Replay ``prompts``; with probability ``probe_rate`` after each request an
    unauthorized user replays it adversarially (see ``PrefixCacheSimulator.probe``)."""
    sim = PrefixCacheSimulator(num_blocks, directory, block_size, max_concurrency, output_tokens,
                               mode=mode, pic_recompute=pic_recompute, pic_scoping=pic_scoping)
    rng = random.Random(seed)
    users = list(directory.users)
    for p in prompts:
        sim.process(p, None if output_lengths is None else output_lengths.get(p.req_id))
        if probe_rate > 0 and rng.random() < probe_rate:
            final_acl = p.segments[-1].acl
            outsiders = [u for u in users if not final_acl.allows(u, directory)]
            if outsiders:
                sim.probe(p, rng.choice(outsiders), probe_mode, rng)
    return sim.records, sim.summary()
