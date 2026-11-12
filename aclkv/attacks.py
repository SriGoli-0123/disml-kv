"""Leakage analysis under realistic attackers (simulator).

Two attackers are modelled:

* **Bypass** (v1): replays a victim's exact tokens straight at the engine
  without the server-keyed scope ids (``PrefixCacheSimulator.probe``).
* **Insider** (v2): a co-authorized colleague who can read every document in
  the victim's prompt re-issues a guessed question *through the trusted
  middleware*.  We give the attacker the strongest position: their retrieval
  returns exactly the victim's evidence (they guessed the question).  The
  attacker observes which blocks hit (in practice via TTFT; the simulator
  reports the ground truth).

A hit is **informative** if the block was computed or reused by some other
user and never by the attacker.  For the hard-pinned evidence catalog a hit
on a published block happens whoever asked since publication, so its
anonymity set is the set of users whose requests got the prefix published.  Hits on the system preamble and pinned
documents are excluded (everyone requests them).  For every informative hit we
compute the **anonymity set**: the distinct users other than the attacker who
had requested that exact evidence prefix (or, for the question tail, that
exact question with that evidence) at any earlier point of the trace.  Using
all history makes the anonymity sets of the baselines *larger* than what the
cache alone reveals, so the comparison is conservative with respect to our
defence.

Reported per policy: fraction of probes that reveal anything; fraction whose
most revealing hit has an anonymity set of exactly one other user
("singled out") or fewer than 3 / 5; fraction that hit the victim's question
text; and the utility of the same trace (cached prompt tokens of honest
requests).

The **order channel** is reported separately: if users can see the order of
their sources, a cache-aware order reveals that other users requested the
prefix it follows (``ordering.order_exposure``).  For every honest request we
take the anonymity set of what its order reveals; ``order_singled_out`` is the
fraction of requests whose order points at exactly one other user.  The order
metrics are NaN for orderings that are not audited (popularity-based v1
orderings).
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass

from .catalog import CatalogConfig
from .data_prep import Chunk
from .pipeline import BuildSettings, make_ordering_state
from .policies import Policy, get_policy
from .simulator import PrefixCacheSimulator
from .workload import Workload


@dataclass
class AttackSummary:
    policy: str
    probes: int
    skipped_no_insider: int
    revealed: float              # P(any informative evidence/tail hit)
    singled_out: float           # P(most revealing hit has anonymity set == 1)
    anon_lt3: float              # P(... < 3)
    anon_lt5: float              # P(... < 5)
    query_revealed: float        # P(hit on the victim's question tail)
    mean_informative_tokens: float
    min_anon_observed: int       # smallest anonymity set over all informative hits (-1 if none)
    bypass_hits: int             # honest-request blocks hit by bypass probes (no scope ids)
    bypass_probes: int
    honest_cached_pct: float
    honest_requests: int
    order_revealed: float = math.nan       # P(an honest request's order reveals >= 1 other user)
    order_singled_out: float = math.nan    # P(... exactly one other user)
    order_anon_lt3: float = math.nan       # P(... 1 or 2 other users)
    order_min_anon: int = -1               # smallest non-empty order anonymity set (-1: none)
    publish_pct: float = 0.0               # catalog publisher prefill, % of honest prompt tokens

    def to_json(self) -> dict:
        return asdict(self)


def _seg_key(prompt, seg) -> str:
    if seg.name == "tail":
        return seg.prefix_key + "|" + str(prompt.meta.get("qid", ""))
    return seg.prefix_key


def run_attack_trace(workload: Workload, corpus: dict[str, Chunk], policy: Policy | str, builder,
                     settings: BuildSettings, num_blocks: int, block_size: int = 16, max_concurrency: int = 16,
                     output_tokens: int = 48, probe_rate: float = 0.2, bypass_rate: float = 0.1,
                     seed: int = 0, mode: str = "prefix", pic_scoping: str = "acl", eviction: str = "lru",
                     catalog: CatalogConfig | None = None) -> AttackSummary:
    pol = get_policy(policy) if isinstance(policy, str) else policy
    pinned = workload.pinned_docs(corpus)
    state = make_ordering_state(pol, settings, pinned, catalog, num_blocks, output_tokens)
    sim = PrefixCacheSimulator(num_blocks, workload.directory, block_size, max_concurrency, output_tokens,
                               mode=mode, pic_scoping=pic_scoping, eviction=eviction)
    rng = random.Random(seed)
    requesters: dict[str, set[str]] = defaultdict(set)
    users = list(workload.directory.users)

    probes = skipped = revealed = singled = lt3 = lt5 = qrev = 0
    info_tokens = 0
    min_anon_seen = None
    bypass_hits = bypass_probes = 0
    honest_pt = honest_ct = 0
    order_sets: list = []
    order_audited = True

    def log(prompt, user):
        for seg in prompt.segments:
            if seg.name in ("doc", "tail"):
                requesters[_seg_key(prompt, seg)].add(user)

    for r in workload.requests:
        docs = r.retrieved_docs(corpus)
        if state.catalog is not None:
            event = state.catalog.maybe_publish(builder, pol, state, pinned)
            if event is not None:
                sim.apply_publish(event)
                if event["mode"] == "hard":
                    # a hit on a hard-pinned block happens whoever asked since publication: it
                    # reveals only the users whose window requests got the prefix published
                    for key, us in state.catalog.published_anonymity().items():
                        requesters[key] |= us
        p = builder.build(r.req_id, r.user, r.question, docs, pol, ordering=state, pinned=pinned)
        p.meta.update({"qid": r.qid})
        if "order_anon" in p.meta:
            order_sets.append(p.meta["order_anon"])
        else:
            order_audited = False
        rec = sim.process(p)
        honest_pt += rec.prompt_tokens
        honest_ct += rec.cached_tokens
        log(p, r.user)

        if bypass_rate > 0 and rng.random() < bypass_rate:
            final_acl = p.segments[-1].acl
            outsiders = [u for u in users if not final_acl.allows(u, workload.directory)]
            if outsiders:
                b = sim.probe(p, rng.choice(outsiders), "no_salt", rng)
                bypass_probes += 1
                bypass_hits += b.honest_hit_blocks      # hits on attacker-made blocks reveal nothing

        if rng.random() >= probe_rate:
            continue
        final_acl = p.segments[-1].acl
        insiders = [u for u in users if u != r.user and final_acl.allows(u, workload.directory)]
        if not insiders:
            skipped += 1
            continue
        attacker = rng.choice(insiders)
        q = builder.build(f"probe-{r.req_id}", attacker, r.question, docs, pol, ordering=state, pinned=pinned)
        q.meta.update({"qid": r.qid, "probe": True, "record_positions": True})
        prec = sim.process(q)
        probes += 1
        anon_sizes, hit_tail, tokens = [], False, 0
        for pos in prec.informative_positions:
            seg = q.segment_at(pos * block_size)
            if seg.name not in ("doc", "tail"):
                continue
            others = requesters.get(_seg_key(q, seg), set()) - {attacker}
            anon_sizes.append(len(others))
            tokens += block_size
            if seg.name == "tail":
                hit_tail = True
        if anon_sizes:
            revealed += 1
            m = min(anon_sizes)
            min_anon_seen = m if min_anon_seen is None else min(min_anon_seen, m)
            singled += m <= 1
            lt3 += m < 3
            lt5 += m < 5
        qrev += hit_tail
        info_tokens += tokens
        log(q, attacker)

    n = max(1, probes)
    revealing = [a for a in order_sets if a is not None and a >= 1]
    m = max(1, len(order_sets))
    order = dict(
        order_revealed=len(revealing) / m if order_audited else math.nan,
        order_singled_out=sum(a == 1 for a in revealing) / m if order_audited else math.nan,
        order_anon_lt3=sum(a < 3 for a in revealing) / m if order_audited else math.nan,
        order_min_anon=min(revealing) if (order_audited and revealing) else -1,
    )
    return AttackSummary(
        policy=pol.name, probes=probes, skipped_no_insider=skipped,
        revealed=revealed / n, singled_out=singled / n, anon_lt3=lt3 / n, anon_lt5=lt5 / n,
        query_revealed=qrev / n, mean_informative_tokens=info_tokens / n,
        min_anon_observed=-1 if min_anon_seen is None else min_anon_seen,
        bypass_hits=bypass_hits, bypass_probes=bypass_probes,
        honest_cached_pct=100.0 * honest_ct / max(1, honest_pt), honest_requests=len(workload.requests),
        publish_pct=100.0 * sim.publish_tokens / max(1, honest_pt), **order,
    )
