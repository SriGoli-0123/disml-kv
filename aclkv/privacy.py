"""Privacy state kept by the trusted middleware.

Two leakage channels survive ACL-derived scoping (see the paper, Sec. 3):

1. **Query leakage.**  If the question tail inherits the documents' scope, any
   co-authorized user who submits the same question obtains a cache hit on the
   victim's *question text* — the attack class (PromptPeek / InputSnatch /
   Early Bird) that request-level salting was introduced to stop.  Fix: the
   tail is always scoped to the requesting user (``Policy.tail_scope``).

2. **Retrieval-fingerprint leakage.**  Even with a private tail, the retrieved
   document *sequence* is a near-unique fingerprint of the query.  A colleague
   who re-issues a guessed question gets hits on the question-specific part of
   the evidence prefix iff someone in the audience recently asked it.

``PromotionTracker`` implements **k-anonymous promotion**: an evidence prefix
is hashed into its shared (ACL) scope only once at least ``k`` distinct
authorized users have requested exactly that prefix; before that it is scoped
to the requesting user.  A cache hit on a shared block therefore reveals only
that >= k distinct users (possibly including the prober) asked for it.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Iterable


def doc_prefix_key(chunk_ids: Iterable[str]) -> str:
    """Stable short digest of an ordered evidence prefix."""
    h = hashlib.blake2b(digest_size=12)
    for c in chunk_ids:
        h.update(c.encode())
        h.update(b"\x1f")
    return h.hexdigest()


class PromotionTracker:
    """Distinct-requester counts per (effective ACL, evidence prefix).

    Bounded LRU so the middleware's memory does not grow with traffic; an
    evicted prefix simply has to be re-promoted (fails closed).
    """

    def __init__(self, k: int, max_entries: int = 500_000):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.max_entries = max_entries
        self._users: OrderedDict[tuple[str, str], set[str]] = OrderedDict()

    def distinct(self, key: tuple[str, str]) -> int:
        return len(self._users.get(key, ()))

    def would_share(self, key: tuple[str, str], user: str) -> bool:
        """Shared scope iff the current user would be (at least) the k-th distinct requester."""
        if self.k <= 1:
            return True
        users = self._users.get(key)
        if users is None:
            return False
        return len(users) + (0 if user in users else 1) >= self.k

    def observe(self, key: tuple[str, str], user: str) -> None:
        users = self._users.get(key)
        if users is None:
            users = set()
            self._users[key] = users
            if len(self._users) > self.max_entries:
                self._users.popitem(last=False)
        else:
            self._users.move_to_end(key)
        if len(users) < self.k:          # once promoted, the exact count no longer matters
            users.add(user)

    def __len__(self) -> int:
        return len(self._users)
