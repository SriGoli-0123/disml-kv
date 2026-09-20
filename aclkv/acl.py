"""Access-control model.

A document ACL is a *conjunction* of atoms.  Each atom names an audience:

- ``public``      every authenticated user
- ``group:<g>``   members of group ``g``
- ``user:<u>``    exactly user ``u``

The KV state of a token depends on every earlier token, so the effective ACL
of a prompt prefix ``P_k = D_1 .. D_k`` is the intersection of the audiences
of all documents in it (challenge C1 in the proposal):

    ACL(P_k) = ACL(D_1) ∩ ... ∩ ACL(D_k)

Intersection of audiences == union of the atom constraints, which is what
``ACL.intersect`` computes.  ``ACL.canonical()`` gives a stable string used to
derive a server-keyed scope id (see ``scope.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

PUBLIC_ATOM = "public"
GROUP_PREFIX = "group:"
USER_PREFIX = "user:"

# Breadth classes used by access-aware ordering: lower == broader audience.
BREADTH_PUBLIC = 0
BREADTH_GROUP = 1
BREADTH_USER = 2
BREADTH_EMPTY = 3


@dataclass(frozen=True)
class Directory:
    """Users, groups and memberships (the authoritative identity source)."""

    users: tuple[str, ...]
    groups: tuple[str, ...]
    membership: Mapping[str, frozenset[str]] = field(default_factory=dict)

    def groups_of(self, user: str) -> frozenset[str]:
        return self.membership.get(user, frozenset())

    def is_member(self, user: str, group: str) -> bool:
        return group in self.groups_of(user)

    def members(self, group: str) -> frozenset[str]:
        return frozenset(u for u in self.users if group in self.groups_of(u))

    def to_json(self) -> dict:
        return {
            "users": list(self.users),
            "groups": list(self.groups),
            "membership": {u: sorted(g) for u, g in self.membership.items()},
        }

    @classmethod
    def from_json(cls, d: Mapping) -> "Directory":
        return cls(
            users=tuple(d["users"]),
            groups=tuple(d["groups"]),
            membership={u: frozenset(g) for u, g in d["membership"].items()},
        )


@dataclass(frozen=True)
class ACL:
    """A conjunction of ACL atoms, kept in canonical form.

    Canonicalisation rules:
    - ``public`` is the identity; it is dropped when any other atom exists.
    - a single ``user:u`` atom subsumes every group atom *for the purpose of
      the audience* (the audience is either ``{u}`` or empty) — we keep the
      user atom only, so ``user:u & group:g`` canonicalises to ``user:u``.
      This is safe because a requester that passed the authorization filter
      is ``u`` and is a member of ``g``; if not, the audience is empty and
      nothing can be reused anyway.
    - two different user atoms have an empty audience → canonical ``empty``.
    """

    atoms: frozenset[str]

    # ------------------------------------------------------------ parsing
    @staticmethod
    def parse(text: str) -> "ACL":
        """Parse ``"public"``, ``"group:hr"``, ``"user:alice"`` or ``"a&b"``."""
        parts = [p.strip() for p in text.split("&") if p.strip()]
        for p in parts:
            if not (p == PUBLIC_ATOM or p.startswith(GROUP_PREFIX) or p.startswith(USER_PREFIX)):
                raise ValueError(f"unknown ACL atom {p!r}")
        return ACL.of(parts)

    @staticmethod
    def of(atoms: Iterable[str]) -> "ACL":
        atoms = set(atoms)
        users = {a for a in atoms if a.startswith(USER_PREFIX)}
        groups = {a for a in atoms if a.startswith(GROUP_PREFIX)}
        if len(users) > 1:
            return EMPTY
        if len(users) == 1:
            return ACL(frozenset(users))
        if groups:
            return ACL(frozenset(groups))
        return PUBLIC

    @staticmethod
    def public() -> "ACL":
        return PUBLIC

    # ---------------------------------------------------------- algebra
    def intersect(self, other: "ACL") -> "ACL":
        """Effective ACL of a prefix that contains both ``self`` and ``other``."""
        if self.is_empty or other.is_empty:
            return EMPTY
        return ACL.of(self.atoms | other.atoms)

    # --------------------------------------------------------- queries
    @property
    def is_public(self) -> bool:
        return self.atoms == frozenset({PUBLIC_ATOM})

    @property
    def is_empty(self) -> bool:
        return len(self.atoms) == 0

    @property
    def user(self) -> str | None:
        for a in self.atoms:
            if a.startswith(USER_PREFIX):
                return a[len(USER_PREFIX):]
        return None

    @property
    def groups(self) -> frozenset[str]:
        return frozenset(a[len(GROUP_PREFIX):] for a in self.atoms if a.startswith(GROUP_PREFIX))

    @property
    def breadth(self) -> int:
        if self.is_empty:
            return BREADTH_EMPTY
        if self.is_public:
            return BREADTH_PUBLIC
        if self.user is not None:
            return BREADTH_USER
        return BREADTH_GROUP

    def allows(self, user: str, directory: Directory) -> bool:
        """Cache-hit gate: may ``user`` reuse state whose effective ACL is ``self``?"""
        if self.is_empty:
            return False
        if self.is_public:
            return True
        u = self.user
        if u is not None:
            return u == user
        return all(directory.is_member(user, g) for g in self.groups)

    def audience(self, directory: Directory) -> frozenset[str]:
        return frozenset(u for u in directory.users if self.allows(u, directory))

    def canonical(self) -> str:
        if self.is_empty:
            return "empty"
        return "&".join(sorted(self.atoms))

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.canonical()


PUBLIC = ACL(frozenset({PUBLIC_ATOM}))
EMPTY = ACL(frozenset())


def effective_prefix_acls(doc_acls: Iterable[ACL], start: ACL = PUBLIC) -> list[ACL]:
    """Running intersection ``ACL(P_k)`` for k = 1..n (challenge C1)."""
    out: list[ACL] = []
    cur = start
    for a in doc_acls:
        cur = cur.intersect(a)
        out.append(cur)
    return out


def authorization_filter(docs, user: str, directory: Directory):
    """Step 1 of the middleware: drop chunks the requester cannot read.

    ``docs`` is any iterable of objects exposing an ``acl: ACL`` attribute.
    Returns ``(kept, dropped)``.
    """
    kept, dropped = [], []
    for d in docs:
        (kept if d.acl.allows(user, directory) else dropped).append(d)
    return kept, dropped
