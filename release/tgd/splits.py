"""Deterministic question-level split assignment.

Every question id is hashed once (salted) into a held-out test pool or the trainable
pool. Hashing by qid rather than by row means a question can never end up on both
sides through a second episode, and the assignment is identical on every rebuild and on
every machine.
"""
from __future__ import annotations

import hashlib

DEFAULT_SALT = "m1lodo"          # the salt behind the published results
DEFAULT_DEV_SALT = "m1lodo-dev"


def _bucket(qid: str, salt: str) -> int:
    return int(hashlib.sha256(f"{salt}:{qid}".encode()).hexdigest(), 16) % 10_000


def pool_of(qid: str, heldout_fraction: float = 0.10, salt: str = DEFAULT_SALT) -> str:
    """``"heldout_test"`` for the first ``heldout_fraction`` of hash space, else ``"trainable"``."""
    return "heldout_test" if _bucket(qid, salt) < heldout_fraction * 10_000 else "trainable"


def is_dev(qid: str, dev_fraction: float = 0.03, salt: str = DEFAULT_DEV_SALT) -> bool:
    """Small dev slice carved out of the trainable pool (loss monitoring only)."""
    return _bucket(qid, salt) < dev_fraction * 10_000
