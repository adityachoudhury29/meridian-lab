"""Meridian Talent Ranker — recruiter-grade hybrid candidate ranking.

A hybrid (dense + sparse + structured) ranking system that reads a job
description, understands what the role *means* (not just its keywords), and
ranks a 100K candidate pool the way a senior recruiter would.

The package is deliberately split so that the *ranking step* (``rank.py``)
depends only on ``numpy`` + the standard library: the neural embedding model
is used exclusively in the offline ``precompute`` path. This keeps the
reproduced ranking step fast, CPU-only and network-free, per the challenge
compute constraints.
"""

__version__ = "1.0.0"
