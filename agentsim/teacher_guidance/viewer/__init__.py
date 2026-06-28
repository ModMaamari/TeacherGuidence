"""
Local web UI for exploring Teacher Guidance trajectories.

Reads the clean ``teacher_guidance_episodes.jsonl`` files produced by a run and serves
a small, dependency-free explorer (stdlib HTTP server + vanilla frontend).
"""

__all__ = ["data_access", "server"]
