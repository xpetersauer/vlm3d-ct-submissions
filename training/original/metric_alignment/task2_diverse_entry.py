#!/usr/bin/env python3
"""Forithmus Task-2 entrypoint: diverse generation + existing learned reranker."""

from __future__ import annotations

import os

from task2_diversity import install_generation_patch


def main() -> None:
    install_generation_patch(capture_path=os.getenv("TASK2_DIVERSITY_LOG"))
    import process
    process.main()


if __name__ == "__main__":
    main()
