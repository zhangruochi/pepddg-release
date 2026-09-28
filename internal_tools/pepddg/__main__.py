"""Module entry point for PepDDG.

Allows agents and shell users to run:

    python -m internal_tools.pepddg --config ...
"""

from __future__ import annotations

from .run_pepddg import main


if __name__ == "__main__":
    raise SystemExit(main())
