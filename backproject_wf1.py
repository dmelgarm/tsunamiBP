#!/usr/bin/env python
"""Deprecated shim.

The single-file script was split into the ``tsbp`` package (``src/tsbp/``).
This thin wrapper preserves the old entry point so existing commands keep
working:

    python backproject_wf1.py [args...]      ==  python -m tsbp [args...]

Prefer ``python -m tsbp`` (or the installed ``tsbp`` command) going forward.
"""
import os
import sys

# Allow running from a source checkout without installing: put src/ on the path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from tsbp.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
