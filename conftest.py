"""Make ``src/tsbp`` importable when running the tests from a source checkout
without ``pip install -e .`` first."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
