"""Test bootstrap that is independent of how the pytest executable is launched.

On Windows, the ``pytest.exe`` entry point can put its Scripts directory (rather
than the current project) at ``sys.path[0]``.  Some IDEs and wrapper scripts also
override the ``pythonpath`` value from pytest.ini.  Add the repository root
before test modules are imported so the application's existing top-level module
layout remains importable in all of those cases.
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root = str(PROJECT_ROOT)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

