"""Compatibility wrapper for the task-labeled object detection module."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

_MODULE_PATH = Path(__file__).with_name("object_detection_tracking tk 4 .py")
_SPEC = spec_from_file_location("_object_detection_tracking_task", _MODULE_PATH)

if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load object detection module from {_MODULE_PATH}")

_MODULE = module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

globals().update(
    {
        name: value
        for name, value in vars(_MODULE).items()
        if not name.startswith("__")
    }
)

if __name__ == "__main__":
    _MODULE.main()
