"""Loads ../idm-vton/inference.py as a module, without relying on `import inference`.

Both idm-vton/ and fitcontroler/ (this directory) have their own inference.py. If both
directories end up on sys.path at once (which train.py and inference.py both used to
do), plain `import inference` silently resolves to whichever one sys.path lists first --
not necessarily idm-vton's. This was a real bug: fitcontroler/inference.py's own
sys.path setup put itself ahead of idm-vton/, so `import inference` inside it was
grabbing itself, and `idm_vton_inference.IDMVTONPipeline` failed with an AttributeError.
Loading idm-vton/inference.py by its explicit file path sidesteps the ambiguity, and
registering it in sys.modules under a private name means it's only ever loaded once.
"""
import importlib.util
import sys
from pathlib import Path

IDM_VTON_INFERENCE_PATH = Path(__file__).resolve().parent.parent / "idm-vton" / "inference.py"
_MODULE_NAME = "_idm_vton_inference_impl"


def load():
    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]
    if not IDM_VTON_INFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"{IDM_VTON_INFERENCE_PATH} not found -- set up ../idm-vton first (see its README)"
        )
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, IDM_VTON_INFERENCE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module
