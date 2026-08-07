"""Script support package; ``run_agent.py`` remains the public CLI."""

from importlib import import_module
import sys


def _alias(name: str, target: str):
    module = import_module(target)
    sys.modules[f"{__name__}.{name}"] = module
    globals()[name] = module
    return module


# Preserve imports used by regression tests and archived analysis without
# restoring obsolete/maintenance executables to the top-level scripts folder.
run_ppa_optimisation = _alias(
    "run_ppa_optimisation",
    "scripts.maintenance.ppa_prompt_evidence",
)
prepare_u55c_validation_subset = _alias(
    "prepare_u55c_validation_subset",
    "scripts.maintenance.prepare_u55c_validation_subset",
)

__all__ = [
    "prepare_u55c_validation_subset",
    "run_ppa_optimisation",
]
