"""Script support package; ``run_agent.py`` remains the public CLI."""

from importlib import import_module
import sys

# Historical tests and archived analysis import the prompt-evidence helper from
# its former launcher path.  Keep that import stable without restoring the
# obsolete top-level executable.
run_ppa_optimisation = import_module("scripts.maintenance.ppa_prompt_evidence")
sys.modules[f"{__name__}.run_ppa_optimisation"] = run_ppa_optimisation

__all__ = ["run_ppa_optimisation"]
