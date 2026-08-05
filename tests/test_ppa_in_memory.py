from pathlib import Path

from agent.config import load_task
from agent.optimise.runner import run_optimisation


def test_optimisation_accepts_task_without_secondary_config(tmp_path: Path) -> None:
    task = load_task(Path("configs/tasks/vector_add_track_a.json"))
    task.data["output_dir"] = str(tmp_path / "ppa")

    result = run_optimisation(task, status_only=True)

    assert result.success is True
    assert result.status == "status_uninitialised"
    assert "config" not in task.data["adapter"]
    assert not list(tmp_path.rglob("optimisation.json"))
