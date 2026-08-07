from pathlib import Path

from agent.config import load_task


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_submission_u55c_manifests_load() -> None:
    for relative in (
        "configs/tasks/atax_u55c.json",
        "configs/tasks/vector_add_track_a.json",
    ):
        task = load_task(REPO_ROOT / relative)
        assert task.data["target"]["part"] == "xcu55c-fsvh2892-2L-e"
        assert task.data["target"]["clock_period_ns"] == 10.0
        assert task.data["adapter"]["kind"] == "autonomous_ppa"
