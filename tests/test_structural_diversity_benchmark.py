import json
from pathlib import Path

from agent.config import load_task


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    REPO_ROOT
    / "configs/tasks/structural_diversity/stream_pipeline_bias_repair_full_agent.json"
)


def test_stream_pipeline_manifest_is_valid_and_targets_u55c():
    task = load_task(MANIFEST)

    assert task.adapter_kind == "auto"
    assert task.data["interface"]["top_function"] == "stream_pipeline"
    assert task.data["target"]["part"] == "xcu55c-fsvh2892-2L-e"
    assert task.data["target"]["minimum_frequency_mhz"] == 100.0
    assert task.data["repair"]["editable_files"] == ["src/stream_pipeline.cpp"]


def test_stream_pipeline_is_structurally_distinct_and_fault_is_observable():
    source = (
        REPO_ROOT
        / "benchmarks/structural_diversity/stream_pipeline/src/stream_pipeline.cpp"
    ).read_text(encoding="utf-8")
    testbench = (
        REPO_ROOT
        / "benchmarks/structural_diversity/stream_pipeline/testbench/stream_pipeline_test.cpp"
    ).read_text(encoding="utf-8")

    assert "#pragma HLS DATAFLOW" in source
    assert source.count("hls::stream<int>") >= 3
    assert "scaled.read() - 3" in source
    assert "input[i] * 2 + 3" in testbench


def test_structural_diversity_index_contains_one_case():
    index = json.loads(
        (
            REPO_ROOT / "configs/tasks/structural_diversity/index.json"
        ).read_text(encoding="utf-8")
    )

    assert index["case_count"] == 1
    assert index["recommended_repetitions"] == 3
    assert index["task_ids"] == ["stream_pipeline_bias_repair_full_agent"]
