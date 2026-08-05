import configparser
import json
from pathlib import Path

from scripts.prepare_u55c_validation_subset import (
    U55C_PART,
    U55C_RESOURCE_LIMITS,
    prepare_subset,
)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_generates_target_specific_manifest_cfg_and_index(tmp_path):
    repo = tmp_path / "repo"
    benchmark = repo / "benchmarks" / "example"
    source = benchmark / "src" / "example.cpp"
    header = benchmark / "src" / "example.h"
    testbench = benchmark / "testbench" / "example_tb.cpp"
    cfg = benchmark / "task.cfg"

    source.parent.mkdir(parents=True)
    testbench.parent.mkdir(parents=True)
    source.write_text("void example(int &x) { x = 1; }\n", encoding="utf-8")
    header.write_text("void example(int &x);\n", encoding="utf-8")
    testbench.write_text("int main() { return 0; }\n", encoding="utf-8")
    cfg.write_text(
        """[hls]
flow_target=vivado
syn.file=src/example.cpp
syn.cflags=-Isrc
syn.top=example
tb.file=testbench/example_tb.cpp
tb.cflags=-Isrc
part=xczu3eg-sfvc784-2-e
clock=10ns
""",
        encoding="utf-8",
    )

    base_manifest = repo / "configs/tasks/combined_full_agent/example.json"
    write_json(
        base_manifest,
        {
            "task_id": "example_repair_full_agent",
            "artifacts": {
                "source": "benchmarks/example/src/example.cpp",
                "testbench": ["benchmarks/example/testbench/example_tb.cpp"],
                "build_files": ["benchmarks/example/task.cfg"],
            },
            "interface": {"top_function": "example"},
            "target": {},
            "output_dir": "runs/example",
            "optimisation": {
                "prompt_constraints": [
                    "Treat 100 MHz and the configured FPGA resource limits as hard constraints.",
                    "Preserve behaviour.",
                ]
            },
        },
    )

    output_root = repo / "configs/tasks/u55c_validation"
    index_path = prepare_subset(
        repo_root=repo,
        base_manifests=[Path("configs/tasks/combined_full_agent/example.json")],
        output_root=output_root,
    )

    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["case_count"] == 1
    assert index["target"]["part"] == U55C_PART
    assert index["target"]["resource_limits"] == U55C_RESOURCE_LIMITS

    generated_manifest_path = repo / index["cases"][0]
    generated = json.loads(generated_manifest_path.read_text(encoding="utf-8"))
    assert generated["task_id"] == "example_repair_full_agent_u55c"
    assert generated["target"]["part"] == U55C_PART
    assert generated["target"]["resource_limits"] == U55C_RESOURCE_LIMITS
    assert generated["parent_task_id"] == "example_repair_full_agent"
    assert generated["output_dir"].endswith("example_repair_full_agent_u55c")
    assert any(
        U55C_PART in item
        for item in generated["optimisation"]["prompt_constraints"]
    )

    generated_cfg = repo / generated["artifacts"]["build_files"][0]
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(generated_cfg, encoding="utf-8")
    hls = parser["hls"]
    assert hls["part"] == U55C_PART
    assert hls["clock"] == "10ns"
    assert (generated_cfg.parent / hls["syn.file"]).resolve() == source.resolve()
    assert (generated_cfg.parent / hls["tb.file"]).resolve() == testbench.resolve()

    syn_include = hls["syn.cflags"].removeprefix("-I")
    tb_include = hls["tb.cflags"].removeprefix("-I")
    assert (generated_cfg.parent / syn_include).resolve() == (benchmark / "src").resolve()
    assert (generated_cfg.parent / tb_include).resolve() == (benchmark / "src").resolve()
