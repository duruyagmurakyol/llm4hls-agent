from __future__ import annotations

import json
from pathlib import Path

from agent.optimise.generate import (
    _attach_latency_recovery_factor,
    _generate_controlled_latency_variant,
    _latency_recovery_exhausted_suffix,
)


def test_controlled_variants_clone_one_template_and_change_only_factor(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("agent.optimise.generate.REPO_ROOT", tmp_path)
    output = tmp_path / "out"
    output.mkdir()

    template_source = (
        "void kernel(int a[8]) {\n"
        "  for (int i = 0; i < 8; ++i) {\n"
        "    #pragma HLS PIPELINE II=1\n"
        "    #pragma HLS UNROLL factor=2\n"
        "    a[i] += 1;\n"
        "  }\n"
        "}\n"
    )
    template_strategy = {
        "name": "recover_latency_tradeoff",
        "parameters": {"factor": 2},
        "source_candidate_index": 2,
        "next_candidate_index": 3,
    }
    (output / "candidate_003.cpp").write_text(template_source, encoding="utf-8")
    (output / "candidate_003_strategy.json").write_text(
        json.dumps(template_strategy),
        encoding="utf-8",
    )
    (output / "candidate_003_model_metadata.json").write_text(
        json.dumps({"provider": "siliconflow"}),
        encoding="utf-8",
    )
    (output / "candidate_003_static_validation.json").write_text(
        json.dumps({"passed": True}),
        encoding="utf-8",
    )
    (output / "candidate_003_csim_validation.json").write_text(
        json.dumps({"passed": True}),
        encoding="utf-8",
    )

    for candidate_index, factor in ((4, 4), (5, 8)):
        strategy_path = output / f"candidate_{candidate_index:03d}_strategy.json"
        strategy = {
            "name": "recover_latency_tradeoff",
            "parameters": {"factor": factor},
            "source_candidate_index": 2,
            "next_candidate_index": candidate_index,
        }
        strategy_path.write_text(json.dumps(strategy), encoding="utf-8")

        candidate_path = _generate_controlled_latency_variant(
            output,
            candidate_index,
            strategy_path,
            strategy,
        )

        assert candidate_path is not None
        assert candidate_path.read_text(encoding="utf-8") == template_source.replace(
            "factor=2",
            f"factor={factor}",
        )
        metadata = json.loads(
            (
                output
                / f"candidate_{candidate_index:03d}_deterministic_generation.json"
            ).read_text(encoding="utf-8")
        )
        assert metadata["generation_mode"] == "controlled_parameter_variant"
        assert metadata["model_call"] is False
        assert metadata["template_candidate_index"] == 3
        assert metadata["template_factor"] == 2
        assert metadata["factor"] == factor
        assert not (
            output / f"candidate_{candidate_index:03d}_model_metadata.json"
        ).exists()


def test_unverified_factor_two_source_is_not_reused(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("agent.optimise.generate.REPO_ROOT", tmp_path)
    output = tmp_path / "out"
    output.mkdir()

    (output / "candidate_003.cpp").write_text(
        "void kernel(int a[8]) {\n"
        "  for (int i = 0; i < 8; ++i) {\n"
        "    #pragma HLS UNROLL factor=2\n"
        "    a[i] += 1;\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )
    (output / "candidate_003_strategy.json").write_text(
        json.dumps(
            {
                "name": "recover_latency_tradeoff",
                "parameters": {"factor": 2},
                "source_candidate_index": 2,
                "next_candidate_index": 3,
            }
        ),
        encoding="utf-8",
    )
    (output / "candidate_003_model_metadata.json").write_text(
        json.dumps({"provider": "siliconflow"}),
        encoding="utf-8",
    )
    (output / "candidate_003_static_validation.json").write_text(
        json.dumps({"passed": True}),
        encoding="utf-8",
    )
    (output / "candidate_003_csim_validation.json").write_text(
        json.dumps({"passed": False}),
        encoding="utf-8",
    )

    strategy_path = output / "candidate_004_strategy.json"
    strategy = {
        "name": "recover_latency_tradeoff",
        "parameters": {"factor": 4},
        "source_candidate_index": 2,
        "next_candidate_index": 4,
    }
    strategy_path.write_text(json.dumps(strategy), encoding="utf-8")

    assert (
        _generate_controlled_latency_variant(
            output,
            4,
            strategy_path,
            strategy,
        )
        is None
    )
    assert not (output / "candidate_004.cpp").exists()


def test_exhausted_ladder_retires_active_strategy_file(tmp_path: Path) -> None:
    for candidate_index, factor in ((3, 2), (4, 4), (5, 8)):
        (tmp_path / f"candidate_{candidate_index:03d}_strategy.json").write_text(
            json.dumps(
                {
                    "name": "recover_latency_tradeoff",
                    "parameters": {"factor": factor},
                }
            ),
            encoding="utf-8",
        )

    strategy_path = tmp_path / "candidate_006_strategy.json"
    strategy = {
        "name": "recover_latency_tradeoff",
        "parameters": {"latency_regression_percent": 93.9},
    }
    strategy_path.write_text(json.dumps(strategy), encoding="utf-8")

    exhausted = _attach_latency_recovery_factor(
        tmp_path,
        strategy_path,
        strategy,
    )

    assert exhausted is not None
    assert exhausted["status"] == "exhausted"
    assert exhausted["completed_factors"] == [2, 4, 8]
    assert "factor" not in exhausted["parameters"]
    assert not strategy_path.exists()

    exhausted_path = tmp_path / "candidate_006_strategy_exhausted.json"
    persisted = json.loads(exhausted_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "exhausted"
    assert persisted["completed_factors"] == [2, 4, 8]


def test_exhausted_recovery_has_explicit_generic_fallback_instruction() -> None:
    suffix = _latency_recovery_exhausted_suffix(True)

    assert "Factors 2, 4, and 8 have already been evaluated" in suffix
    assert "Ignore the recover_latency_tradeoff strategy text" in suffix
    assert "without a mandated unroll factor" in suffix
