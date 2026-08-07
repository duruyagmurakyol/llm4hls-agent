from agent.optimise.generate import minimal_edit_prompt_suffix


def test_output_contract_requires_material_change() -> None:
    prompt = minimal_edit_prompt_suffix()
    assert "return the original source unchanged" not in prompt
    assert "must contain at least one executable or HLS-directive change" in prompt
    assert "do not return the implementation parent unchanged" in prompt
