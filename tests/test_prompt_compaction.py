from agent.prompt_compaction import compact_user_prompt


def test_drops_redundant_baseline_and_keeps_parent():
    prompt = """Constraints:
- Preserve behaviour.

Previous candidate source:
// parent comment
int top() { return 1; }

Original baseline source:
int top() { return 0; }
"""
    compacted, stats = compact_user_prompt(prompt)
    assert "return 1" in compacted
    assert "return 0" not in compacted
    assert "parent comment" not in compacted
    assert stats["dropped_redundant_baseline_source"] is True


def test_keeps_baseline_when_it_is_only_source():
    prompt = "Baseline source:\n// explanation\nint top() { return 0; }\n"
    compacted, stats = compact_user_prompt(prompt)
    assert "return 0" in compacted
    assert "explanation" not in compacted
    assert stats["dropped_redundant_baseline_source"] is False


def test_preserves_comment_tokens_inside_strings_and_raw_strings():
    prompt = '''Previous candidate source:
const char *a = "// keep"; // remove
const char *b = R"tag(/* keep */)tag";
/* remove
   remove */
int top() { return 1; }
'''
    compacted, _ = compact_user_prompt(prompt)
    assert '"// keep"' in compacted
    assert 'R"tag(/* keep */)tag"' in compacted
    assert "// remove" not in compacted
    assert "remove */" not in compacted


def test_compacts_metric_aliases_and_equal_ranges():
    prompt = """Baseline metrics:
- clock_period_ns: 5.0
- latency_best_cycles: 10
- latency_average_cycles: 10
- latency_worst_cycles: 10
- interval_min_cycles: 8
- interval_max_cycles: 8
- resources_lut_used: 128
"""
    compacted, stats = compact_user_prompt(prompt)
    assert "clk_ns=5.0" in compacted
    assert "lat_cycles=10" in compacted
    assert "lat_avg_cycles" not in compacted
    assert "ii_cycles=8" in compacted
    assert "lut=128" in compacted
    assert stats["metric_blocks_compacted"] == 1


def test_compacts_only_editable_fence_not_context_fence():
    prompt = """EDITABLE FILE: src/a.cpp
```
// remove
int top() { return 1; }
```

FILE: test.cpp
```
// semantic context must remain
int main() {}
```
"""
    compacted, _ = compact_user_prompt(prompt)
    assert "// remove" not in compacted
    assert "semantic context must remain" in compacted


def test_deduplicates_exact_instruction_lines_and_is_idempotent():
    prompt = """Required:
- Preserve behaviour.
Other:
- Preserve behaviour.
- Keep bounds safe.
"""
    compacted, stats = compact_user_prompt(prompt)
    compacted_twice, _ = compact_user_prompt(compacted)
    assert compacted.count("Preserve behaviour") == 1
    assert stats["duplicate_instruction_lines_removed"] == 1
    assert compacted_twice == compacted
