# Repair Attempt Artefact Contract

Schema version: 1

Every direct-API repair run uses one run directory containing `repair_attempts.json`, `budget_summary.json`, `result.json`, and one directory per attempted repair:

```text
<run>/
├── attempt_001/
├── attempt_002/
├── budget_summary.json
├── repair_attempts.json
└── result.json
```

Each `attempt_NNN/` directory always contains:

```text
diagnosis.json
system_prompt.txt
prompt.txt
raw_response.txt
candidate.cpp
diff.patch
strategy.json
validation.json
token_usage.json
result.json
```

## File meanings

- `diagnosis.json`: initial, final, and active structured diagnosis for the attempt.
- `system_prompt.txt`: exact system guidance supplied to the repair model.
- `prompt.txt`: exact user prompt, including source, context, diagnosis, and retry evidence.
- `raw_response.txt`: exact provider response. It is empty when the provider fails before returning content.
- `candidate.cpp`: source retained after the attempt. `result.json` distinguishes an accepted model candidate from the previous last-valid source.
- `diff.patch`: unified source diff. It is empty when no model candidate was accepted.
- `strategy.json`: deterministic description and fingerprint of the effective source edit.
- `validation.json`: explicit `passed`, `failed`, or `not_run` records for generation, output validation, scope checks, host validation, and independent CSim.
- `token_usage.json`: model, provider, token counts, latency, and whether usage was charged to the task budget.
- `result.json`: attempt summary and index of the canonical artefacts.

Detailed legacy logs such as `host_validation_before.log`, `host_validation_after.log`, `independent_validation.log`, `api_response.json`, and `output_validation.json` may also be present. They are debugging evidence rather than the stable analysis interface.

## Candidate provenance

`candidate.cpp` always exists so analysis tools do not need outcome-specific directory logic.

- `candidate_record_kind: generated_candidate` means the model output passed pre-write validation and became the attempted candidate.
- `candidate_record_kind: last_valid_source` means provider failure or output rejection prevented a model candidate from being accepted; the file contains the preserved source from before that attempt.

## Validation status

Every validation stage records:

- `run`: whether the stage executed;
- `status`: `passed`, `failed`, or `not_run`;
- `passed`: boolean when run, otherwise `null`;
- optional reason, failure class, evidence, and log path.

A stage that was not reached must never be represented as a failed stage.

## Run-level budget record

`budget_summary.json` contains:

- initial limits;
- consumed and remaining resources;
- token totals;
- raw budget events;
- usage aggregated by stage and by repair phase;
- stop reason and run termination reason;
- attempt count and final repair outcome.

The authoritative accounting source remains `BudgetState`; the JSON record is a serialised report of that state.
