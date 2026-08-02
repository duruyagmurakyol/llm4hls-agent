# Output storage policy

Every execution writes its complete, disposable record under `runs/`. This
directory is ignored by Git, so verbose model transcripts, compiler output,
temporary workspaces, prompts, and intermediate files do not make commits
noisy.

Use the three locations deliberately:

| Location | Purpose | Git status |
| --- | --- | --- |
| `runs/<workflow>/<experiment>/<timestamp>/` | Full record for one execution: logs, raw responses, validation output, workspace, and the run-level `result.json`. | Ignored |
| `results/` | Compact, machine-readable aggregate results and comparison tables produced from runs. | Tracked when reviewed |
| `evidence/<claim>/` | The minimum source, diff, report excerpt, or validation artifact needed to substantiate a retained result. | Tracked when reviewed |

## What a useful run must retain

Keep the configuration, model and timestamp, exact prompt or API request,
raw model response, command output from every validation stage, final result
JSON, and the before/after source or diff. Existing experiment runners already
retain these files in their run directories.

## Promotion rule

Do not copy every run into Git. After reviewing a run or a group of runs,
commit its aggregate metrics to `results/`. Add material to `evidence/` only
when it is needed to reproduce or defend a specific conclusion; name the
directory for that conclusion and include a short `README.md` describing the
source run and commit.

For large retained run bundles, archive the entire timestamped directory in
external object storage and record its URI and checksum in the relevant
`results/` or `evidence/` entry. The repository remains the index and concise
record, rather than the log archive.
