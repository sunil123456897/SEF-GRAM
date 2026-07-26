# Colab Bridge jobs

The Colab worker watches `jobs/pending/*.json` on branch `infra/colab-bridge`.

Example:

```json
{
  "version": 1,
  "id": "test-run-001",
  "ref": "main",
  "entrypoint": "experiments/run_step6_final_eval.py",
  "args": [],
  "env": {},
  "pip_packages": [],
  "timeout_seconds": 3600,
  "result_files": ["results/**", "metrics.json"]
}
```

Lifecycle:

```
jobs/pending/job.json
        |
        v
jobs/running/job.json
        |
        v
jobs/completed/job.json
or
jobs/failed/job.json
```

Results are written to:

```
results/<job-id>/summary.json
results/<job-id>/stdout.txt
results/<job-id>/stderr.txt
results/<job-id>/artifacts/*
```
