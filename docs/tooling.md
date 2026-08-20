# Tooling quality gates

This repository is one maintained Python project. Tooling is selected by actual repository need rather than by installing every portfolio-wide option.

## Canonical stack

| Responsibility | Tool | Repository contract |
| --- | --- | --- |
| Dependency authority | `uv` | `pyproject.toml` + committed `uv.lock`; `uv lock --check` rejects drift. |
| Runtime requirements for Docker/pip consumers | `uv export` | `requirements.txt` is a frozen generated export from the committed lock, not a second dependency authority. |
| Formatter/linter | Ruff | Existing Ruff configuration is reused; `ruff format --check` and `ruff check` are blocking. |
| Static typing | Pyrefly | `pyrefly check` is blocking. Existing pre-adoption debt is recorded in `.pyrefly-baseline.json`; new errors are not added implicitly. |
| Runtime boundary validation | Pydantic | Used at the untrusted `sources/videos.json` JSON boundary. Internal Python dictionaries are not duplicated into Pydantic models without a trust-boundary reason. |
| Pre-commit runner | `prek` | Built-in whitespace/JSON/YAML checks plus the same `scripts/check-fast` used by CI. |
| Local/CI command | `scripts/check` | `scripts/check-fast` plus the full unittest suite. |

## Local commands

```bash
uv sync --locked
bash scripts/check-fast
bash scripts/check
uv run prek run --all-files
```

`scripts/check-fast` verifies lock drift, the frozen runtime export, Ruff format/lint, Pyrefly, `task` shell syntax, and Python compilation. `scripts/check` then runs the unittest suite.

## Deliberately not installed

The repository has no maintained TypeScript/JavaScript workspace, so Biome, Oxlint, `tsc --noEmit`, and Zod are not applicable. It is not a genuine multi-project monorepo, so Nx is not applicable. Adding these tools would create dependency and CI cost without validating maintained source.

BasedPyright was previously configured but not executed by CI. It is removed rather than running a second Python type checker in parallel; Pyrefly is the one blocking type gate.

## CI timing evidence

Before this migration, Test workflow run `32333750513` completed in approximately 30 seconds based on its GitHub Actions timestamps. The pull request that introduces this stack records the final locked Test run so future changes can compare against the same workflow rather than an estimated local benchmark.

## Baseline policy

`.pyrefly-baseline.json` is a migration boundary, not an ignore-all configuration. It contains the pre-existing errors observed when Pyrefly was first enabled. If code changes make an entry obsolete, remove it from the baseline. Do not update the baseline merely to make a new error pass.
