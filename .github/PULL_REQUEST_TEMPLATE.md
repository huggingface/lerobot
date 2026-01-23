## Title

Short, imperative summary (e.g., "fix(robots): handle None in sensor parser"). See [CONTRIBUTING.md](../CONTRIBUTING.md) for PR conventions.

## Type / Scope

- **Type**: (Bug | Feature | Docs | Performance | Test | CI | Chore)
- **Scope**: (optional — name of module or package affected)

## Summary / Motivation

- One-paragraph description of what changes and why.
- Why this change is needed and any trade-offs or design notes.

## Related issues

- Fixes / Closes: # (if any)
- Related: # (if any)

## What changed

- Short, concrete bullets of the modifications (files/behaviour).
- Short note if this introduces breaking changes and migration steps.

## How was this tested (or how to run locally)

- Tests added: list new tests or test files.
- Manual checks / dataset runs performed.
- Instructions for the reviewer

Example:

- Ran the relevant tests:

  ```bash
  pytest -q tests/ -k <keyword>
  ```

- Reproduce with a quick example or CLI (if applicable):

  ```bash
  lerobot-train --some.option=true
  ```

## Checklist (required before merge)

- [ ] Linting/formatting run (`pre-commit run -a`)
- [ ] All tests pass locally (`pytest`)
- [ ] Documentation updated
- [ ] CI is green

## Reviewer notes

- Anything the reviewer should focus on (performance, edge-cases, specific files) or general notes.
- Anyone in the community is free to review the PR.
