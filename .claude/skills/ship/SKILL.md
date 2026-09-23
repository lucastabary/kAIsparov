---
name: ship
description: Take the current changes to main the kAIsparov way — a short-lived branch, Conventional Commits split by logical change, docs updated in the same commit, a fast-forward merge, and a push only after the user confirms.
argument-hint: "[what the change is about, or a commit subject]"
disable-model-invocation: true
---

# /ship — from working tree to main

Hint from the user, if any: `$ARGUMENTS`

The rules are in `CLAUDE.md` (sections **Git** and **Claude Code setup**); this is the
order to apply them in. Stop and ask whenever a step is ambiguous: a wrong commit on
main is expensive, a question is not.

## 1. Look before touching anything

- `git status`, `git diff HEAD --stat`, `git log --oneline -5`, current branch.
- Nothing to ship: say so and stop. Never stage `runs/`, `data/`, `*.pth`,
  `credentials.txt` or `.env*` — they are git-ignored on purpose.

## 2. Branch

On `main`, create `<type>/<short-slug>` (`feat/…`, `fix/…`, `perf/…`, `docs/…`,
`tooling/…`) named after the change. Already on a work branch: stay on it.

## 3. Docs are part of the change

For each change, check what it makes wrong: `README.md`, `docs.md`,
`config/README.md`, the model's `README.md`, `CLAUDE.md`. Fix them **in the same
commit**. Add a `CHANGELOG.md` entry under `[Unreleased]` only for a change someone
would need explained (*what it means*, not what changed).

## 4. Commit, one logical change at a time

- Split unrelated changes into separate commits (stage by path; never `git add -i`).
- Subject: `type(scope): imperative subject`, lowercase, concise; the scope is the
  package touched (`core`, `model`, `training`, `bench`, `tracking`, …). A breaking
  change gets `!` (`feat(tracking)!: …`).
- A body explaining *why* when the subject does not make it obvious.
- End the message with the co-author trailer from your current instructions.
- The commit hook runs ruff, mypy, lint-imports and pytest (~1.5 min) and blocks the
  commit if one fails: fix the cause and commit again. Never `--no-verify`.

## 5. Merge into main, fast-forward only

```bash
git checkout main
git merge --ff-only <branch>
```

If `main` moved on meanwhile, go back to the branch, `git rebase main`, rerun the
checks, then merge. Never create a merge commit unless the user asks for one.

## 6. Push, only with a yes

Show the user what will leave the machine: `git log --oneline origin/main..main`.
Push (`git push origin main`) only after an explicit yes. Then delete the merged
branch locally (`git branch -d <branch>`), and report the CI run: `gh run list -L 1`,
then `gh run watch <id> --exit-status` if they want to wait for it.

A release is a separate decision (see **Releases** in `CLAUDE.md`): never tag unasked.
