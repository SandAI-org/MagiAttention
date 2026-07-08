---
name: debug-test-failures
description: >-
  Systematic approach to debug failing tests. Use when a user reports test
  failures, regression bugs, or assertion mismatches. Core method: reproduce
  the failure, then use git history to determine whether the failure is caused
  by a recent commit or by the user's uncommitted changes.
---

# Debug Test Failures

## Principle

**Test failures always have a cause in code changes.** The fastest debugging
path is to figure out *which change* broke it, not to guess at the logic.

## Step 1: Reproduce and Extract Key Info

Run the failing test, capture full output:

```bash
python -m pytest -sq <test_file> 2>&1 | head -80
python -m pytest -sq <test_file> 2>&1 | tail -50
```

Extract from the error:
- Which test case(s) failed
- The **actual vs expected** values
- The **file and line** of the failing assertion

## Step 2: Determine the Scope of Failure

Check if this test *ever* passed on the current branch:

```bash
# What files have uncommitted changes?
git status

# What committed changes touch files related to the failure?
git log --oneline -20 -- <relevant_source_files>
```

This splits into two cases:

### Case A: The user has uncommitted changes in related files

```bash
git diff -- <relevant_source_files>
```

Read the diff carefully. The bug is likely in the uncommitted changes.
Compare the diff against the assertion error to find the mismatch.

### Case B: No uncommitted changes in related files

The regression was introduced by a recent commit. Proceed to Step 3.

## Step 3: Walk Git History to Find the Offending Commit

```bash
# List recent commits touching the relevant files
git log --oneline --all -- <file_path>
```

Then inspect each suspect commit:

```bash
git show <commit_hash> -- <file_path>
```

Walk commits **chronologically** and identify:
1. **Last known good state** — what the logic looked like before
2. **Offending commit** — where behavior changed
3. **Intent** — was the change a refactor, feature, or bugfix?

Common regression patterns:
- **Refactoring that widens a condition** — e.g. merging two flags into one,
  where the new condition covers more cases than intended
- **Default value changes** — a dataclass/config default was changed, silently
  affecting callers that relied on the old default
- **Silent override in initialization** — `__init__` / `__post_init__` /
  constructor overwrites a user-provided value under a too-broad condition

## Step 4: Confirm Root Cause by Comparing Before/After

Once you identify the suspect commit, compare the old and new logic side by
side. Verify that:
- The old logic would produce the **expected** test output
- The new logic produces the **actual** (wrong) test output
- The behavioral difference is **unintentional** (not a deliberate design change)

## Step 5: Apply Minimal Fix and Verify

1. Make the **smallest change** that restores correct behavior
2. Ensure the original intent of the offending commit is preserved
3. Re-run the failing test to confirm all cases pass

```bash
python -m pytest -sq <test_file> 2>&1 | tail -5
```

## Anti-Patterns

- **Don't update expected values** to match broken output without understanding
  why they differ — the tests encode domain knowledge.
- **Don't guess at the fix** without tracing the cause — always check git
  history first.
- **Don't ignore the "other case"** — if the fix narrows a condition, verify
  that the broader case (which the offending commit intended to handle) still
  works correctly.
