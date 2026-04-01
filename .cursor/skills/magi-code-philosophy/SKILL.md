---
name: magi-code-philosophy
description: >-
  Core engineering philosophies for the Magi Attention codebase. Use when
  writing, reviewing, or modifying any code in this project. Covers config
  consistency, static-analysis-friendly readability, and test coverage
  requirements. Any deviation from these philosophies must be strictly
  commented with justification.
---

# Magi Code Philosophy

These are the non-negotiable engineering principles for this codebase.
Every contributor — human or AI — must follow them. When a principle
cannot be followed, a **DEVIATION comment** is required (see bottom).

---

## Philosophy 1: Config Consistency

> **A config field's value should mean what the user set it to.**

If internal logic must transform, override, or reinterpret a user-supplied
value, this is a deviation that requires explicit justification.

### Rules

1. **Preserve user intent** — When a user sets `field=X`, reading `obj.field`
   should return `X` or something recognizably equivalent. If the value must
   be normalized, store the original intent in a private field or property
   before overwriting.

2. **Validate early, normalize minimally** — `__post_init__` should primarily
   assert invariants. Normalization should be the smallest necessary
   adjustment. Never silently clamp or discard user input.

3. **Derived fields ≠ overwritten fields** — Values computed from other
   fields should be **new** fields, not overwrites of user input. Exception:
   sentinel values (e.g., `-1` = "auto-detect") are designed to be replaced,
   but still require a comment.

4. **In-place mutation must be documented** — If `__post_init__` or a helper
   mutates a nested structure (e.g., `num_tokens *= 2` for packed KV), the
   site must comment: what is mutated, why, and how to recover the original.

5. **Forced override needs justification** — When code forces a field value
   from another field (e.g., `deterministic |= reduce_op != "sum"`), explain
   why the user's choice is being overridden.

### Deviation Format (Config)

```python
# DEVIATION: <one-line summary>
# Reason: <why the user-facing value cannot be kept as-is>
# Recovery: <how to access original intent, or "none">
```

---

## Philosophy 2: Readability via Static Navigability

> **Every symbol in the code must be statically resolvable and jump-to-able.**

Code is read far more often than written. The reader should be able to
Ctrl+Click (or equivalent) on any name and land on its definition. If the
IDE's static analysis cannot resolve a symbol, the code is not readable
enough.

### Rules

1. **Explicit imports over dynamic lookups** — Use direct imports, not
   `getattr(module, name)` or `globals()[name]`. If dynamic dispatch is
   truly needed, use a typed registry/dict with the concrete types visible
   at the registration site.

2. **Typed dicts and enums over magic strings** — Prefer `Enum` members
   and `TypedDict` keys over raw string literals. Strings are invisible to
   static analysis; enums and typed keys are jump-to-able.

3. **No untyped `**kwargs` pass-through in public APIs** — Public-facing
   functions should declare their parameters explicitly. `**kwargs` may be
   used internally (e.g., forwarding to a backend), but the public signature
   must be self-documenting.

4. **Avoid deep `Any` typing** — `Any` kills jump-to-definition. Use
   `Protocol`, generics, or union types. Reserve `Any` for truly
   polymorphic boundaries (e.g., serialization).

5. **String-based dispatch must have a central map** — If behavior branches
   on a string value, define the mapping in one place (a dict or match/case)
   so that all targets are visible together and searchable.

6. **Re-exports must be explicit** — When `__init__.py` re-exports symbols,
   use explicit `from .module import Name` rather than `import module` with
   `__all__`. This ensures the IDE can resolve the re-exported name.

### Deviation Format (Readability)

```python
# DEVIATION: <what is not statically resolvable>
# Reason: <why dynamic dispatch / Any / kwargs is unavoidable here>
# Mitigation: <how a reader can still find the target, e.g., "see registry at X">
```

---

## Philosophy 3: Test Completeness

> **Where there is code, there must be tests.**

No feature, bug fix, refactor, or config change is considered done until
it has corresponding test coverage. Untested code is assumed broken.

### Rules

1. **Every public function/class has a test** — If it's importable from
   outside its module, it needs at least one test exercising its primary
   path and one test for its most important edge case.

2. **Config normalization must be tested** — For every `__post_init__`
   normalization or deviation, there must be a test that:
   - Sets the user-facing value
   - Asserts the normalized internal value
   - Asserts the original intent is recoverable (if applicable)

3. **Bug fixes come with regression tests** — The test must reproduce the
   original failure first (red), then pass with the fix (green).

4. **Solver/algorithm changes need correctness tests** — Any change to a
   solver (`overlap_solver`, `dispatch_solver`, `dist_attn_solver`, etc.)
   must include tests that verify the output solution against known-good
   reference values.

5. **Test names describe the scenario** — Use descriptive names like
   `test_overlap_config_degree_zero_normalizes_to_one`, not `test_config_1`.
   The name should read as a specification.

6. **No test-only code in production modules** — Test helpers, fixtures, and
   mocks live in `tests/`. Production code should not contain `if TESTING:`
   branches or similar.

### Deviation Format (Test)

```
# DEVIATION: <what is not tested>
# Reason: <why testing is impractical, e.g., requires multi-GPU hardware>
# Tracking: <issue/TODO reference for future coverage>
```

---

## General Deviation Protocol

When **any** philosophy cannot be followed, add a structured comment at the
deviation site. The format depends on the philosophy (see each section
above). The key invariant is:

> **Silence is not acceptable. If the code deviates, the code says so.**

Reviewers (human or AI) should flag any deviation that lacks a comment as a
blocking issue.

---

## Quick Checklist

Before submitting code, verify:

**Config Consistency**
- [ ] Every `__post_init__` field overwrite has a DEVIATION comment
- [ ] User intent is recoverable via private field or property
- [ ] Sentinel values are documented in the class docstring

**Readability**
- [ ] All symbols are Ctrl+Click navigable (no unresolvable dynamic lookups)
- [ ] Public APIs have explicit typed signatures (no bare `**kwargs`)
- [ ] String-based dispatch has a central, visible mapping

**Tests**
- [ ] Every new/changed public API has corresponding tests
- [ ] Config normalizations have dedicated test cases
- [ ] Bug fixes include a regression test
- [ ] Test names describe the scenario, not just a number
