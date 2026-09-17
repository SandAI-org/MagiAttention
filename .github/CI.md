# MagiAttention CI

Repository-level CI resources are grouped by role: workflows in `.github/workflows/`,
executable helpers in `.github/scripts/`, helper tests in `.github/tests/`, and
versioned protocol inputs in `.github/configs/`.

## Declarative CI inputs

`.github/configs/ci_input_policy.json` is the single source of truth for package input
selection. Each node declares a repository-layout-independent `root` and
separate `trigger`, `wheel`, and `portable` exclusions, all relative to that
root. To exclude a path from CI, edit only those exclusion lists; do not add a
second path list to the workflow or digest scripts. Validation fails closed if
a wheel or portable input is not covered by the trigger layer. The canonical
policy projection participates in source identity, while policy/helper changes
also invalidate the CI protocol.

## Runtime and runner

GPU CI uses task runners labeled `ci-spot-job` plus the platform label declared in `.github/configs/ci_platforms.json`. H100 builds one wheel with `MAGI_ATTENTION_BUILD_COMPUTE_CAPABILITY=90,100`, so both `magi_attn_ext` and `magi_attn_comm` contain SM90 and SM100 code. Every enabled platform installs those exact wheel bytes. Set the Actions variable `B300_CI_ENABLED=true` only after the B300 runner is available; otherwise the planner omits B300 entirely.

The package-owned H100 profile tests the canonical backend names `ffa` and `sdpa`, preserving the existing Hopper coverage. The B300 profile instead tests `fa4`, `cutedsl`, and `sdpa`; `fa4` is the test enum value for the FFA_FA4 implementation. The H100 build performs the normal FFA AOT prebuild once, and B300 remains install-only. Unsupported parameter combinations are filtered by the test suite. The H100 profile alone retains `NCCL_NVLS_ENABLE=0`; B300 does not inherit that Hopper-specific workaround. The Dispatcher creates each task directly from `registry.cn-sh-01.sensecore.cn/sandai-ccr/magi-base:26.05.2`, matching `.github/configs/base_image_tag.txt`; the workflow must not declare a nested GitHub Actions container because task runners do not provide Docker. This image provides the complete environment required by both MagiAttention and MagiAttnExtensions; a plain NGC PyTorch image is insufficient. Shared storage is available at `/home/niubility2/ci_workspace`.

PR CI uses `pull_request_target`, so the workflow definition always comes from the base repository. The GPU job references the protected `ci-internal` environment and starts only after its required reviewer approves it. After approval, the trusted base workflow explicitly checks out the PR head and merges the latest target branch before testing. This permits contributions from forks without allowing a fork to replace the workflow that grants runner access.

The approval is the current security boundary: reviewers must inspect untrusted changes before allowing arbitrary PR code to execute on an internal runner. Approved fork PRs set `CI_WORKSPACE_ROOT` to `${RUNNER_TEMP}/magi-attention-ci`; they do not publish portable markers or intentionally write normal CI artifacts to the shared cache. The H100 build job transfers its two wheels to platform test jobs through a one-day GitHub Actions artifact, preserving the single-wheel contract without using shared storage. The current Dispatcher pool still exposes the shared mount to the task, so reviewer approval must be treated as authorization for the fork code to access internal runner resources.

TODO: provision a dedicated fork runner label whose task specification does not mount shared storage, then route fork PRs to that pool. Path selection inside a shared task is not a security boundary. Same-repository PRs and `main` pushes continue to use the shared v2 cache. Every GPU job runs `.github/scripts/verify_task_runner.sh` before accessing storage.

## v2 wheel artifacts

`.github/configs/ci_dependencies.json` is the common source-dependency protocol. MagiAttention currently has no repository dependencies, so its list is empty; CI still runs the same resolver and records an empty runtime lock. Future dependency wheels use `v2/dependency-artifacts/magi-attention`, keyed by consumer namespace, dependency repository and ID, install path, exact tracked source content, base image, platform, and resolver recipe. Branches and tags are resolved once to immutable commits. Reuse fails closed on any manifest, identity, layout, or wheel digest mismatch, and publication uses private staging followed by atomic rename.

The build job is prepared for authenticated cross-repository checkout with the `DEPENDENCY_REPO_TOKEN` Actions secret and `CI_DEPENDENCY_USE_TOKEN=true`. The resolver clears the credential and Authorization header persisted by `actions/checkout` before adding this single explicit credential. The token needs read-only Contents permission for any repository later added to the dependency manifest. For fork PRs, access to this protected environment and its secrets occurs only after `ci-internal` reviewer approval.

`.github/scripts/build_v2_wheel.sh` publishes immutable standalone artifacts under:

```text
${CI_WORKSPACE_ROOT}/v2/standalone-artifacts/magi-attention/
  <cache-name>/<fingerprint>/{*.whl,manifest.json}
```

The fingerprint includes schema, base image family, tracked source digest, package version, build recipe, the fixed SM90+SM100 build target, and—for extensions—the exact main-package wheel fingerprint. Publication uses private staging followed by atomic rename. Existing directories are verified before reuse. Test jobs use install-only mode and fail rather than rebuilding a missing wheel on another GPU architecture.

Standalone wheel artifacts are local CI build outputs. Downstream consumers must independently build or verify the wheel they intend to publish.

## Portable validation

The standalone workflow tests both nodes:

- `magi_attention`
- `magi_attn_extensions`, recursively bound to the main-package validation fingerprint

Success is recorded under:

```text
${CI_WORKSPACE_ROOT}/v2/portable-validations/magi-attention/
  <platform>/v<schema>/<node>/<fingerprint>/success.json
```

The portable fingerprint includes policy-selected normalized tracked source content, exact base image tag, platform, the complete package-owned platform test profile, a stable protocol identifier, the recipe version, and the canonical portable policy projection. H100 and B300 therefore produce independent certifications. Repository-specific `.github` files may be excluded as source files without removing policy semantics from the identity.

The standalone layout uses the defaults `PORTABLE_SOURCE_ROOT=.` and `PORTABLE_BASE_TAG_FILE=.github/configs/base_image_tag.txt`. A downstream vendored layout may set those two generic variables to its own paths; the protocol contains no downstream-specific names or path detection.

`.github/tests/test_portable_validation.sh` constructs equivalent standalone and generic vendored layouts. It verifies that both layouts produce identical fingerprints, that a downstream layout accepts a marker written by the standalone layout, and that malformed markers fail closed.

Only trusted runs of `SandAI-org/MagiAttention` may publish portable markers. Failure and cancellation never produce markers. A downstream consumer must recompute the fingerprint from its vendored MagiAttention content. Different source, tests, submodule gitlinks, runtime, platform, or protocol produce a miss.

Trusted standalone runs also verify these markers after building and checking their wheels. A hit skips the corresponding test suite; a miss runs the tests and publishes the marker. Coverage is uploaded only when the MagiAttention test suite actually ran. Fork runs do not consume or publish shared markers.

A portable marker certifies tests, not wheel bytes, and must never replace wheel manifest verification or a downstream project's own publication policy.

## Required check

Branch protection should require `ci_gate`, not the conditionally skipped GPU job.
