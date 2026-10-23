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

GPU CI uses task runners labeled `ci-spot-job` plus the `runner_label` that `.github/configs/ci_platforms.json` declares for each platform profile. A platform profile is a test configuration, and several profiles may run on one runner label: `h100` runs on `h100`, and both `b300` and `b300_fa4` run on `b300`. H100 builds one wheel with `MAGI_ATTENTION_BUILD_COMPUTE_CAPABILITY=90,100,103`, so both `magi_attn_ext` and `magi_attn_comm` contain SM90, SM100 and SM103 code. Every enabled platform installs those exact wheel bytes. Set the Actions variable `B300_CI_ENABLED=true` only after the B300 runner is available; otherwise the planner omits every profile whose runner label is `b300`.

The package-owned `h100` profile tests the canonical backend names `ffa` and `sdpa`, preserving the existing Hopper coverage. The `b300` profile tests `cutedsl` and `sdpa`. The `b300_fa4` profile tests `fa4` and `cutedsl`; `fa4` is the test enum value for the FFA_FA4 implementation. Before a `b300_fa4` portable test actually runs, the package-owned test recipe installs `flash_attn_cute` and its native utilities from the exact checked-out MagiAttention source for the architectures in `MAGI_ATTENTION_TEST_FLASH_ATTN_CUTE_ARCHS` (`sm100`). A runner-local content-addressed stamp shares that installation between the main and extensions test invocations in the same job. On each platform that installs `flash_attn_cute`, the main-package test recipe then runs `tools/precompile_ffa_fa4.py` with the installed package. The script writes the FFA_FA4 kernels to `MAGI_ATTENTION_FFA_FA4_CACHE_DIR`, which is a new cache dir in the temporary test directory of each run. Pytest loads the FFA_FA4 kernels from this cache dir. If the precompile fails, the job fails. `MAGI_ATTENTION_JIT_COMPILE_DISABLED` applies only to native FFA kernels. Thus the tests JIT-compile each FFA_FA4 kernel that is not in the precompiled grid. Both B300 profiles set `MAGI_ATTENTION_FFA_CUTEDSL_CACHE_ENABLED=1`; each test run then sets `MAGI_ATTENTION_FFA_CUTEDSL_CACHE_DIR` to a new cache dir in its temporary test directory, which its test processes share. Portable hits skip both the tests and this setup. The H100 build performs the normal FFA AOT prebuild once. Unsupported parameter combinations are filtered by the test suite. The H100 profile alone retains `NCCL_NVLS_ENABLE=0`; B300 does not inherit that Hopper-specific workaround. The Dispatcher creates each task directly from `registry.cn-sh-01.sensecore.cn/sandai-ccr/magi-base:26.05.2`, matching `.github/configs/base_image_tag.txt`; the workflow must not declare a nested GitHub Actions container because task runners do not provide Docker. This image provides the remaining environment required by both MagiAttention and MagiAttnExtensions; a plain NGC PyTorch image is insufficient. Shared storage is available at `/home/niubility2/ci_workspace`.

Every job installs one `nvidia-cutlass-dsl` and `quack-kernels` pair from exactly one of two sources, chosen by whether the `PIP_CONSTRAINT` files pin the pair with `==`. After installation, `.github/scripts/install_requirements.sh` prints the installed pair and fails the job if it differs from that source:

- No pair pin in `PIP_CONSTRAINT` (the `h100` and `b300` test jobs; the base image's own constraint file does not pin the pair): after `requirements.txt`, the script upgrades only these two packages to the newest release that their `requirements.txt` lines allow, and requires the inclusive upper bound of each line (4.8.0 and 0.6.5). Upgrading the whole requirements file would also upgrade unrelated packages such as `setuptools` and `packaging`.
- `PIP_CONSTRAINT=.github/configs/ci_constraints.txt` (the `b300_fa4` test job and the `build_wheels` job): the profile and the build step append this file to the `PIP_CONSTRAINT` of the base image, so both files apply. The constraints file pins `nvidia-cutlass-dsl==4.4.2` and `quack-kernels==0.4.1`. `flash_attn_cute`, which is built from the FlashAttention submodule, fails to import with `nvidia-cutlass-dsl` 4.6.0 or newer, so the `fa4` tests need this pair. The build job runs with the `h100` profile but replaces its DSL source with this constraint, because the standalone wheel fingerprint does not cover Python dependency versions. Constraints cannot carry extras, so `requirements.txt` keeps the `[cu13]` extra.

The script has no mode that keeps an installed pair, such as the release in the base image. CI validates the two ends of the supported range, 4.4.2 with quack 0.4.1 and 4.8.0 with quack 0.6.5; see `docs/source/user_guide/install.md` for the supported pairs.

`magi_attn_extensions` still imports when `flash_attn_cute` does not, but its FA4 tests require `flash_attn_cute` on SM100. The `b300` profile therefore sets `CI_TEST_EXTENSIONS=false`, which skips the MagiAttnExtensions test suite and its portable marker on that platform; the wheel import checks still run. The other profiles set `CI_TEST_EXTENSIONS=true`. `CI_TEST_EXTENSIONS` is a CI script variable, not a runtime variable of `magi_attention`.

The platform test profile is part of the portable fingerprint. The profile names the constraints file by path, and `.github` files are not source input, so the recipe digest also includes the SHA-256 of the content of the profile's `PIP_CONSTRAINT` file. A missing file fails the fingerprint. A change to `ci_constraints.txt` therefore changes the `b300_fa4` fingerprint and leaves the other profiles unchanged.

PR CI uses `pull_request_target`, so the workflow definition always comes from the base repository. Change detection and platform planning run from base-repository code before any approval. The GPU jobs reference the protected `ci-internal` environment and cannot start until one of its required reviewers approves the workflow run. After approval, `actions/checkout` opts into fork checkout and checks out GitHub's PR merge ref. Before any repository script runs, the workflow verifies that the merge commit parents exactly match the immutable base and head SHAs from the event. A later PR update therefore causes verification to fail rather than running a different revision under an older approval.

### Fork PR approval and trust

`ci-internal` approval is the security boundary for fork PRs. It is a code-trust decision, not merely an authorization to consume GPU capacity. An approved fork runs arbitrary repository code on an internal task runner with the job's secrets, network access, and the shared `/home/niubility2/ci_workspace` mount. It may consume and publish standalone wheels, compiled artifacts, dependency artifacts, and portable validation results in the shared v2 cache. `allow-unsafe-pr-checkout: true` is therefore intentional and is safe only under this review policy.

Before approving a fork workflow run, the reviewer must:

- verify that the workflow run's base and head SHAs correspond to the exact PR revision being reviewed;
- review the complete diff against the target branch, including generated or unusually large files;
- inspect all code that can execute during checkout, installation, build, import, test collection, and tests, especially `setup.py`, `pyproject.toml`, `.github/scripts/`, `.github/tests/`, and shell scripts;
- inspect every submodule URL and gitlink update, because recursive checkout fetches and later executes submodule content;
- look for credential access, unexpected network requests, writes outside the workspace, direct access to `/home/niubility2`, background processes, and changes intended to influence future cache hits;
- reject the approval if the code cannot be confidently reviewed, and require the contributor to simplify or explain the change instead of using CI as a sandbox.

Approval treats the reviewed fork revision as fully trusted. Restricting normal scripts to a runner-local cache would not isolate malicious code while the shared mount remains accessible, so approved forks use the same content-addressed shared cache as same-repository PRs and `main` pushes. Manifest and digest verification detects corruption but does not establish artifact provenance; the human review is what authorizes fork code to write reusable cache entries. Every GPU job runs `.github/scripts/verify_task_runner.sh` before accessing storage.

For every event, the H100 build job also transfers its two wheels and manifests to platform test jobs through a run-scoped GitHub Actions artifact. This preserves the single-wheel contract and guarantees that all platform jobs test the exact wheel produced by that run, including when the shared cache misses.

## v2 wheel artifacts

`.github/configs/ci_dependencies.json` is the common source-dependency protocol. MagiAttention currently has no repository dependencies, so its list is empty; CI still runs the same resolver and records an empty runtime lock. Future dependency wheels use `v2/dependency-artifacts/magi-attention`, keyed by consumer namespace, dependency repository and ID, install path, exact tracked source content, base image, platform, and resolver recipe. Branches and tags are resolved once to immutable commits. Reuse fails closed on any manifest, identity, layout, or wheel digest mismatch, and publication uses private staging followed by atomic rename.

The build job is prepared for authenticated cross-repository checkout with the `DEPENDENCY_REPO_TOKEN` Actions secret and `CI_DEPENDENCY_USE_TOKEN=true`. The resolver clears the credential and Authorization header persisted by `actions/checkout` before adding this single explicit credential. The token needs read-only Contents permission for any repository later added to the dependency manifest. For fork PRs, access to this protected environment and its secrets occurs only after `ci-internal` reviewer approval.

`.github/scripts/build_v2_wheel.sh` publishes immutable standalone artifacts under:

```text
${CI_WORKSPACE_ROOT}/v2/standalone-artifacts/magi-attention/
  <cache-name>/<fingerprint>/{*.whl,manifest.json}
```

The fingerprint includes schema, base image family, tracked source digest, package version, build recipe, the fixed SM90+SM100+SM103 build target, and—for extensions—the exact main-package wheel fingerprint. Publication uses private staging followed by atomic rename. Existing directories are verified before reuse. Test jobs use install-only mode and fail rather than rebuilding a missing wheel on another GPU architecture.

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

The portable fingerprint includes policy-selected normalized tracked source content, exact base image tag, platform, the complete package-owned platform test profile, a stable protocol identifier, the recipe version, and the canonical portable policy projection. Each platform profile therefore produces independent certifications, including the two profiles on the B300 runner. Repository-specific `.github` files may be excluded as source files without removing policy semantics from the identity.

The standalone layout uses the defaults `PORTABLE_SOURCE_ROOT=.` and `PORTABLE_BASE_TAG_FILE=.github/configs/base_image_tag.txt`. A downstream vendored layout may set those two generic variables to its own paths; the protocol contains no downstream-specific names or path detection.

`.github/tests/test_portable_validation.sh` constructs equivalent standalone and generic vendored layouts. It verifies that both layouts produce identical fingerprints, that a downstream layout accepts a marker written by the standalone layout, and that malformed markers fail closed.

Only trusted runs of `SandAI-org/MagiAttention` may publish portable markers. GPU jobs from a pull request become trusted only after `ci-internal` approval; `main` pushes are trusted directly. Failure and cancellation never produce markers. A downstream consumer must recompute the fingerprint from its vendored MagiAttention content. Different source, tests, submodule gitlinks, runtime, platform, or protocol produce a miss.

Trusted standalone runs also verify these markers after building and checking their wheels. A hit skips the corresponding test suite; a miss runs the tests and publishes the marker. Coverage is uploaded only when the MagiAttention test suite actually ran. Approved fork runs consume and publish markers under the same rules as other trusted runs.

A portable marker certifies tests, not wheel bytes, and must never replace wheel manifest verification or a downstream project's own publication policy.

## Required check

Branch protection should require `ci_gate`, not the conditionally skipped GPU job.
