# MagiAttention CI

## Runtime and runner

GPU CI uses the task-based runner labels `[ci-spot-job, h100]` and runs the job in `registry.cn-sh-01.sensecore.cn/sandai-ccr/magi-base:26.05.2`, matching `.github/workflows/base_image_tag.txt`. This image provides the complete environment required by both MagiAttention and MagiAttnExtensions; a plain NGC PyTorch image is insufficient. The host shared storage `/mnt/afs/ci_workspace` is mounted in the job container at `/home/niubility2/ci_workspace`.

PR CI uses `pull_request_target`, so the workflow definition always comes from the base repository. The GPU job references the protected `ci-internal` environment and starts only after its required reviewer approves it. After approval, the trusted base workflow explicitly checks out the PR head and merges the latest target branch before testing. This permits contributions from forks without allowing a fork to replace the workflow that grants runner access.

The approval is a security boundary: reviewers must inspect untrusted changes before allowing arbitrary PR code to execute on an internal runner. Fork runs use a separate job without the shared-storage volume, never publish validation markers or shared wheel artifacts, and keep temporary artifacts under `${RUNNER_TEMP}/magi-attention-ci`. Same-repository PRs and `main` pushes use the shared v2 cache. Every GPU job runs `.github/scripts/verify_task_runner.sh` before accessing its selected storage.

## v2 wheel artifacts

`.github/scripts/build_v2_wheel.sh` publishes immutable standalone artifacts under:

```text
${CI_WORKSPACE_ROOT}/v2/standalone-artifacts/magi-attention/
  <cache-name>/<fingerprint>/{*.whl,manifest.json}
```

The fingerprint includes schema, base image family, tracked source digest, package version, build recipe, and—for extensions—the validated main-package dependency. Publication uses private staging followed by atomic rename. Existing directories are verified before reuse.

Standalone wheel artifacts are local CI build outputs. Downstream consumers must independently build or verify the wheel they intend to publish.

## Portable validation

The standalone workflow tests both nodes:

- `magi_attention`
- `magi_attn_extensions`, recursively bound to the main-package validation fingerprint

Success is recorded under:

```text
${CI_WORKSPACE_ROOT}/v2/portable-validations/magi-attention/
  v<schema>/<node>/<fingerprint>/success.json
```

The portable fingerprint includes normalized tracked source content, exact base image tag, platform, and the byte identity of `.github/scripts/portable_validation.sh`, which is also the canonical test entrypoint. Repository-specific `.github` files are excluded from source identity; the protocol script is hashed separately.

The standalone layout uses the defaults `PORTABLE_SOURCE_ROOT=.` and `PORTABLE_BASE_TAG_FILE=.github/workflows/base_image_tag.txt`. A downstream vendored layout may set those two generic variables to its own paths; the protocol contains no downstream-specific names or path detection.

`.github/scripts/test_portable_validation.sh` constructs equivalent standalone and generic vendored layouts. It verifies that both layouts produce identical fingerprints, that a downstream layout accepts a marker written by the standalone layout, and that malformed markers fail closed.

Only trusted runs of `SandAI-org/MagiAttention` may publish portable markers. Failure and cancellation never produce markers. A downstream consumer must recompute the fingerprint from its vendored MagiAttention content. Different source, tests, submodule gitlinks, runtime, platform, or protocol produce a miss.

A portable marker certifies tests, not wheel bytes, and must never replace wheel manifest verification or a downstream project's own publication policy.

## Required check

Branch protection should require `ci_gate`, not the conditionally skipped GPU job.
