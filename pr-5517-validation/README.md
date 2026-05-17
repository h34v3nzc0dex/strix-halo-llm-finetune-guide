# PR #5517 validation on gfx1151

Local pytest run for our own [PR #5517](https://github.com/unslothai/unsloth/pull/5517)
(*"fix(studio/worker): inject --gcc-install-dir for HIP source builds on
Ubuntu 24.04"*) — mirrors the dep set Unsloth's Backend CI runs against, so
the result here is what CI will show too.

## Hardware / venv

- AMD Ryzen AI MAX+ 395, Radeon 8060S (`gfx1151`), 128 GB unified
- Ubuntu 24.04 LTS, kernel 6.19.14 mainline
- Sacrificial venv at `/tmp/unsloth-5517-test` (Python 3.12)
- Deps installed: `studio/backend/requirements/studio.txt` + the same extras
  the Backend CI workflow installs, including `torch>=2.4,<2.11` CPU build
  and `transformers>=4.51,<5.5`
- Production venv at `/srv/aurora-ai/venv` was NOT touched

## Test file results — all 17/17 passed

The file under change, run alone:

```
tests/test_training_worker_flash_attn.py::test_should_try_runtime_flash_attn_install_threshold_and_skip PASSED
tests/test_training_worker_flash_attn.py::test_runtime_flash_attn_prefers_prebuilt_wheel              PASSED
tests/test_training_worker_flash_attn.py::test_runtime_flash_attn_falls_back_to_pypi                  PASSED
tests/test_training_worker_flash_attn.py::test_runtime_flash_attn_skip_env_avoids_all_install_work    PASSED
tests/test_training_worker_flash_attn.py::test_runtime_flash_attn_skips_on_blackwell                  PASSED
tests/test_training_worker_flash_attn.py::test_causal_conv1d_fast_path_preserves_wheel_first_install_args  PASSED
tests/test_training_worker_flash_attn.py::test_causal_conv1d_fast_path_includes_qwen3_6_variants      PASSED
tests/test_training_worker_flash_attn.py::test_mamba_ssm_path_preserves_wheel_first_install_args      PASSED
# ↑ 8 pre-existing tests — no regressions from this PR's changes

tests/test_training_worker_flash_attn.py::test_hipcc_gcc_install_dir_picks_highest_with_headers       PASSED
tests/test_training_worker_flash_attn.py::test_hipcc_gcc_install_dir_picks_14_when_headers_exist      PASSED
tests/test_training_worker_flash_attn.py::test_hipcc_gcc_install_dir_returns_none_when_no_match       PASSED
tests/test_training_worker_flash_attn.py::test_hipcc_gcc_install_dir_returns_none_on_non_linux        PASSED
tests/test_training_worker_flash_attn.py::test_hipcc_gcc_install_dir_returns_none_on_non_x86_64       PASSED
tests/test_training_worker_flash_attn.py::test_install_injects_gcc_install_dir_on_hip_source_build    PASSED
tests/test_training_worker_flash_attn.py::test_install_appends_to_existing_hipcc_compile_flags        PASSED
tests/test_training_worker_flash_attn.py::test_install_respects_user_gcc_install_dir                  PASSED
tests/test_training_worker_flash_attn.py::test_install_does_not_inject_env_on_cuda                    PASSED
# ↑ 9 tests added by this PR — all passing

============================== 17 passed in 1.16s ==============================
```

## Full backend suite — zero regressions

Same dep set + same `-k` deselect filter the Backend CI workflow uses:

```
1073 passed, 47 skipped, 35 deselected, 7 subtests passed in 12.51s
```

CI's documented baseline for this suite is *"831 passed, 5 skipped, 35 deselected"*
(see `.github/workflows/studio-backend-ci.yml`). The pass count grew because
the repo has had additions since that comment was written; the
35-deselected matches exactly (same GPU-specific tests skipped by the same
filter), and no failures.

## Reproducing

```bash
python3.12 -m venv /tmp/unsloth-5517-test
cd /path/to/your/unsloth-checkout  # branch: fix/hipcc-gcc-install-dir-ubuntu-24-04

/tmp/unsloth-5517-test/bin/pip install -r studio/backend/requirements/studio.txt
/tmp/unsloth-5517-test/bin/pip install \
  python-multipart aiofiles sqlalchemy cryptography \
  pyyaml jinja2 mammoth unpdf requests \
  'numpy<3' pytest pytest-asyncio httpx
/tmp/unsloth-5517-test/bin/pip install --index-url https://download.pytorch.org/whl/cpu 'torch>=2.4,<2.11'
/tmp/unsloth-5517-test/bin/pip install 'transformers>=4.51,<5.5'

cd studio/backend
/tmp/unsloth-5517-test/bin/python -m pytest tests/test_training_worker_flash_attn.py -v
```

Full log: `test-suite.log` next to this README.
