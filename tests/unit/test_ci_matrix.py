# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exercise real platform configurations and conservative PR selection."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github/scripts/ci/generate_matrix.py"
CONFIGS = ROOT / ".github/configs"
SPEC = importlib.util.spec_from_file_location("ci_matrix", SCRIPT)
MATRIX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MATRIX)
PLATFORMS = MATRIX._load_platforms(CONFIGS)


def _run(
    event="pull_request",
    files="",
    labels="",
    dispatch="",
    pr_number="",
    run="1234",
    attempt="1",
    env=None,
):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--event-name",
            event,
            "--configs-dir",
            str(CONFIGS),
            "--changed-files",
            files,
            "--pr-labels",
            labels,
            "--dispatch-platforms",
            dispatch,
            "--pr-number",
            pr_number,
            "--workflow-run",
            run,
            "--workflow-attempt",
            attempt,
        ],
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
    )
    if result.returncode:
        return result, None
    return result, json.loads(result.stdout.removeprefix("platforms="))


@pytest.mark.parametrize("platform", PLATFORMS)
@pytest.mark.parametrize(
    "path",
    [
        "torch_fl/accelerator/{platform}/__init__.py",
        "torch_fl/configs/backends_{platform}.conf",
        "csrc/aten/backends/{platform}/op.cc",
        ".github/scripts/hooks/set_env_{platform}.sh",
        ".github/scripts/set_env_{platform}.sh",
        ".github/configs/{platform}.yml",
        ".github/workflows/integration-test-{platform}.yml",
    ],
)
def test_platform_owned_files_select_only_that_lane(platform, path):
    result, selected = _run(files=path.format(platform=platform), pr_number="42")
    assert result.returncode == 0, result.stderr
    assert selected == [platform]


@pytest.mark.parametrize(
    "files",
    [
        "",
        "README.md",
        "docs/development/testing.md",
        "docs/assets/ci.png",
        "README.md\ndocs/guide.rst",
    ],
)
def test_documentation_needs_no_hardware(files):
    assert _run(files=files, pr_number="42")[1] == []


@pytest.mark.parametrize(
    "path",
    [
        "torch_fl/__init__.py",
        "csrc/aten/generated/register.inc",
        "csrc/runtime/generator.cc",
        ".github/scripts/set_env.sh",
        ".github/scripts/lib/set_env_common.sh",
        ".github/version-pins.env",
        "cmake/flagos_platforms.json",
        "tests/integration/ops/test_add_dispatch.py",
        ".github/workflows/ci.yml",
        "setup.py",
        "pyproject.toml",
        "unknown/future_backend.cc",
        "prefix/torch_fl/accelerator/ascend/op.py",
    ],
)
def test_shared_or_unknown_code_runs_every_lane_even_with_vendor_changes(path):
    assert (
        _run(files=f"torch_fl/accelerator/ascend/op.py\n{path}", pr_number="42")[1]
        == PLATFORMS
    )


def test_multiple_vendors_select_the_union():
    assert _run(
        files="torch_fl/accelerator/ascend/op.py\ncsrc/aten/backends/musa/op.cc",
        pr_number="42",
    )[1] == ["ascend", "musa"]


def test_shared_cuda_runtime_selects_all_consumers():
    assert _run(files="csrc/runtime/accelerator/cuda/device.cc", pr_number="42")[1] == [
        "cuda",
        "dcu",
        "ppu",
    ]
    assert _run(files="csrc/runtime/allocator/backends/cuda_memory.h", pr_number="42")[
        1
    ] == [
        "cuda",
        "metax",
        "ppu",
    ]


@pytest.mark.parametrize("platform", PLATFORMS)
def test_runtime_watch_paths_cover_the_build_tables_actual_source_directory(platform):
    table = json.loads((ROOT / "cmake/flagos_platforms.json").read_text())
    runtime = table["accelerators"][platform]["runtime_dir"]
    assert (
        platform
        in _run(files=f"csrc/runtime/accelerator/{runtime}/device.cc", pr_number="42")[
            1
        ]
    )


@pytest.mark.parametrize("event", ["push", "schedule", "workflow_dispatch"])
def test_non_pr_events_default_to_every_lane(event):
    assert (
        _run(event=event, files="README.md", labels="platform:ascend")[1] == PLATFORMS
    )


@pytest.mark.parametrize("labels", ["", "platform:ascend"])
def test_shared_check_passes_publish_only_after_the_head_was_checked(
    tmp_path, monkeypatch, labels
):
    marker = tmp_path / "published"
    head = "a" * 40

    def no_changes(*args):
        return []

    def publish(*args):
        marker.write_text(repr(args))

    monkeypatch.setattr(MATRIX, "_changed_files", no_changes)
    monkeypatch.setattr(MATRIX, "_publish_common", publish)
    monkeypatch.setattr(
        "sys.argv",
        [
            str(SCRIPT),
            "--event-name",
            "workflow_call",
            "--configs-dir",
            str(CONFIGS),
            "--base-ref",
            "",
            "--head-sha",
            head,
            "--pr-labels",
            labels,
            "--pr-number",
            "42",
            "--workflow-run",
            "1234",
            "--workflow-attempt",
            "1",
        ],
    )
    MATRIX.main()
    assert marker.read_text() == repr((head, "1234-1"))


def test_pr_dispatch_uses_files_and_labels_instead_of_branch_defaults():
    assert _run(
        event="workflow_dispatch",
        files="torch_fl/accelerator/ascend/op.py",
        dispatch="",
        pr_number="42",
    )[1] == ["ascend"]
    assert _run(
        event="workflow_dispatch",
        files="README.md",
        labels="platform:musa",
        pr_number="42",
    )[1] == ["musa"]


def test_pr_attempt_defaults_to_the_event_attempt(tmp_path, monkeypatch):
    args = [
        "--event-name",
        "pull_request",
        "--configs-dir",
        str(CONFIGS),
        "--changed-files",
        "README.md",
        "--pr-labels",
        "platform:ascend",
        "--pr-number",
        "42",
        "--workflow-run",
        "1234",
    ]
    marker = tmp_path / "owner"
    monkeypatch.setattr(
        MATRIX,
        "_publish_common",
        lambda revision, owner: marker.write_text(repr((revision, owner))),
    )
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), *args])
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    MATRIX.main()
    assert marker.read_text() == repr(("HEAD", "1234-2"))


@pytest.mark.parametrize("number", ["0", "-1", "01", "＋1", "42; echo"])
def test_pr_dispatch_rejects_invalid_number(number):
    result, selected = _run(event="workflow_dispatch", pr_number=number)
    assert result.returncode == 2
    assert selected is None


@pytest.mark.parametrize(
    "run, attempt",
    [("", "1"), ("0", "1"), ("01", "1"), ("1234", "0"), ("1234", "")],
)
def test_pr_dispatch_requires_an_attempt_one_owner(run, attempt):
    result, selected = _run(run=run, attempt=attempt, pr_number="42")
    assert result.returncode == 2
    assert selected is None


@pytest.mark.parametrize(
    "path",
    [
        ".github/workflows/integration-tests.yml",
        ".github/workflows/lint.yml",
    ],
)
def test_shared_pipeline_code_runs_all_platforms(path):
    assert _run(files=path, pr_number="42")[1] == PLATFORMS


@pytest.mark.parametrize(
    "labels, expected",
    [
        ("platform:musa,platform:ascend,platform:musa", ["ascend", "musa"]),
        ("platform:ascend,platform:all", PLATFORMS),
        ("platform:unknown", PLATFORMS),
        ("platform:ascend,platform:unknown", PLATFORMS),
    ],
)
def test_routing_labels_override_file_filters_without_duplicates(labels, expected):
    assert _run(files="README.md", labels=labels, pr_number="42")[1] == expected


@pytest.mark.parametrize(
    "requested, expected",
    [
        ("", PLATFORMS),
        ("all", PLATFORMS),
        ("musa,ascend,musa", ["ascend", "musa"]),
    ],
)
def test_manual_selection(requested, expected):
    assert _run(event="workflow_dispatch", dispatch=requested)[1] == expected


@pytest.mark.parametrize("requested", ["unknown", "all,ascend", "ascend,unknown"])
def test_invalid_manual_selection_fails(requested):
    result, selected = _run(event="workflow_dispatch", dispatch=requested)
    assert result.returncode == 2
    assert selected is None


def test_missing_watch_metadata_falls_back_to_all(tmp_path):
    (tmp_path / "cuda.yml").write_text("display_name: CUDA\n")
    assert MATRIX._platforms_from_files(["op.cc"], ["cuda"], tmp_path) == ["cuda"]
    assert MATRIX._platforms_from_files(None, PLATFORMS, CONFIGS) == PLATFORMS


def test_git_diff_preserves_deleted_and_renamed_platform_paths(tmp_path, monkeypatch):
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(tmp_path), *args], text=True
        ).strip()

    git("init", "-q")
    git("config", "user.name", "CI test")
    git("config", "user.email", "ci@example.invalid")
    git("config", "commit.gpgsign", "false")
    old = tmp_path / "ascend.cc"
    old.write_text("original")
    git("add", ".")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    old.rename(tmp_path / "musa.cc")
    git("add", "-A")
    git("commit", "-qm", "rename")
    monkeypatch.chdir(tmp_path)
    assert MATRIX._changed_files("unused", base) == ["ascend.cc", "musa.cc"]
    assert MATRIX._changed_files("missing") is None
