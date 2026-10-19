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

"""Execute trusted checkout resolution with a fake, local-only GitHub CLI."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
RUN = WORKFLOW["jobs"]["resolve-checkout"]["steps"][0]["run"]
SHA = "a" * 40


def _resolve(
    tmp_path,
    number="",
    pr=None,
    candidates=None,
    event="push",
    revision="",
    runs=None,
):
    binary = tmp_path / "bin"
    binary.mkdir()
    gh = binary / "gh"
    gh.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "import sys\n"
        "if os.environ.get('FAKE_CLI_LOG'):\n"
        "    with open(os.environ['FAKE_CLI_LOG'], 'a') as log:\n"
        "        log.write(' '.join(sys.argv[1:]) + '\\n')\n"
        "endpoint = sys.argv[2]\n"
        "if '/actions/workflows/ci.yml/runs' in endpoint:\n"
        "    print(os.environ['FAKE_RUNS'])\n"
        "elif '/pulls/' in endpoint:\n"
        "    print(os.environ['FAKE_PR'])\n"
        "elif 'cancel' in endpoint:\n"
        "    print('{}')\n"
        "elif '/pulls?' in endpoint:\n"
        "    print(os.environ['FAKE_CANDIDATES'])\n"
        "else:\n"
        "    print(os.environ['FAKE_PR'])\n"
    )
    gh.chmod(0o755)
    output = tmp_path / "output"
    output.touch()
    env = {
        **os.environ,
        "PATH": f"{binary}:{os.environ['PATH']}",
        "EVENT_REPOSITORY": "flagos-ai/Torch-FL",
        "EVENT_REF": SHA,
        "EVENT_NAME": event,
        "EVENT_PULL_REQUEST": "",
        "PR_NUMBER": number,
        "EXPECTED_REVISION": revision,
        "GITHUB_OUTPUT": str(output),
        "FAKE_PR": json.dumps(pr or {}),
        "FAKE_CANDIDATES": json.dumps(candidates or []),
        "FAKE_RUNS": json.dumps({"workflow_runs": runs or []}),
        "FAKE_CLI_LOG": str(tmp_path / "cli.log"),
    }
    process = subprocess.run(
        ["bash", "-e", "-c", RUN], env=env, capture_output=True, text=True
    )
    return process, output.read_text()


def test_normal_run_pins_the_event_commit(tmp_path):
    process, output = _resolve(tmp_path)
    assert process.returncode == 0, process.stderr
    assert output == f"repository=flagos-ai/Torch-FL\nref={SHA}\npull_request=\n"


def test_fork_run_resolves_both_repository_and_immutable_head(tmp_path):
    pr = {
        "state": "open",
        "head": {"repo": {"full_name": "contributor/fork"}, "sha": "b" * 40},
    }
    process, output = _resolve(tmp_path, "42", pr)
    assert process.returncode == 0, process.stderr
    assert output == (f"repository=contributor/fork\nref={'b' * 40}\npull_request=42\n")


def test_trusted_dispatch_rejects_a_head_that_advanced_after_dispatch(tmp_path):
    pr = {
        "state": "open",
        "head": {"repo": {"full_name": "contributor/fork"}, "sha": "b" * 40},
    }
    process, output = _resolve(tmp_path, "42", pr, revision=SHA)
    assert process.returncode == 1
    assert "no longer points at" in process.stdout
    assert output == ""


def test_pull_request_event_resolves_its_number_without_event_payload(tmp_path):
    pr = {
        "number": 588,
        "state": "open",
        "head": {"repo": {"full_name": "flagos-ai/Torch-FL"}, "sha": SHA},
    }
    process, output = _resolve(
        tmp_path,
        pr=pr,
        candidates=[
            {"number": 588, "state": "open"},
            {"number": 589, "state": "closed"},
        ],
        event="pull_request",
    )
    assert process.returncode == 0, process.stderr
    assert output == f"repository=flagos-ai/Torch-FL\nref={SHA}\npull_request=588\n"


def test_pull_request_event_without_an_identifiable_pr_fails_behind_a_guard(
    tmp_path,
):
    process, output = _resolve(
        tmp_path,
        candidates=[
            {"number": 587, "state": "open"},
            {"number": 588, "state": "open"},
        ],
        event="pull_request",
    )
    assert process.returncode == 1
    assert "Could not identify this commit's pull request" in process.stdout
    assert output == ""


@pytest.mark.parametrize("number", ["0", "-1", "01", "42; echo unsafe", "$(whoami)"])
def test_invalid_pr_numbers_fail_before_any_checkout(tmp_path, number):
    process, output = _resolve(tmp_path, number)
    assert process.returncode == 1
    assert output == ""


@pytest.mark.parametrize(
    "pr",
    [
        {
            "state": "closed",
            "head": {"repo": {"full_name": "contributor/fork"}, "sha": SHA},
        },
        {"state": "open", "head": {"repo": None, "sha": SHA}},
        {
            "state": "open",
            "head": {"repo": {"full_name": "contributor/fork"}, "sha": "main"},
        },
        {
            "state": "open",
            "head": {"repo": {"full_name": "bad/repo/extra"}, "sha": SHA},
        },
    ],
)
def test_unavailable_or_invalid_heads_fail_before_checkout(tmp_path, pr):
    process, output = _resolve(tmp_path, "42", pr)
    assert process.returncode == 1
    assert output == ""


def test_documented_events_cancel_older_work_within_one_branch_or_pr():
    events = WORKFLOW[True]
    assert {"labeled", "unlabeled"} <= set(events["pull_request"]["types"])
    assert "paths-ignore" in events["push"]
    assert "paths-ignore" not in events["pull_request"]
    group = WORKFLOW["jobs"]["ci-complete"]["concurrency"]["group"]
    assert "github.event_name" in group
    assert "github.head_ref" in group
    assert "github.ref_name" in group
    assert WORKFLOW["jobs"]["ci-complete"]["concurrency"]["cancel-in-progress"]


def test_privileged_control_workflows_only_checkout_the_trusted_event_commit():
    for filename in ("slash-command.yml", "flaky-retry.yml"):
        workflow = yaml.safe_load((ROOT / ".github/workflows" / filename).read_text())
        for job in workflow["jobs"].values():
            if "uses" in job:
                assert "steps" not in job
                continue
            checkouts = [
                step
                for step in job["steps"]
                if str(step.get("uses", "")).startswith("actions/checkout@")
            ]
            if any(isinstance(step, dict) and "uses" in step for step in job["steps"]):
                assert len(checkouts) == 1
                assert checkouts[0]["with"]["ref"] == "${{ github.sha }}"
                assert checkouts[0]["with"]["persist-credentials"] is False
        assert "contents" in workflow["permissions"]


def test_pull_request_cancels_only_active_runs_for_older_heads(tmp_path):
    pr = {
        "state": "open",
        "head": {"repo": {"full_name": "flagos-ai/Torch-FL"}, "sha": SHA},
    }
    candidates = [{"number": 588, "state": "open"}]
    runs = [
        {
            "id": 111,
            "head_sha": "b" * 40,
            "status": "queued",
            "event": "pull_request",
            "pull_requests": [{"number": 588}],
        },
        {
            "id": 222,
            "head_sha": "b" * 40,
            "status": "completed",
            "event": "pull_request",
            "pull_requests": [{"number": 588}],
        },
        {
            "id": 333,
            "head_sha": SHA,
            "status": "in_progress",
            "event": "pull_request",
            "pull_requests": [{"number": 588}],
        },
        {"id": 444, "head_sha": "c" * 40, "status": "in_progress", "event": "push"},
        {
            "id": 555,
            "head_sha": "d" * 40,
            "status": "waiting",
            "event": "workflow_dispatch",
            "pull_requests": [{"number": 588}],
        },
        {
            "id": 666,
            "head_sha": "e" * 40,
            "status": "queued",
            "event": "pull_request",
            "pull_requests": [{"number": 589}],
        },
        {"id": 777, "head_sha": "f" * 40, "status": "queued", "event": "push"},
    ]
    process, output = _resolve(
        tmp_path, pr=pr, candidates=candidates, event="pull_request", runs=runs
    )
    assert process.returncode == 0, process.stderr
    assert output == f"repository=flagos-ai/Torch-FL\nref={SHA}\npull_request=588\n"
    log = (tmp_path / "cli.log").read_text()
    assert "actions/runs/111/cancel" in log
    assert "actions/runs/555/cancel" in log
    assert "actions/runs/222/cancel" not in log
    assert "actions/runs/333/cancel" not in log
    assert "actions/runs/444/cancel" not in log
    assert "actions/runs/666/cancel" not in log
    assert "actions/runs/777/cancel" not in log


def test_push_does_not_cancel_ci_runs(tmp_path):
    process, output = _resolve(tmp_path)
    assert process.returncode == 0, process.stderr
    assert output == f"repository=flagos-ai/Torch-FL\nref={SHA}\npull_request=\n"
    log = tmp_path / "cli.log"
    assert not log.exists() or "cancel" not in log.read_text()
