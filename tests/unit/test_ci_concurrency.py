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

"""Verify cancellation groups, hardware admission, and workflow wiring."""

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github/scripts/ci/ci_concurrency.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("ci_concurrency", SCRIPT)
CONCURRENCY = importlib.util.module_from_spec(SPEC)
sys.modules["ci_concurrency"] = CONCURRENCY
SPEC.loader.exec_module(CONCURRENCY)
CI = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
INTEGRATION = yaml.safe_load(
    (ROOT / ".github/workflows/integration-tests.yml").read_text()
)
SLASH = yaml.safe_load((ROOT / ".github/workflows/slash-command.yml").read_text())
SHA = "a" * 40
OWNER = "1234-1"
BASE_URL = "https://github.com/flagos-ai/Torch-FL/actions/runs"


def _status(record_id, state, owner=OWNER, context="ci-common"):
    return {
        "id": record_id,
        "state": state,
        "context": context,
        "target_url": f"{BASE_URL}/{owner.replace('-', '/attempts/')}",
    }


class FakeGitHub:
    repository = "flagos-ai/Torch-FL"

    def __init__(self, statuses=None, head=SHA, active=None):
        self.statuses = [] if statuses is None else statuses
        self.head = head
        self.active = (
            {
                "id": 1234,
                "head_sha": SHA,
                "conclusion": None,
                "run_attempt": 1,
            }
            if active is None
            else active
        )

    def repo_api(self, path, **kwargs):
        if path.startswith("commits/") and path.endswith("/statuses?per_page=100"):
            assert kwargs.get("paginate") is True
            return [self.statuses]
        if path.startswith("actions/workflows/ci.yml/runs?"):
            assert kwargs["jq_args"] == {"revision": self.head}
            return [[self.active]] if isinstance(self.active, dict) else [self.active]
        if path.startswith("statuses/"):
            payload = kwargs["payload"]
            self.statuses.append(
                {
                    "id": max((item["id"] for item in self.statuses), default=0) + 1,
                    **payload,
                }
            )
            return None
        raise AssertionError(path)


def test_main_and_pr_groups_are_separate_and_cancel_older_runs():
    assert "concurrency" not in CI
    concurrency = CI["jobs"]["ci-complete"]["concurrency"]
    assert concurrency["cancel-in-progress"] is True
    assert "github.head_ref" in concurrency["group"]
    assert "github.ref_name" in concurrency["group"]
    assert "paths-ignore" in CI[True]["push"]
    assert "paths-ignore" not in CI[True]["pull_request"]
    assert "dev-musa" in CI[True]["pull_request"]["branches"]
    assert "cancel-superseded" not in CI["jobs"]


def test_manual_dispatch_keeps_the_full_hardware_matrix():
    inputs = CI[True]["workflow_dispatch"]["inputs"]
    assert inputs["platforms"]["default"] == "all"
    assert inputs["pr_number"]["default"] == ""
    assert inputs["revision"]["default"] == ""


def test_hardware_owner_includes_trusted_workflow_dispatch_runs():
    requests = []

    class ActiveRunGitHub(FakeGitHub):
        def repo_api(self, path, **kwargs):
            if path.startswith("actions/workflows/ci.yml/runs?"):
                requests.append(kwargs)
                return [[self.active]]
            return super().repo_api(path, **kwargs)

    assert CONCURRENCY.active_runs(ActiveRunGitHub(), SHA) == {1234: 1}
    expression = requests[0]["jq"]
    assert '"workflow_dispatch"' in expression
    assert '".github/workflows/ci.yml"' in expression
    assert '"workflow_call"' not in expression


def test_hardware_waits_for_shared_checks(monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", SHA)
    assert CONCURRENCY.admit(FakeGitHub(), SHA, "42", OWNER) == (
        False,
        "Shared checks have not passed for this revision.",
    )


def test_publish_common_records_the_latest_run():
    github = FakeGitHub()
    CONCURRENCY.publish_common(github, SHA, OWNER)
    assert [item["context"] for item in github.statuses] == ["ci-common"]
    assert github.statuses[0]["target_url"] == f"{BASE_URL}/1234/attempts/1"


@pytest.mark.parametrize("state", ["failure", "error", "pending"])
def test_cancelled_or_failed_shared_checks_block_hardware(monkeypatch, state):
    monkeypatch.setenv("GITHUB_SHA", SHA)
    github = FakeGitHub(statuses=[_status(1, state)])
    assert not CONCURRENCY.admit(github, SHA, "42", OWNER)[0]
    assert len(github.statuses) == 1


def test_shared_result_needs_its_own_active_run(monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", SHA)
    github = FakeGitHub(statuses=[_status(1, "success", "5678-1")])
    assert not CONCURRENCY.admit(github, SHA, "42", OWNER)[0]
    assert len(github.statuses) == 1


def test_shared_check_pass_creates_one_shared_delivery(monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", SHA)
    github = FakeGitHub()
    CONCURRENCY.publish_common(github, SHA, OWNER)
    assert CONCURRENCY.admit(github, SHA, "42", OWNER) == (
        True,
        "Hardware delivery admitted.",
    )
    assert CONCURRENCY.admit(github, SHA, "42", OWNER) == (
        True,
        "Hardware delivery already admitted for this run.",
    )
    assert [item["context"] for item in github.statuses] == [
        "ci-common",
        "ci-delivery",
    ]


def test_rerun_attempt_is_bound_to_the_rerun(monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", SHA)
    github = FakeGitHub(
        active=[
            {
                "id": 5678,
                "head_sha": SHA,
                "conclusion": None,
                "run_attempt": 2,
            }
        ]
    )
    CONCURRENCY.publish_common(github, SHA, "5678-2")
    assert CONCURRENCY.admit(github, SHA, "42", "5678-2")[0]
    assert not CONCURRENCY.admit(github, SHA, "42", OWNER)[0]


def test_new_commit_with_old_pending_delivery_does_not_reserve(monkeypatch):
    new_sha = "b" * 40
    monkeypatch.setenv("GITHUB_SHA", new_sha)
    github = FakeGitHub(
        head=new_sha,
        active=[
            {"id": 5678, "head_sha": new_sha, "conclusion": None, "run_attempt": 1}
        ],
        statuses=[
            _status(1, "success", "1234-1"),
            _status(2, "pending", "1234-1", context="ci-delivery"),
            _status(3, "success", "5678-1"),
        ],
    )
    assert not CONCURRENCY.admit(github, new_sha, "42", "5678-1")[0]
    assert len(github.statuses) == 3


def test_platform_lanes_publish_without_claiming_another_delivery(monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", SHA)
    github = FakeGitHub()
    CONCURRENCY.publish_common(github, SHA, OWNER)
    assert CONCURRENCY.admit(github, SHA, "42", OWNER)[0]
    assert CONCURRENCY.publish_platform_outcome(github, SHA, "success", OWNER, "cuda")
    assert CONCURRENCY.publish_platform_outcome(github, SHA, "success", OWNER, "musa")
    assert [item["context"] for item in github.statuses[2:]] == [
        "ci-platform:cuda",
        "ci-platform:musa",
    ]


def test_cancelled_run_and_finished_delivery_cannot_publish():
    github = FakeGitHub(
        statuses=[_status(1, "pending", "5678-1", context="ci-delivery")]
    )
    assert not CONCURRENCY.publish_platform_outcome(
        github, SHA, "failure", OWNER, "cuda"
    )
    github.statuses[0] = _status(1, "success", context="ci-delivery")
    assert not CONCURRENCY.publish_platform_outcome(
        github, SHA, "success", OWNER, "cuda"
    )
    assert len(github.statuses) == 1


def test_hardware_runner_rechecks_the_newest_shared_pass(monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", SHA)
    github = FakeGitHub()
    CONCURRENCY.publish_common(github, SHA, OWNER)
    assert CONCURRENCY.admit(github, SHA, "42", OWNER)[0]
    assert CONCURRENCY.should_run(github, SHA, "42", OWNER)[0]
    CONCURRENCY.publish_common(github, SHA, "5678-1")
    assert not CONCURRENCY.should_run(github, SHA, "42", OWNER)[0]
    assert CONCURRENCY.publish_platform_outcome(github, SHA, "success", OWNER, "cuda")


def test_completed_run_cannot_claim_hardware(monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", SHA)
    github = FakeGitHub()
    CONCURRENCY.publish_common(github, SHA, OWNER)
    github.active = []
    assert not CONCURRENCY.admit(github, SHA, "42", OWNER)[0]
    assert not CONCURRENCY.should_run(github, SHA, "42", OWNER)[0]
    assert [item["context"] for item in github.statuses] == ["ci-common"]


@pytest.mark.parametrize("owner", ["0-1", "1234-0", "1234", "1234-1-2"])
def test_owner_attempt_has_to_be_well_formed(owner):
    with pytest.raises(ValueError):
        CONCURRENCY.publish_common(FakeGitHub(), SHA, owner)


@pytest.mark.parametrize("value", ["main; echo unsafe", "../../other", "A" * 40])
def test_invalid_revision_is_rejected(value):
    with pytest.raises(ValueError):
        CONCURRENCY.publish_common(FakeGitHub(), value, OWNER)


def test_integration_workflow_gates_hardware_on_admission():
    jobs = INTEGRATION["jobs"]
    assert ["admission", "configuration"] == jobs["integration-tests"]["needs"]
    assert ["admission", "integration-tests"] == jobs["publish-hardware"]["needs"]
    assert "inputs.platform" in jobs["admission"]["concurrency"]["group"]
    assert "should-run" in jobs["integration-tests"]["steps"][1]["run"]
    publish = jobs["publish-hardware"]["steps"]
    assert publish[0]["uses"] == "actions/checkout@v4"
    assert '--platform "$PLATFORM"' in publish[-1]["run"]


def test_ci_owns_pr_routing():
    detect = CI["jobs"]["discover-platforms"]["steps"][-1]
    assert {"GH_TOKEN", "WORKFLOW_RUN", "WORKFLOW_ATTEMPT"} <= set(detect["env"])
    assert "--workflow-run" in detect["run"]
    assert "--workflow-attempt" in detect["run"]
    assert "platform-pipeline" not in CI["jobs"]["ci-complete"]["needs"]
    assert "lock_group" not in SLASH["jobs"]["dispatch"]["outputs"]
