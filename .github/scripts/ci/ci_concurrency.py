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

"""Record shared checks and gate hardware delivery for PR workflows."""

import argparse
import os
import re
import sys

from github_api import GitHub, GitHubError

SHA = re.compile(r"[0-9a-f]{40}")
OWNER = re.compile(r"[1-9][0-9]*-[1-9][0-9]*")
COMMON = "ci-common"
DELIVERY = "ci-delivery"
COMMON_STATES = {"pending", "success", "failure", "error"}
DELIVERY_STATES = {"pending", "success", "failure", "error", "cancelled"}
LANE_STATES = DELIVERY_STATES - {"pending"}


def _revision(value):
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise ValueError("revision must be a full lowercase commit SHA")
    return value


def _lane_context(platform):
    if not isinstance(platform, str) or not re.fullmatch(
        r"[a-z][a-z0-9_-]{0,48}", platform
    ):
        raise ValueError("platform must be a short lowercase identifier")
    return f"ci-platform:{platform}"


def publish_status(github, revision, state, context, description, target_url=None):
    _revision(revision)
    if state not in {"pending", "success", "failure", "error"}:
        raise ValueError("invalid commit status state")
    if not context or not description or any(c in description for c in "\r\n"):
        raise ValueError("commit status context and description must be single lines")
    if len(context) > 255 or len(description) > 140:
        raise ValueError("commit status context or description is too long")
    payload = {"state": state, "context": context, "description": description}
    if target_url is not None:
        payload["target_url"] = target_url
    github.repo_api(f"statuses/{revision}", method="POST", payload=payload)


def latest_statuses(github, revision):
    """Return the newest status record for each relevant context."""
    revision = _revision(revision)
    pages = github.repo_api(f"commits/{revision}/statuses?per_page=100", paginate=True)
    if not isinstance(pages, list) or not all(isinstance(page, list) for page in pages):
        raise ValueError("GitHub returned an invalid commit-status response")
    records = [
        item
        for page in pages
        for item in page
        if isinstance(item, dict)
        and (
            item.get("context") in {COMMON, DELIVERY}
            or str(item.get("context", "")).startswith("ci-platform:")
        )
    ]
    if not all(
        isinstance(item.get("id"), int) and isinstance(item.get("context"), str)
        for item in records
    ):
        raise ValueError("GitHub returned an invalid commit-status record")
    if len({item["id"] for item in records}) != len(records):
        raise ValueError("GitHub returned ambiguous CI status identifiers")
    latest = {}
    for item in records:
        context = item["context"]
        if context not in latest or latest[context]["id"] < item["id"]:
            latest[context] = item
    return list(latest.values())


def _owner_url(github, owner):
    run, attempt = owner.split("-", 1)
    return (
        f"https://github.com/{github.repository}/actions/runs/{run}/attempts/{attempt}"
    )


def _owner_from_url(github, url):
    if not isinstance(url, str):
        return None
    repository = re.escape(github.repository)
    match = re.fullmatch(
        rf"https://github\.com/{repository}/actions/runs/"
        rf"([1-9][0-9]*)(?:/attempts/([1-9][0-9]*))?/?",
        url,
    )
    return f"{match.group(1)}-{match.group(2) or '1'}" if match else None


def _latest_status(statuses, context):
    records = [item for item in statuses if item.get("context") == context]
    return max(records, key=lambda item: item["id"]) if records else None


def _latest_common(github, statuses):
    common = _latest_status(statuses, COMMON)
    if common is None or _owner_from_url(github, common.get("target_url")) is None:
        return None
    return common


def active_runs(github, revision):
    """Return active CI workflows for this revision, keyed by id."""
    revision = _revision(revision)
    pages = github.repo_api(
        "actions/workflows/ci.yml/runs?per_page=100&exclude_pull_requests=true",
        paginate=True,
        jq=(
            '[.workflow_runs[] | select(.status == "in_progress" or '
            '.status == "waiting" or .status == "queued") | '
            "select(.head_sha == $revision) | "
            'select(.event == "pull_request" or (.event == "workflow_dispatch" and '
            '.path == ".github/workflows/ci.yml")) | '
            "{id, conclusion, head_sha, run_attempt}]"
        ),
        jq_args={"revision": revision},
    )
    if not isinstance(pages, list) or not all(isinstance(page, list) for page in pages):
        raise ValueError("GitHub returned an invalid active-run response")
    runs = [item for page in pages for item in page]
    if not all(
        isinstance(item.get("id"), int)
        and item["id"] > 0
        and item.get("head_sha") == revision
        and item.get("conclusion") is None
        for item in runs
    ):
        raise ValueError("GitHub returned an invalid active CI run")
    if len({item["id"] for item in runs}) != len(runs):
        raise ValueError("GitHub returned ambiguous CI run identifiers")
    return {item["id"]: item["run_attempt"] for item in runs}


def publish_common(github, revision, owner):
    """Record one shared-check pass for a revision."""
    revision = _revision(revision)
    if not isinstance(owner, str) or not OWNER.fullmatch(owner):
        raise ValueError("status owner must be a workflow run and attempt")
    publish_status(
        github,
        revision,
        "success",
        COMMON,
        "Shared checks passed for this exact revision.",
        target_url=_owner_url(github, owner),
    )
    return owner


def _check_admission(github, revision, pull_request, owner):
    """Return the delivery decision for a run that still owns its revision."""
    if not isinstance(pull_request, str) or not re.fullmatch(
        r"[1-9][0-9]*", pull_request
    ):
        raise ValueError("pull request must be a positive integer")
    if not isinstance(owner, str) or not OWNER.fullmatch(owner):
        raise ValueError("status owner must be a workflow run and attempt")
    if revision != _revision(os.environ.get("GITHUB_SHA") or revision):
        raise ValueError("hardware admission must run from its event commit")
    requesting_run, attempt = (int(value) for value in owner.split("-", 1))
    if active_runs(github, revision).get(requesting_run) != attempt:
        return False, None, "The requesting CI run has already completed."
    statuses = latest_statuses(github, revision)
    common = [item for item in statuses if item.get("context") == COMMON]
    delivery = [item for item in statuses if item.get("context") == DELIVERY]
    lanes = [
        item
        for item in statuses
        if str(item.get("context", "")).startswith("ci-platform:")
    ]
    if any(item.get("state") not in COMMON_STATES for item in common):
        raise ValueError("GitHub returned an invalid shared-check status")
    if any(item.get("state") not in DELIVERY_STATES for item in delivery + lanes):
        raise ValueError("GitHub returned an invalid hardware-delivery status")
    latest_common = _latest_common(github, statuses)
    if latest_common is None or latest_common.get("state") != "success":
        return False, None, "Shared checks have not passed for this revision."
    if _owner_from_url(github, latest_common.get("target_url")) != owner:
        return False, None, "Another workflow run owns this shared-check result."
    return True, _latest_status(statuses, DELIVERY), "This run owns the revision."


def admit(github, revision, pull_request, owner):
    """Reserve one PR revision's hardware delivery for this run."""
    revision = _revision(revision)
    allowed, delivery, message = _check_admission(github, revision, pull_request, owner)
    if not allowed:
        return False, message
    if (
        delivery is not None
        and _owner_from_url(github, delivery.get("target_url")) != owner
    ):
        return False, "A newer workflow run owns hardware delivery."
    if delivery is not None and delivery.get("state") != "pending":
        return False, "Hardware delivery already finished for this run."
    if delivery is None:
        publish_status(
            github,
            revision,
            "pending",
            DELIVERY,
            f"Hardware tests admitted for PR #{pull_request}, run {owner}.",
            target_url=_owner_url(github, owner),
        )
        return True, "Hardware delivery admitted."
    return True, "Hardware delivery already admitted for this run."


def should_run(github, revision, pull_request, owner):
    """Return whether a hardware job still belongs to the current review."""
    revision = _revision(revision)
    allowed, delivery, message = _check_admission(github, revision, pull_request, owner)
    if not allowed:
        return False, message
    if (
        delivery is None
        or delivery.get("state") != "pending"
        or _owner_from_url(github, delivery.get("target_url")) != owner
    ):
        return False, "This run no longer owns the hardware reservation."
    return True, message


def publish_platform(github, revision, owner, platform, state):
    """Record one platform lane's result."""
    if state not in LANE_STATES:
        raise ValueError("invalid platform-delivery status state")
    description = {
        "success": "Hardware lane passed.",
        "failure": "Hardware lane failed.",
        "error": "Hardware lane did not complete.",
        "cancelled": "Hardware lane was cancelled.",
    }[state]
    publish_status(
        github,
        revision,
        state,
        _lane_context(platform),
        description,
        target_url=_owner_url(github, owner),
    )


def publish_platform_outcome(github, revision, result, owner, platform):
    """Publish a lane outcome while its run still owns the pending delivery."""
    revision = _revision(revision)
    if not isinstance(owner, str) or not OWNER.fullmatch(owner):
        raise ValueError("status owner must be a workflow run and attempt")
    _lane_context(platform)
    state = (
        "success"
        if result == "success"
        else "failure"
        if result == "failure"
        else "error"
    )
    delivery = _latest_status(latest_statuses(github, revision), DELIVERY)
    if (
        delivery is None
        or delivery.get("state") != "pending"
        or _owner_from_url(github, delivery.get("target_url")) != owner
    ):
        return False
    publish_platform(github, revision, owner, platform, state)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY", ""))
    sub = parser.add_subparsers(dest="command", required=True)
    common = sub.add_parser("publish-common")
    common.add_argument("--revision", required=True)
    common.add_argument("--owner", required=True)
    admission = sub.add_parser("admit")
    admission.add_argument("--revision", required=True)
    admission.add_argument("--pull-request", required=True)
    admission.add_argument("--owner", required=True)
    active = sub.add_parser("should-run")
    active.add_argument("--revision", required=True)
    active.add_argument("--pull-request", required=True)
    active.add_argument("--owner", required=True)
    hardware = sub.add_parser("publish-hardware")
    hardware.add_argument("--revision", required=True)
    hardware.add_argument("--result", required=True)
    hardware.add_argument("--owner", required=True)
    hardware.add_argument("--platform", required=True)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repository):
        print("::error::repository must be owner/name", file=sys.stderr)
        return 2
    try:
        github = GitHub(args.repository)
        if args.command == "publish-common":
            publish_common(github, args.revision, args.owner)
            print("Published shared-check success.")
        elif args.command in {"admit", "should-run"}:
            operation = admit if args.command == "admit" else should_run
            allowed, message = operation(
                github, args.revision, args.pull_request, args.owner
            )
            if not allowed:
                raise ValueError(message)
            print(message)
        else:
            published = publish_platform_outcome(
                github, args.revision, args.result, args.owner, args.platform
            )
            print(
                "Published hardware result."
                if published
                else "This review no longer owns hardware delivery."
            )
    except (GitHubError, ValueError, OSError) as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
