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

"""Validate /ci, replace unfinished PR runs, and dispatch the current head."""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import quote

from github_api import GitHub, GitHubError
from generate_matrix import _load_platforms, _platforms_from_dispatch


def parse_command(body, platforms):
    first_line = body.splitlines()[0].strip() if body.splitlines() else ""
    words = first_line.split()
    if not words or words[0] != "/ci":
        return None
    if len(words) < 2 or words[1] != "run":
        raise ValueError("Use /ci run [all | platform ...].")
    requested = ",".join(words[2:])
    try:
        selected = _platforms_from_dispatch(requested, platforms)
    except ValueError as exc:
        raise ValueError(
            "Invalid platform selection. Use /ci run all or /ci run <platform ...>. "
            f"Supported platforms: {', '.join(platforms)}."
        ) from exc
    if not selected:
        raise ValueError("No CI platforms are configured.")
    return selected


def _cancel_older_runs(github, revision, number):
    """Stop unfinished CI runs that describe an older version of the PR."""
    pages = github.repo_api(
        "actions/workflows/ci.yml/runs?per_page=100",
        paginate=True,
    )
    runs = {
        run.get("id")
        for page in pages
        for run in page
        if isinstance(run, dict)
        and run.get("head_sha") != revision
        and run.get("status") in {"queued", "in_progress", "waiting"}
        and run.get("event") in {"pull_request", "workflow_dispatch"}
        and any(
            str(pull_request.get("number")) == str(number)
            for pull_request in run.get("pull_requests") or []
            if isinstance(pull_request, dict)
        )
    }
    for run in runs:
        github.repo_api(
            f"actions/runs/{run}/cancel",
            method="POST",
        )


def handle(event, platforms, github, result=None):
    issue = event.get("issue", {})
    comment = event.get("comment", {})
    if not issue.get("pull_request"):
        return "ignored: comment is not on a pull request"
    body = comment.get("body", "")
    if not body.startswith("/ci"):
        return "ignored: not a CI command"
    login = comment.get("user", {}).get("login", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]+(?:\[bot\])?", login):
        return "ignored: commenter login is unavailable"
    permission = github.repo_api(f"collaborators/{quote(login, safe='')}/permission")
    if permission.get("permission") not in {"write", "maintain", "admin"}:
        return "ignored: /ci requires repository write access"
    number = issue.get("number")
    if not isinstance(number, int) or number <= 0:
        raise ValueError("PR number must be a positive integer.")
    try:
        selected = parse_command(body, platforms)
    except ValueError as exc:
        github.repo_api(
            f"issues/{number}/comments", method="POST", payload={"body": str(exc)}
        )
        return "rejected: invalid command"
    if selected is None:
        return "ignored: not a CI command"
    pr = github.repo_api(f"pulls/{number}")
    if pr.get("state") != "open":
        return "ignored: PR is closed"
    head = pr.get("head", {})
    repository = (head.get("repo") or {}).get("full_name", "")
    sha = head.get("sha", "")
    if not re.fullmatch(
        r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository
    ) or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("The PR head repository or commit is unavailable.")
    labels = (
        ["platform:all"]
        if selected == platforms
        else [f"platform:{p}" for p in selected]
    )
    requested = "all" if selected == platforms else ",".join(selected)
    _cancel_older_runs(github, sha, number)
    default_branch = github.api(f"repos/{github.repository}")["default_branch"]
    github.command(
        "workflow",
        "run",
        "ci.yml",
        "--repo",
        github.repository,
        "--ref",
        default_branch,
        "--field",
        f"platforms={requested}",
        "--field",
        f"pr_number={number}",
        "--field",
        f"revision={sha}",
    )
    # Create routing labels before changing the PR's existing label set.
    existing = github.repo_api("labels?per_page=100", paginate=True)
    known = {label["name"] for page in existing for label in page}
    for label in labels:
        if label not in known:
            github.repo_api(
                "labels",
                method="POST",
                payload={
                    "name": label,
                    "color": "0075ca",
                    "description": "CI platform routing",
                },
            )
    current = github.repo_api(f"issues/{number}/labels?per_page=100", paginate=True)
    old = {label["name"] for page in current for label in page}
    github.repo_api(
        f"issues/{number}/labels", method="POST", payload={"labels": labels}
    )
    for label in sorted(old - set(labels)):
        if label.startswith("platform:"):
            github.repo_api(
                f"issues/{number}/labels/{quote(label, safe='')}", method="DELETE"
            )
    # The caller starts ci.yml on the trusted default branch and binds that
    # run to this PR's immutable head. Admission checks the same head.
    github.repo_api(
        f"issues/{number}/comments",
        method="POST",
        payload={
            "body": f"CI dispatched for `{requested}` on PR #{number}. "
            f"[View workflow runs](https://github.com/{github.repository}/actions/workflows/ci.yml)",
        },
    )
    github.repo_api(
        f"issues/comments/{comment['id']}/reactions",
        method="POST",
        payload={"content": "+1"},
    )
    if result is not None:
        result.update(pr_number=str(number), revision=sha, requested=requested)
    return f"dispatched: {requested} for PR #{number}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-file", default=os.environ.get("GITHUB_EVENT_PATH"))
    parser.add_argument("--configs-dir", default=".github/configs")
    args = parser.parse_args()
    event = json.loads(Path(args.event_file).read_text())
    github = GitHub(event["repository"]["full_name"])
    try:
        if event.get("event_name") == "workflow_dispatch":
            inputs = event.get("inputs", {})
            number = inputs.get("pr_number", "")
            pr = github.repo_api(f"pulls/{number}")
            sha = pr.get("head", {}).get("sha", "")
            if pr.get("state") != "open" or not re.fullmatch(r"[0-9a-f]{40}", sha):
                raise ValueError("The dispatched PR has no open head commit.")
            platforms = _load_platforms(Path(args.configs_dir))
            selected = _platforms_from_dispatch(inputs.get("platforms", ""), platforms)
            requested = "all" if selected == platforms else ",".join(selected)
            _cancel_older_runs(github, sha, number)
            with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
                stream.write(
                    f"pr_number={number}\nrevision={sha}\n"
                    f"requested_platforms={requested}\n"
                )
            print(f"dispatched: {requested} for PR #{number}")
            return 0
        outputs = {}
        result = handle(event, _load_platforms(Path(args.configs_dir)), github, outputs)
        if outputs:
            with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
                output.write(f"requested_platforms={outputs.get('requested', '')}\n")
                output.write(f"pr_number={outputs.get('pr_number', '')}\n")
                output.write(f"revision={outputs.get('revision', '')}\n")
        print(result)
    except (GitHubError, ValueError) as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
