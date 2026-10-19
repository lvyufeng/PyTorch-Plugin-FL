# CI selection and maintainer controls

`ci.yml` resolves one immutable source commit, runs the shared checks, routes
that review to the affected platforms, and invokes the reusable
`integration-tests.yml` pipeline once per selected platform. It must land on
the default branch before either event listener can load this revised
workflow definition.

## Automatic platform selection

| Event | Hardware selection |
| --- | --- |
| PR with `platform:all` | Every configured platform |
| PR with one or more `platform:<name>` labels | The union of those platforms |
| PR with an unknown routing label | Every platform, conservatively |
| PR without routing labels | Platforms affected by the changed files |
| Documentation-only PR without routing labels | No hardware jobs; shared checks still run |
| Push or daily schedule | Every platform |
| Manual dispatch | `platforms=all` or a comma-separated platform list |
| `/ci` command | The platform names after `/ci run`; `all` runs every platform |

Routing labels take precedence over file filtering. Adding or removing labels
reruns PR CI. Supported platform names are `ascend`, `cuda`, `dcu`, `gcu`,
`kunlunxin`, `metax`, `musa` and `ppu`.

Each `.github/configs/<platform>.yml` declares `paths_watch`: exact file names,
directory prefixes ending in `/`, or shell-style globs. Its own configuration
and dedicated workflow automatically belong to that lane. Shared code or an
unrecognized code path selects all lanes; a recognized vendor change must not
hide an unrelated shared change in the same PR. Missing or invalid watch
metadata also selects all lanes. A failed Git comparison selects all lanes.

Selection compares the immutable PR base and head commits and includes both
paths when files move between platforms. Watch lists include shared consumers:
CUDA runtime sources affect CUDA, DCU and PPU, while the CUDA allocator header
also affects MetaX. Keep these relationships aligned with
`cmake/flagos_platforms.json` and the runtime build configuration when adding or
changing a platform. Changes to the shared CI control scripts and workflows
select every platform.

## Required check and hardware results

The top-level job named **CI complete** runs after the shared checks and
platform routing. It passes only when every shared job and routing step
succeeds, as well as validating the selected platform list. A selected platform
that unexpectedly skips, fails or is cancelled does not hold up this check;
its result is recorded separately. A documentation-only PR can pass after its
shared checks and routing succeed.

A repository administrator can make **CI complete** the required status check
in the branch protection rule or ruleset. Adding the job does not change that
repository setting automatically. Do not substitute individual platform job
names for this aggregate check.

## `/ci` comments

Repository writers, maintainers and administrators can post a command on the
first line of a PR comment:

```text
/ci run
/ci run all
/ci run ascend
/ci run ascend cuda
/ci run ascend,cuda
```

The command validates platform names, keeps unrelated labels and replaces
routing labels. It resolves the open PR's current head, cancels unfinished CI
runs for an older head of that PR, and dispatches `ci.yml` on the default branch
with that head fixed as `revision`. A maintainer can start the same trusted path with the
**Slash command (/ci)** workflow's `pr_number` and `platforms` inputs; the job
resolves the PR head itself. A `/ci` comment and a manual dispatch for the same
PR share a concurrency group, so a newer request replaces the older control
job. The trusted workflow comes from the base repository; the source commit
comes from the PR. Every shared check, hardware checkout and manifest loader
receives that same source. This supports fork PRs without loading workflow
definitions from the fork. Authoring a PR alone does not authorize this
privileged hardware dispatch route.

A valid command is acknowledged only after dispatch succeeds. Invalid commands
from authorized users receive a usage response. Closed PRs and deleted fork
heads cannot dispatch. An explicit `/ci run` platform selection overrides the
PR's automatic file and label routing for that run. `/ci` reruns the shared
checks before requesting hardware.

Requested runs provide diagnostics for the PR head. Their workflow run belongs
to the default branch; they do not replace the ordinary PR's required **CI
complete** check. They publish shared-check and hardware-delivery statuses
bound to the PR commit, so hardware stays tied to the revision that passed the
shared checks.

## Cancellation and queues

The **CI complete** gate uses one native concurrency group for each branch and
one for each pull request, with `cancel-in-progress: true`. It is a job-level
group because an event-dependent workflow-level expression made fork pull
requests fail before any job started. A new commit or label change supersedes
older work at that gate. A newer push to `main` similarly supersedes an older
unfinished `main` push, while other CI branches use their own groups. Already
completed runs remain completed, and different PRs and branches do not cancel
each other.

Each PR or `/ci` CI run resolves its head in the first job and cancels
queued, waiting and in-progress `ci.yml` runs for older PR heads before its own
shared checks request a runner. Runs for repository pushes and the nightly
schedule use separate concurrency groups and are not cancelled. A manual
trusted dispatch repeats the same cancellation for an older head, then starts
the current revision. The trusted dispatch compares the expected head with
the PR's head again inside `ci.yml`, which refuses to start if the PR has moved.

Native cancellation stops the shared checks at the concurrency gate. GitHub may
leave a separate reusable hardware job running after the shared-check run
completes or `/ci` starts another dispatch. The hardware job checks delivery
ownership when it obtains a runner and before starting the build. A superseded
job therefore cannot build or publish a new result, but the hardware runner may
take time to be reclaimed. Observe cancellation behavior on the real runner
fleet before treating instant runner termination as guaranteed.

Documentation-only pushes to a CI branch do not start the workflow. Markdown
and reStructuredText are excluded case-insensitively from file-based hardware
routing, but a documentation-only PR still runs shared checks.

A PR run publishes a `ci-common` success status only after lint, codegen checks
and platform-agnostic checks pass, platform routing succeeds, and the PR still
points at the tested commit. Each publication links to the workflow run and
attempt that produced it. A new workflow attempt creates its own `ci-common`
record, so an older rerun cannot use it to restart hardware.

The `admission` job accepts hardware only from the run that owns the newest
`ci-common` success for that commit. It publishes a revision-bound
`ci-delivery` status before configuration or hardware starts. This status is
the revision-level admission record; it does not replace the per-platform
results. A newer `ci-common` result supersedes an older hardware reservation;
the older run cannot publish or reclaim admission. Admission is also rejected
when the same run already has a pending or finished delivery, so a repeated
`/ci` does not start another matrix for that commit.

A push with a new commit receives its own `ci-common` and `ci-delivery`
history. The first CI job for a newer PR head cancels unfinished work for the
older head. A stale hardware job that is still winding down checks the commit
status before building, so the statuses keep the two results separate until
its runner is reclaimed.

The shared-check and delivery statuses are attached to the source commit.
GitHub may not display statuses from a fork head in the pull request's Checks
tab; they are still available on the commit and are used by trusted
admission. An ordinary `pull_request` check run belongs to the synthetic merge
commit, while `/ci` belongs to the default branch, so neither substitutes for
the `ci-complete` result.

Each platform has its own concurrency group, so the configured runner fleet
still determines how many platform builds execute in parallel. Admission
groups only clean up repeated attempts inside one reusable workflow; the
cross-run owner is the commit status. Platform lanes from one CI run are not
serialized with `max-parallel: 1`: every selected lane can queue and start as
soon as its own runner is available. The schedule continues to run the full
matrix once a day without PR admission.

When a platform completes, its reusable workflow records a
`ci-platform:<name>` status. The result is published only while that run still
owns the pending `ci-delivery` record. A successful revision-level admission
therefore starts different platforms independently, and stale hardware
results remain isolated from later commits.

The workflow-dispatch form of `integration-tests.yml` runs one selected
platform for diagnostics and bypasses PR admission. It does not publish a
PR-head result. PR hardware starts through the reusable workflow from
`ci.yml`.

Intentional cancellation does not notify the Feishu CI channel. A cancelled
run has no new CI result to report; the replacement run sends its own result.

## Bounded network retry

`flaky-retry.yml` listens for completed, failed runs of `ci.yml`. It checks the
actual run and all failed root jobs using the GitHub API. It excludes the
dependent **CI complete** summary from root-cause classification.

Only recognized network failures are retried: for example pip read timeouts,
DNS failures, connection resets, selected curl transport failures and HTTP
502/503/504 responses. Every failed root job must qualify. Assertion, import,
compiler, missing-package and OOM failures take precedence over network noise.
Unknown failures, driver faults, collective hangs, plain timeouts, unavailable
logs and cancellations remain failed for investigation.

The policy requests `gh run rerun --failed` at most once, including dependent
jobs so the aggregate gate is recalculated. It checks the current attempt again
immediately before requesting a rerun; a duplicate event or manual rerun must
not consume another retry. GitHub CLI calls have a 120-second limit and the
supplemental workflow has a five-minute limit. Retry does not turn a failed CI
result into a pass: the rerun must succeed. No AI service credentials are needed.

Privileged control workflows checkout scripts from their own trusted event
commit and never execute a failed PR's scripts or downloaded artifacts. Both
[`issue_comment`](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#issue_comment)
and
[`workflow_run`](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#workflow_run)
require their workflow definitions to exist on the default branch.

## Validation and rollout

Platform-agnostic CI runs regression tests for selection, routing, required gate
results, checkout resolution, comment dispatch, network classification and retry
budgets. GitHub operations in these tests are simulated; tests do not post real
comments or start hardware runs.

Before considering rollout verified, run hardware CI, configure the required
check, then exercise comment dispatch for an actual fork PR and observe a real
network-failure retry. The local tests and Actionlint checks cannot verify runner
availability, repository write permissions or GitHub's event delivery.

The first rollout should confirm three things on the runners: a new PR commit
stops superseded shared checks, superseded queued hardware stops before
acquiring a runner, and an already running reusable job stops before the build
and cannot publish after a newer review takes admission. This change cannot
prove how quickly GitHub terminates a reusable job that has already acquired a
hardware runner. Unexpected non-CI cancellations should still be investigated
in GitHub Actions.
