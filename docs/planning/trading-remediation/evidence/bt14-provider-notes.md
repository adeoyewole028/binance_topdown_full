# BT14 provider and runtime constraints (2026-10-03)

Current primary sources checked during BT14 preparation:

- [Render persistent disks](https://render.com/docs/disks): paid service, one attached service instance, runtime-only access; separate web/worker services cannot share this disk; deploys have downtime. The existing free+disk blueprint is unsupported.
- [Render blueprint reference](https://render.com/docs/blueprint-spec): `autoDeployTrigger: off` disables automatic deployment; `healthCheckPath` is supported. Proposed same-host supervisor owns HTTP and execution/scanning children. Actual plan/budget remains unapproved; do not apply a paid blueprint.
- [Render deployments](https://render.com/docs/deploys) and [trigger-deploy API](https://api-docs.render.com/reference/create-deploy): a release must specify the verified commitId; API deployment does not disable auto-deploy automatically. Existing push hooks plus platform auto-deploy are competing release owners.
- [Python 3.11.17](https://www.python.org/downloads/release/python-31117/): security release October 1, 2026, source-only; local Windows 3.11.3 is historical test evidence, not a current production security claim. A current pinned runtime requires clean validation before acceptance.

No Render service, provider account, external destination, credentials, notification, hosting purchase or deployment was accessed or changed. These are documentation checks. Hosting budget/access, actual alert destination and rollout authorization remain external gates. Local backup copies on the same disk do not establish host-loss RPO: an authorized off-host copy and tested retrieval are required before activation.

2026-10-04: isolated Astral-managed CPython3.11.17 clean installation succeeded without system PATH/registry changes; Windows evidence remains separate from production Linux. The [Linux fsync manual](https://man7.org/linux/man-pages/man2/fsync.2.html) confirms syncing file contents alone does not guarantee directory entries reach disk. The BT14 manager now syncs staging/parent/published backup directories and incident publication/archive directories on Linux; actual Linux persistence/fault proof remains required.
