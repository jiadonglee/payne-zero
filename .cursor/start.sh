#!/bin/bash
# Per-boot reconciliation for Payne Zero.
#
# When a Cloud Agent boots from a prebuilt environment build, `install` is NOT
# rerun, and the fresh git checkout leaves the ~7 GB of Git LFS runtime data as
# pointer stubs. The atmosphere package loads several of these tables at import
# time, so they must be materialized on every boot. `git lfs pull` is
# idempotent: when the LFS objects are already in the local cache (baked into
# the build snapshot) it is a fast local checkout with no network download.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo "[payne-zero start] materializing Git LFS runtime data"
for attempt in 1 2 3 4; do
    if git lfs pull; then
        echo "[payne-zero start] Git LFS runtime data ready"
        exit 0
    fi
    echo "[payne-zero start] git lfs pull failed (attempt ${attempt}); retrying" >&2
    sleep $((4 * attempt))
done

echo "[payne-zero start] ERROR: could not materialize Git LFS runtime data" >&2
exit 1
