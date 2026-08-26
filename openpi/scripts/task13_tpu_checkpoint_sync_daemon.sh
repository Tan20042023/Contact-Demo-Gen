#!/usr/bin/env bash
# Copy each completed local Orbax checkpoint step to GCS without touching the
# TPU training process. Local POSIX Orbax commits each numbered step by atomic
# directory rename, so a visible numeric child directory is complete.
set -euo pipefail

: "${TASK13_TPU_LOCAL_CHECKPOINT_DIR:?Set the local checkpoint directory for one run}"
: "${TASK13_TPU_GCS_CHECKPOINT_DIR:?Set the corresponding GCS checkpoint directory}"
: "${TASK13_TPU_SYNC_POLL_SECONDS:=60}"
: "${TASK13_TPU_SYNC_ONCE:=0}"

local_root="${TASK13_TPU_LOCAL_CHECKPOINT_DIR%/}"
gcs_root="${TASK13_TPU_GCS_CHECKPOINT_DIR%/}"

sync_step() {
  local step="$1"
  local local_step="${local_root}/${step}"
  local gcs_step="${gcs_root}/${step}"
  local local_bytes remote_bytes

  [[ -d "${local_step}" ]] || return 0
  if gcloud storage ls "${gcs_step}/UPLOAD_COMPLETE" >/dev/null 2>&1; then
    return 0
  fi

  echo "Syncing completed local checkpoint step ${step} to ${gcs_step}"
  gcloud storage rsync --recursive "${local_step}" "${gcs_step}"
  local_bytes="$(find "${local_step}" -type f -printf '%s\n' | awk '{total += $1} END {print total + 0}')"
  remote_bytes="$(gcloud storage du -s "${gcs_step}/**" | awk '{print $1}')"
  if [[ "${local_bytes}" != "${remote_bytes}" ]]; then
    echo "Refusing completion marker for step ${step}: local=${local_bytes}, gcs=${remote_bytes}" >&2
    return 1
  fi
  printf 'completed_utc=%s\nlocal_bytes=%s\nsource_step=%s\n' "$(date -u +%FT%TZ)" "${local_bytes}" "${step}" \
    | gcloud storage cp - "${gcs_step}/UPLOAD_COMPLETE"
  echo "Checkpoint step ${step} safely uploaded (${local_bytes} bytes)."
}

sync_visible_steps() {
  [[ -d "${local_root}" ]] || return 0
  while IFS= read -r step; do
    sync_step "${step}"
  done < <(find "${local_root}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
    | awk '/^[0-9]+$/' | sort -n)
}

while true; do
  sync_visible_steps
  [[ "${TASK13_TPU_SYNC_ONCE}" == "1" ]] && break
  sleep "${TASK13_TPU_SYNC_POLL_SECONDS}"
done
