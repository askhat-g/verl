#!/usr/bin/env bash
# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
# ===================================================================================
# End-to-End GRPO Runner on GKE Node Auto-Provisioning (NAP) TPU v5p (`bodaborg-v5p-nap`)
#
# Modeled after `tests/special_tpu/run_tpu_e2e_ci.sh`, this script automates the full
# lifecycle of running Qwen3-0.6B GRPO on the shared GKE NAP TPU v5p cluster:
#   1. Connects to `bodaborg-v5p-nap` using an isolated KUBECONFIG file.
#   2. Renders `examples/tpu/gke/ray-jobset-v5p-nap.yaml` with a unique cluster name
#      (`verl-nap-<user>-<run_id>`) and applies the Ray Head Service + Kueue JobSet.
#   3. Waits for Kueue admission, GKE NAP TPU v5p node provisioning, GCSFuse sidecar
#      readiness, and Ray cluster formation (`8.0 TPU` across `tpu-group-0` & `tpu-group-1`).
#   4. Syncs the local `verl` repository to the Ray head pod, submits the GRPO job
#      (`examples/tpu/grpo/run_qwen3_0_6b_torchtitan.sh`), and streams/polls status.
#   5. Pulls the complete Ray job logs to local disk (`LOG_FILE`) and runs
#      `tests/special_tpu/verify_tpu_e2e_log.py` to validate training rewards & gradients.
#   6. Strictly tears down ONLY the JobSet and Service created by this invocation via
#      an `EXIT` trap (`kubectl delete jobset/${RAY_CLUSTER_NAME} svc/${RAY_CLUSTER_NAME}-head-svc`),
#      releasing Kueue quota and NAP TPU nodes without touching any other workloads.
#
# Usage:
#   # Fast 5-step smoke test (validates full trainer + vLLM rollout + Raiden weight sync):
#   SMOKE_TEST=1 bash examples/tpu/grpo/run_grpo_v5p_nap.sh
#
#   # Full 100-step Qwen3-0.6B GSM8K GRPO training run:
#   bash examples/tpu/grpo/run_grpo_v5p_nap.sh
#
#   # Heterogeneous slices: Qwen3-4B with 1 x 8-chip Trainer slice (2x2x2) + 2 x 4-chip Sampler slices (2x2x1):
#   SMOKE_TEST=1 \
#   JOB_SCRIPT=examples/tpu/grpo/run_qwen3_4b_torchtitan.sh \
#   TRAINER_HOSTS_PER_SLICE=2 TRAINER_NODE_TOPOLOGY=2x2x2 \
#   ROLLOUT_SLICE_REPLICAS=2 ROLLOUT_HOSTS_PER_SLICE=1 ROLLOUT_NODE_TOPOLOGY=2x2x1 \
#   ROLLOUT_TP=4 \
#   bash examples/tpu/grpo/run_grpo_v5p_nap.sh
# ===================================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
IMAGE_ENV_FILE="${REPO_ROOT}/tests/special_tpu/gke/tpu-ci-image.env"
JOBSET_TEMPLATE="${REPO_ROOT}/examples/tpu/gke/ray-jobset-v5p-nap.yaml"

if [[ -f "${IMAGE_ENV_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${IMAGE_ENV_FILE}"
fi

# --- Cluster & Kueue configuration ---
GCP_PROJECT="${GCP_PROJECT:-cloud-tpu-shared-capacity}"
GKE_CLUSTER="${GKE_CLUSTER:-bodaborg-v5p-nap}"
GKE_REGION="${GKE_REGION:-europe-west4}"
RAY_NAMESPACE="${RAY_NAMESPACE:-default}"
KUEUE_QUEUE="${KUEUE_QUEUE:-multislice-queue}"
PRIORITY_CLASS="${PRIORITY_CLASS:-medium}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-default}"
HEAD_NODEPOOL="${HEAD_NODEPOOL:-cpu-np}"
TPU_NODE_SELECTOR_ACCEL="${TPU_NODE_SELECTOR_ACCEL:-tpu-v5p-slice}"
WORKER_NODE_TOPOLOGY="${WORKER_NODE_TOPOLOGY:-2x2x1}"
RESERVATION_NAME="${RESERVATION_NAME:-cloudtpu-20260902214500-1810493672}"
GCS_BUCKET="${GCS_BUCKET:-torchprime}"
IMAGE="${IMAGE:-${VERL_TPU_CI_IMAGE:-us-west2-docker.pkg.dev/tpu-pytorch/raycluster/verl-tpu:v20261006-tsync1006}}"

# --- Ray cluster sizing (default: 1 x 2x2x1 trainer slice + 1 x 2x2x1 rollout slice = 8 TPU v5p chips) ---
HEAD_CPU="${HEAD_CPU:-12}"
HEAD_MEMORY="${HEAD_MEMORY:-48Gi}"
WORKER_REPLICAS="${WORKER_REPLICAS:-2}"
HOSTS_PER_SLICE="${HOSTS_PER_SLICE:-1}"
WORKER_CPU="${WORKER_CPU:-48}"
WORKER_MEMORY="${WORKER_MEMORY:-400Gi}"
WORKER_TPU_CHIPS="${WORKER_TPU_CHIPS:-4}"
TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-v5p}"

# Per-role slice sizing (defaults inherit from WORKER_REPLICAS / HOSTS_PER_SLICE / WORKER_NODE_TOPOLOGY)
TRAINER_SLICE_REPLICAS="${TRAINER_SLICE_REPLICAS:-1}"
TRAINER_HOSTS_PER_SLICE="${TRAINER_HOSTS_PER_SLICE:-${HOSTS_PER_SLICE}}"
TRAINER_NODE_TOPOLOGY="${TRAINER_NODE_TOPOLOGY:-${WORKER_NODE_TOPOLOGY}}"

DEFAULT_ROLLOUT_SLICES=$(( WORKER_REPLICAS > TRAINER_SLICE_REPLICAS ? WORKER_REPLICAS - TRAINER_SLICE_REPLICAS : 1 ))
ROLLOUT_SLICE_REPLICAS="${ROLLOUT_SLICE_REPLICAS:-${DEFAULT_ROLLOUT_SLICES}}"
ROLLOUT_HOSTS_PER_SLICE="${ROLLOUT_HOSTS_PER_SLICE:-${HOSTS_PER_SLICE}}"
ROLLOUT_NODE_TOPOLOGY="${ROLLOUT_NODE_TOPOLOGY:-${WORKER_NODE_TOPOLOGY}}"

NNODES_TRAINER="${NNODES_TRAINER:-$((TRAINER_SLICE_REPLICAS * TRAINER_HOSTS_PER_SLICE))}"
N_CHIPS_TRAINER="${N_CHIPS_TRAINER:-${WORKER_TPU_CHIPS}}"
NNODES_ROLLOUT="${NNODES_ROLLOUT:-$((ROLLOUT_SLICE_REPLICAS * ROLLOUT_HOSTS_PER_SLICE))}"
N_CHIPS_ROLLOUT="${N_CHIPS_ROLLOUT:-${WORKER_TPU_CHIPS}}"

# --- Job & verification configuration ---
JOB_SCRIPT="${JOB_SCRIPT:-examples/tpu/grpo/run_qwen3_0_6b_torchtitan.sh}"
if [[ -n "${MODEL_PATH:-}" ]]; then
  DEFAULT_PREWARM_PATH="${MODEL_PATH}"
elif [[ "${JOB_SCRIPT}" == *"8b"* || "${JOB_SCRIPT}" == *"8B"* ]]; then
  DEFAULT_PREWARM_PATH="/data/jialei/assets/hf/Qwen3-8B"
elif [[ "${JOB_SCRIPT}" == *"4b"* || "${JOB_SCRIPT}" == *"4B"* ]]; then
  DEFAULT_PREWARM_PATH="/data/jialei/assets/hf/Qwen3-4B"
else
  DEFAULT_PREWARM_PATH="/data/jialei/assets/hf/Qwen3-0.6B"
fi
PREWARM_MODEL_PATH="${PREWARM_MODEL_PATH:-${DEFAULT_PREWARM_PATH}}"
SMOKE_TEST="${SMOKE_TEST:-0}"
VERIFY_LOG="${VERIFY_LOG:-1}"
KEEP_CLUSTER="${KEEP_CLUSTER:-0}"
LOCAL_PORT="${LOCAL_PORT:-$((18265 + RANDOM % 1000))}"
CLUSTER_READY_TIMEOUT="${CLUSTER_READY_TIMEOUT:-1200}"
if [[ "${SMOKE_TEST}" == "1" ]]; then
  DEFAULT_JOB_TIMEOUT=3600
else
  DEFAULT_JOB_TIMEOUT=14400
fi
JOB_TIMEOUT="${JOB_TIMEOUT:-${DEFAULT_JOB_TIMEOUT}}"
POLL_INTERVAL="${POLL_INTERVAL:-15}"

# --- Unique run identifier & DNS-safe JobSet name (<= 32 chars so pod hostnames stay <= 63 chars) ---
RUN_OWNER="$(echo "${USER:-user}" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9' | cut -c1-8)"
RUN_ID="${RUN_ID:-$(date +%m%d%H%M)-$(printf '%04x' $((RANDOM % 65536)))}"
DEFAULT_CLUSTER_NAME="verl-nap-${RUN_OWNER}-${RUN_ID}"
RAY_CLUSTER_NAME="${RAY_CLUSTER_NAME:-${DEFAULT_CLUSTER_NAME:0:32}}"
RAY_CLUSTER_NAME="${RAY_CLUSTER_NAME%-}"

LOG_DIR="${LOG_DIR:-/tmp/verl_nap_logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RAY_CLUSTER_NAME}-grpo.log}"
RENDERED_YAML="/tmp/${RAY_CLUSTER_NAME}-jobset.yaml"

USE_EXISTING_KUBECONFIG="${USE_EXISTING_KUBECONFIG:-0}"
if [[ "${USE_EXISTING_KUBECONFIG}" != "1" ]]; then
  export KUBECONFIG="/tmp/kubeconfig-${RAY_CLUSTER_NAME}"
fi

# Ensure `ray` CLI is on PATH (check pyenv installations if not already in PATH)
if ! command -v ray >/dev/null 2>&1; then
  for d in "${HOME}/.pyenv/versions"/*/bin "${HOME}/.pyenv/shims" "${HOME}/.local/bin"; do
    if [[ -x "${d}/ray" ]]; then
      export PATH="${d}:${PATH}"
      break
    fi
  done
fi

PF_PID=""
HEAD_POD=""
CLUSTER_APPLIED=0

stop_port_forward() {
  if [[ -n "${PF_PID}" ]] && kill -0 "${PF_PID}" 2>/dev/null; then
    kill "${PF_PID}" 2>/dev/null || true
    wait "${PF_PID}" 2>/dev/null || true
  fi
  PF_PID=""
}

ensure_port_forward() {
  if [[ -n "${PF_PID}" ]] && kill -0 "${PF_PID}" 2>/dev/null; then
    if curl -fsS --max-time 3 "http://127.0.0.1:${LOCAL_PORT}/api/version" >/dev/null 2>&1; then
      return 0
    fi
  fi
  stop_port_forward
  kubectl port-forward -n "${RAY_NAMESPACE}" "svc/${RAY_CLUSTER_NAME}-head-svc" "${LOCAL_PORT}:8265" >/dev/null 2>&1 &
  PF_PID=$!
  for _ in {1..20}; do
    if curl -fsS --max-time 2 "http://127.0.0.1:${LOCAL_PORT}/api/version" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "[NAP Runner] WARNING: Port-forward to svc/${RAY_CLUSTER_NAME}-head-svc:${LOCAL_PORT} did not respond within 20s." >&2
  return 1
}

cleanup() {
  local exit_code=$?
  echo ""
  echo "==================================================================================="
  echo "[NAP Runner] Cleaning up resources for ${RAY_CLUSTER_NAME} (exit_code=${exit_code})..."
  echo "==================================================================================="
  stop_port_forward
  rm -f "${RENDERED_YAML}"

  if [[ "${CLUSTER_APPLIED}" == "1" ]]; then
    if [[ "${KEEP_CLUSTER}" == "1" ]]; then
      echo "[NAP Runner] KEEP_CLUSTER=1 set; leaving JobSet ${RAY_CLUSTER_NAME} running."
      echo "[NAP Runner] To delete it manually when done, run:"
      echo "  kubectl delete jobset/${RAY_CLUSTER_NAME} svc/${RAY_CLUSTER_NAME}-head-svc -n ${RAY_NAMESPACE}"
    else
      # SAFETY: Delete ONLY the specific JobSet and Service created by this script.
      echo "[NAP Runner] Deleting JobSet ${RAY_CLUSTER_NAME} and Service ${RAY_CLUSTER_NAME}-head-svc in namespace ${RAY_NAMESPACE}..."
      kubectl delete jobset "${RAY_CLUSTER_NAME}" -n "${RAY_NAMESPACE}" --ignore-not-found=true --wait=true --timeout=180s || true
      kubectl delete service "${RAY_CLUSTER_NAME}-head-svc" -n "${RAY_NAMESPACE}" --ignore-not-found=true || true
      echo "[NAP Runner] Verified remaining pods for ${RAY_CLUSTER_NAME}:"
      kubectl get pods -n "${RAY_NAMESPACE}" -l "verl-nap/cluster=${RAY_CLUSTER_NAME}" --no-headers 2>/dev/null || true
    fi
  fi

  if [[ "${USE_EXISTING_KUBECONFIG}" != "1" && -n "${KUBECONFIG:-}" && -f "${KUBECONFIG}" ]]; then
    rm -f "${KUBECONFIG}"
  fi
  exit "${exit_code}"
}
trap cleanup EXIT INT TERM

echo "==================================================================================="
echo "[NAP Runner] GKE NAP TPU v5p GRPO Runner"
echo "  Project / Cluster : ${GCP_PROJECT} / ${GKE_CLUSTER} (${GKE_REGION})"
echo "  Namespace / Queue : ${RAY_NAMESPACE} / ${KUEUE_QUEUE}"
echo "  Ray JobSet Name   : ${RAY_CLUSTER_NAME} (owner=${RUN_OWNER}, run_id=${RUN_ID})"
echo "  Trainer Topology  : ${TRAINER_SLICE_REPLICAS} slice(s) x ${TRAINER_HOSTS_PER_SLICE} host(s) (${TRAINER_NODE_TOPOLOGY}, ${WORKER_TPU_CHIPS} chips/host)"
echo "  Rollout Topology  : ${ROLLOUT_SLICE_REPLICAS} slice(s) x ${ROLLOUT_HOSTS_PER_SLICE} host(s) (${ROLLOUT_NODE_TOPOLOGY}, ${WORKER_TPU_CHIPS} chips/host, ROLLOUT_TP=${ROLLOUT_TP:-all})"
echo "  Container Image   : ${IMAGE}"
echo "  Job Script        : ${JOB_SCRIPT} (SMOKE_TEST=${SMOKE_TEST}, prewarm=${PREWARM_MODEL_PATH})"
echo "  Local Log File    : ${LOG_FILE}"
echo "==================================================================================="

if [[ "${USE_EXISTING_KUBECONFIG}" != "1" ]]; then
  echo "[NAP Runner] Fetching GKE credentials into isolated KUBECONFIG=${KUBECONFIG}..."
  gcloud container clusters get-credentials "${GKE_CLUSTER}" \
    --region "${GKE_REGION}" \
    --project "${GCP_PROJECT}"
fi

# Render the JobSet + Service YAML template using Python string.Template
export RAY_CLUSTER_NAME RAY_NAMESPACE KUEUE_QUEUE PRIORITY_CLASS SERVICE_ACCOUNT
export HEAD_NODEPOOL TPU_NODE_SELECTOR_ACCEL WORKER_NODE_TOPOLOGY RESERVATION_NAME
export GCS_BUCKET IMAGE HEAD_CPU HEAD_MEMORY WORKER_REPLICAS HOSTS_PER_SLICE
export TRAINER_SLICE_REPLICAS TRAINER_HOSTS_PER_SLICE TRAINER_NODE_TOPOLOGY
export ROLLOUT_SLICE_REPLICAS ROLLOUT_HOSTS_PER_SLICE ROLLOUT_NODE_TOPOLOGY PREWARM_MODEL_PATH
export WORKER_CPU WORKER_MEMORY WORKER_TPU_CHIPS TPU_ACCELERATOR_TYPE RUN_ID RUN_OWNER

python3 - "${JOBSET_TEMPLATE}" "${RENDERED_YAML}" <<'PY'
import os
import pathlib
import string
import sys

tpl_path, out_path = sys.argv[1], sys.argv[2]
tpl = string.Template(pathlib.Path(tpl_path).read_text())
rendered = tpl.substitute(os.environ)
pathlib.Path(out_path).write_text(rendered)
PY

CLUSTER_APPLIED=1
if kubectl get jobset "${RAY_CLUSTER_NAME}" -n "${RAY_NAMESPACE}" >/dev/null 2>&1; then
  echo "[NAP Runner] JobSet ${RAY_CLUSTER_NAME} already exists in namespace ${RAY_NAMESPACE}; reusing it..."
else
  echo "[NAP Runner] Applying Ray Head Service and JobSet ${RAY_CLUSTER_NAME}..."
  kubectl apply -f "${RENDERED_YAML}"
fi

EXPECTED_WORKER_PODS=$(( TRAINER_SLICE_REPLICAS * TRAINER_HOSTS_PER_SLICE + ROLLOUT_SLICE_REPLICAS * ROLLOUT_HOSTS_PER_SLICE ))
EXPECTED_TOTAL_PODS=$((1 + EXPECTED_WORKER_PODS))
EXPECTED_TPU_CHIPS=$((EXPECTED_WORKER_PODS * WORKER_TPU_CHIPS))
EXPECTED_TPU_SLICES=$((TRAINER_SLICE_REPLICAS + ROLLOUT_SLICE_REPLICAS))

echo "[NAP Runner] Waiting up to ${CLUSTER_READY_TIMEOUT}s for Kueue admission, NAP node scale-up, and ${EXPECTED_TOTAL_PODS} pods (1 head + ${EXPECTED_WORKER_PODS} TPU workers)..."
StartWait=$(date +%s)
KueueLogged=0
LastPodSummary=""

while true; do
  Elapsed=$(( $(date +%s) - StartWait ))
  if (( Elapsed >= CLUSTER_READY_TIMEOUT )); then
    echo "[NAP Runner] ERROR: Timed out after ${Elapsed}s waiting for ${RAY_CLUSTER_NAME} pods."
    kubectl get jobset "${RAY_CLUSTER_NAME}" -n "${RAY_NAMESPACE}" -o yaml || true
    kubectl get pods -n "${RAY_NAMESPACE}" -l "verl-nap/cluster=${RAY_CLUSTER_NAME}" -o wide || true
    exit 1
  fi

  # 1. Check Kueue Workload admission status
  if (( KueueLogged == 0 )); then
    WlStatus=$(kubectl get workloads.kueue.x-k8s.io -n "${RAY_NAMESPACE}" \
      -l "kueue.x-k8s.io/job-uid=$(kubectl get jobset "${RAY_CLUSTER_NAME}" -n "${RAY_NAMESPACE}" -o jsonpath='{.metadata.uid}' 2>/dev/null || true)" \
      -o json 2>/dev/null | python3 -c '
import json, sys
try:
    items = json.load(sys.stdin).get("items", [])
    if not items:
        print("pending|Waiting for Kueue Workload creation")
        sys.exit(0)
    conds = {c["type"]: c for c in items[0].get("status", {}).get("conditions", [])}
    adm = conds.get("Admitted", {})
    qr = conds.get("QuotaReserved", {})
    if adm.get("status") == "True":
        msg = adm.get("message", "Admitted by Kueue")
        print(f"admitted|{msg}")
    else:
        msg = adm.get("message") or qr.get("message") or "Waiting for queue quota"
        print(f"queued|{msg}")
except Exception as e:
    print(f"unknown|{e}")
' || echo "unknown|")
    WlState="${WlStatus%%|*}"
    WlMsg="${WlStatus#*|}"
    if [[ "${WlState}" == "admitted" ]]; then
      echo "[NAP Runner] Kueue admitted ${RAY_CLUSTER_NAME} (${Elapsed}s): ${WlMsg}"
      KueueLogged=1
    elif (( Elapsed % 30 < 5 )); then
      echo "[NAP Runner] Kueue status (${Elapsed}s): [${WlState}] ${WlMsg}"
    fi
  fi

  # 2. Check Pod readiness (including gke-gcsfuse-sidecar) and fail fast on fatal pod errors
  PodCheck=$(kubectl get pods -n "${RAY_NAMESPACE}" -l "verl-nap/cluster=${RAY_CLUSTER_NAME}" -o json 2>/dev/null | python3 -c '
import json, sys
data = json.load(sys.stdin)
items = data.get("items", [])
ready_count = 0
head_pod = ""
summaries = []
fatal = ""
for p in items:
    name = p["metadata"]["name"]
    role = p["metadata"].get("labels", {}).get("verl-nap/role", "?")
    phase = p.get("status", {}).get("phase", "Pending")
    node = p.get("spec", {}).get("nodeName", "<unscheduled>")
    conds = {c["type"]: c["status"] for c in p.get("status", {}).get("conditions", [])}
    ready_str = conds.get("Ready", "False")
    is_ready = (phase == "Running" and ready_str == "True")
    if is_ready:
        ready_count += 1
        if role == "head":
            head_pod = name
    for cs in p.get("status", {}).get("containerStatuses", []) + p.get("status", {}).get("initContainerStatuses", []):
        cname = cs.get("name", "?")
        waiting = cs.get("state", {}).get("waiting", {})
        reason = waiting.get("reason", "")
        wmsg = waiting.get("message", "")
        if reason in ("ImagePullBackOff", "ErrImagePull", "CrashLoopBackOff", "CreateContainerConfigError"):
            fatal = f"Pod {name} container {cname} in {reason}: {wmsg}"
        terminated = cs.get("state", {}).get("terminated", {})
        ecode = terminated.get("exitCode", 0)
        if terminated and ecode != 0:
            fatal = f"Pod {name} container {cname} exited with code {ecode}"
    summaries.append(f"{name}({role}:{phase}/Ready={ready_str}@{node})")
print(json.dumps({
    "total": len(items),
    "ready": ready_count,
    "head_pod": head_pod,
    "fatal": fatal,
    "summary": ", ".join(summaries),
}))
')

  FatalErr=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["fatal"])' "${PodCheck}")
  ReadyCount=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["ready"])' "${PodCheck}")
  TotalCount=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["total"])' "${PodCheck}")
  HEAD_POD=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["head_pod"])' "${PodCheck}")
  PodSummary=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["summary"])' "${PodCheck}")

  if [[ -n "${FatalErr}" ]]; then
    echo "[NAP Runner] ERROR: ${FatalErr}"
    for p in $(kubectl get pods -n "${RAY_NAMESPACE}" -l "verl-nap/cluster=${RAY_CLUSTER_NAME}" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null); do
      echo "--- Logs for pod ${p} ---"
      kubectl logs -n "${RAY_NAMESPACE}" "${p}" --all-containers=true --tail=50 || true
    done
    kubectl describe pods -n "${RAY_NAMESPACE}" -l "verl-nap/cluster=${RAY_CLUSTER_NAME}" || true
    exit 1
  fi

  if [[ "${PodSummary}" != "${LastPodSummary}" ]] || (( Elapsed % 30 == 0 )); then
    echo "[NAP Runner] Pods (${Elapsed}s): ${ReadyCount}/${EXPECTED_TOTAL_PODS} ready (created=${TotalCount}) | ${PodSummary:-none}"
    LastPodSummary="${PodSummary}"
  fi

  if (( ReadyCount == EXPECTED_TOTAL_PODS )) && [[ -n "${HEAD_POD}" ]]; then
    echo "[NAP Runner] All ${EXPECTED_TOTAL_PODS} pods are Running & Ready (Head pod: ${HEAD_POD})."
    break
  fi

  sleep 5
done

# 3. Port-forward Ray Dashboard / Jobs API (note: bodaborg-v5p-nap grants pods/portforward and pods/log, not pods/exec)
ensure_port_forward
echo "[NAP Runner] Ray Dashboard / Jobs API port-forwarded to http://127.0.0.1:${LOCAL_PORT} (PID=${PF_PID})"

# 4. Wait for Ray head + TPU workers to register all TPU chips and tpu-group-* slices via HTTP API
echo "[NAP Runner] Waiting for Ray cluster to register ${EXPECTED_TPU_CHIPS} TPU chips across ${EXPECTED_TPU_SLICES} slices..."
while true; do
  Elapsed=$(( $(date +%s) - StartWait ))
  if (( Elapsed >= CLUSTER_READY_TIMEOUT )); then
    echo "[NAP Runner] ERROR: Timed out waiting for ${EXPECTED_TPU_CHIPS} TPUs in Ray cluster."
    exit 1
  fi

  ensure_port_forward
  ClusterStatusJson=$(curl -fsS --max-time 5 "http://127.0.0.1:${LOCAL_PORT}/api/cluster_status?format=text" 2>/dev/null || echo "{}")
  ParsedCluster=$(python3 -c '
import json, sys
try:
    d = json.loads(sys.argv[1])
    usage = d.get("data", {}).get("clusterStatus", {}).get("loadMetricsReport", {}).get("usage", {})
    tpus = int(usage.get("TPU", [0, 0])[1])
    slices = sum(1 for k in usage if k.startswith("tpu-group-"))
    summary = d.get("data", {}).get("autoscalingStatus", "").replace("\n", " | ")
    print(f"{tpus}|{slices}|{summary}")
except Exception:
    print("0|0|")
' "${ClusterStatusJson}")

  RegisteredTPUs="${ParsedCluster%%|*}"
  RestCluster="${ParsedCluster#*|}"
  RegisteredSlices="${RestCluster%%|*}"
  ClusterSummary="${RestCluster#*|}"

  if (( RegisteredTPUs >= EXPECTED_TPU_CHIPS && RegisteredSlices >= EXPECTED_TPU_SLICES )); then
    echo "[NAP Runner] Ray cluster ready! ${ClusterSummary}"
    break
  fi

  echo "[NAP Runner] Waiting for Ray workers (${Elapsed}s): ${RegisteredTPUs}/${EXPECTED_TPU_CHIPS} TPUs, ${RegisteredSlices}/${EXPECTED_TPU_SLICES} tpu-group slices..."
  sleep 5
done

# 5. Build runtime-env JSON and submit the GRPO Ray job via port-forwarded Ray Jobs API
TOTAL_TRAINER_CHIPS=$((NNODES_TRAINER * N_CHIPS_TRAINER))
if [[ "${SMOKE_TEST}" == "1" ]]; then
  # Match TPU CI smoke test overrides so 5 steps produce non-zero rewards and finite positive grad_norm
  MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-1024}"
  ROLLOUT_N="${ROLLOUT_N:-4}"
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${TOTAL_TRAINER_CHIPS}}"
  PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-${TOTAL_TRAINER_CHIPS}}"
  VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
  VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-32}"
  TEST_FREQ="${TEST_FREQ:-5}"
  TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-5}"
fi

RuntimeEnvJson=$(python3 - <<PY
import json
env_vars = {
    "TPU_ACCELERATOR_TYPE": "${TPU_ACCELERATOR_TYPE}",
    "OMP_NUM_THREADS": "4",
    "SMOKE_TEST": "${SMOKE_TEST}",
    "NNODES_TRAINER": "${NNODES_TRAINER}",
    "N_CHIPS_TRAINER": "${N_CHIPS_TRAINER}",
    "NNODES_ROLLOUT": "${NNODES_ROLLOUT}",
    "N_CHIPS_ROLLOUT": "${N_CHIPS_ROLLOUT}",
}
for k, v in {
    "ROLLOUT_TP": "${ROLLOUT_TP:-}",
    "MODEL_PATH": "${MODEL_PATH:-}",
    "MAX_RESPONSE_LEN": "${MAX_RESPONSE_LEN:-}",
    "ROLLOUT_N": "${ROLLOUT_N:-}",
    "TRAIN_BATCH_SIZE": "${TRAIN_BATCH_SIZE:-}",
    "PPO_MINI_BATCH_SIZE": "${PPO_MINI_BATCH_SIZE:-}",
    "VAL_BATCH_SIZE": "${VAL_BATCH_SIZE:-}",
    "VAL_MAX_SAMPLES": "${VAL_MAX_SAMPLES:-}",
    "TEST_FREQ": "${TEST_FREQ:-}",
    "TOTAL_TRAINING_STEPS": "${TOTAL_TRAINING_STEPS:-}",
}.items():
    if v:
        env_vars[k] = v
print(json.dumps({"env_vars": env_vars, "excludes": [".git", "tests", "docs", ".github"]}))
PY
)

echo "[NAP Runner] Submitting Ray job (${JOB_SCRIPT}) to http://127.0.0.1:${LOCAL_PORT}..."
echo "[NAP Runner] RuntimeEnv: ${RuntimeEnvJson}"

ensure_port_forward
SubmitOut=$(ray job submit \
  --address "http://127.0.0.1:${LOCAL_PORT}" \
  --working-dir "${REPO_ROOT}" \
  --runtime-env-json "${RuntimeEnvJson}" \
  --no-wait \
  -- bash "${JOB_SCRIPT}" 2>&1)
echo "${SubmitOut}"

JobId=$(echo "${SubmitOut}" | grep -oE "raysubmit_[a-zA-Z0-9]+" | head -n 1 || true)
if [[ -z "${JobId}" ]]; then
  echo "[NAP Runner] ERROR: Failed to parse Ray Job ID from submission output."
  exit 1
fi
echo "[NAP Runner] Submitted Job ID: ${JobId}"

# 6. Poll job status via Ray HTTP API (auto-healing port-forward if needed) and stream progress
JobStart=$(date +%s)
JobState="PENDING"
while (( $(date +%s) - JobStart < JOB_TIMEOUT )); do
  ensure_port_forward || true
  JobInfoJson=$(curl -fsS --max-time 10 "http://127.0.0.1:${LOCAL_PORT}/api/jobs/${JobId}" 2>/dev/null || echo "{}")
  Status=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("status", "UNKNOWN"))' "${JobInfoJson}" 2>/dev/null || echo "UNKNOWN")
  JobElapsed=$(( $(date +%s) - JobStart ))

  if [[ "${Status}" == "SUCCEEDED" ]]; then
    JobState="SUCCEEDED"
    echo "[NAP Runner] Job ${JobId} SUCCEEDED after ${JobElapsed}s."
    break
  elif [[ "${Status}" == "FAILED" || "${Status}" == "STOPPED" ]]; then
    JobState="${Status}"
    echo "[NAP Runner] ERROR: Job ${JobId} ended with ${Status} after ${JobElapsed}s:"
    echo "${JobInfoJson}"
    break
  fi

  # Fetch latest progress line from /api/jobs/${JobId}/logs
  LatestProgress=$(curl -fsS --max-time 10 "http://127.0.0.1:${LOCAL_PORT}/api/jobs/${JobId}/logs" 2>/dev/null | python3 -c '
import json, re, sys
try:
    logs = json.load(sys.stdin).get("logs", "")
    matches = [line for line in logs.splitlines() if re.search(r"step:|actor/grad_norm|val-core/openai/gsm8k|RAIDEN PARITY|Rollout|Initial", line)]
    print(matches[-1][:160] if matches else "")
except Exception:
    print("")
' || true)
  echo "[NAP Runner] Job ${JobId} [${Status}] (${JobElapsed}s/${JOB_TIMEOUT}s)... ${LatestProgress}"
  sleep "${POLL_INTERVAL}"
done

if [[ "${JobState}" == "PENDING" ]]; then
  echo "[NAP Runner] ERROR: Job ${JobId} timed out after ${JOB_TIMEOUT}s. Stopping job..."
  ensure_port_forward || true
  curl -fsS -X POST "http://127.0.0.1:${LOCAL_PORT}/api/jobs/${JobId}/stop" >/dev/null 2>&1 || true
fi

# 7. Pull full Ray job logs to local disk via HTTP API
echo "[NAP Runner] Pulling full Ray job logs for ${JobId} to ${LOG_FILE}..."
ensure_port_forward
curl -fsS --max-time 60 "http://127.0.0.1:${LOCAL_PORT}/api/jobs/${JobId}/logs" | python3 -c '
import json, sys
data = json.load(sys.stdin)
sys.stdout.write(data.get("logs", ""))
' > "${LOG_FILE}"

LogLines=$(wc -l < "${LOG_FILE}" || echo 0)
echo "[NAP Runner] Saved ${LogLines} lines to ${LOG_FILE}. Last 80 lines:"
echo "-----------------------------------------------------------------------------------"
tail -n 80 "${LOG_FILE}" || true
echo "-----------------------------------------------------------------------------------"

if [[ "${JobState}" != "SUCCEEDED" ]]; then
  echo "[NAP Runner] ERROR: Ray job ${JobId} did not succeed (state=${JobState}). Full log: ${LOG_FILE}"
  exit 1
fi

if [[ "${VERIFY_LOG}" == "1" ]]; then
  echo "[NAP Runner] Verifying GRPO metrics in ${LOG_FILE}..."
  python3 "${REPO_ROOT}/tests/special_tpu/verify_tpu_e2e_log.py" grpo "${LOG_FILE}" "${SMOKE_TEST}"
fi

echo "[NAP Runner] SUCCESS! Log saved to ${LOG_FILE}. Proceeding to cluster teardown via EXIT trap..."
