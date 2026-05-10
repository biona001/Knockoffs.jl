#!/bin/bash

set -euo pipefail

WORKDIR="${WORKDIR:-/home/groups/sabatti/.julia/dev/Knockoffs/linear_cholesky_updates}"
SCRIPT_DIR="${WORKDIR}/fig2"
PROJECT_DIR="$(dirname "${WORKDIR}")"
JOBLOG_DIR="${WORKDIR}/joblogs"
RESULT_DIR="${SCRIPT_DIR}/results"

submit_one() {
    local strategy="$1"
    local cpus="$2"
    local mem="$3"
    local ntasks=270

    mkdir -p "${JOBLOG_DIR}" "${RESULT_DIR}"
    echo "Submitting fig2-${strategy}: ${ntasks} tasks, ${cpus} CPU(s), ${mem} total memory"

    sbatch \
        --job-name="fig2-${strategy}" \
        --array="1-${ntasks}" \
        --time="48:00:00" \
        --cpus-per-task="${cpus}" \
        --mem="${mem}" \
        --partition="owners,normal,candes,zihuai" \
        --chdir="${WORKDIR}" \
        --output="${JOBLOG_DIR}/fig2-${strategy}-%A_%a.out" \
        --export="ALL,FIG2_UPDATE_STRATEGY=${strategy},WORKDIR=${WORKDIR},PROJECT_DIR=${PROJECT_DIR},SCRIPT_DIR=${SCRIPT_DIR},RESULT_DIR=${RESULT_DIR},JULIA_NUM_THREADS=${cpus},FIG2_SKIP_P20000_SERIAL=${FIG2_SKIP_P20000_SERIAL:-false}" \
        "${SCRIPT_DIR}/submit_fig2.sh" --worker
}

if [[ "${1:-}" != "--worker" ]]; then
    submit_one standard 1 32G
    submit_one early 1 32G
    submit_one parallel2 2 32G
    submit_one parallel4 4 32G
    submit_one parallel8 8 32G
    submit_one parallel16 16 32G
    submit_one parallel32 32 32G
    exit 0
fi

echo "Job ${SLURM_JOB_ID:-$JOB_ID} task ${SLURM_ARRAY_TASK_ID} (${FIG2_UPDATE_STRATEGY}) started on: " `hostname -s`
echo "Job ${SLURM_JOB_ID:-$JOB_ID} task ${SLURM_ARRAY_TASK_ID} (${FIG2_UPDATE_STRATEGY}) started on: " `date `

ml julia/1.11.4 R/4.0.2 java/11.0.11 python/3.9.0 openssl/3.0.7 system
ml julia/1.11.4 R/4.0.2 cmake/3.24.2 harfbuzz/1.4.8 fribidi/1.0.12 libgit2/1.1.0 openssl/3.0.7
export JULIA_DEPOT_PATH="/home/groups/sabatti/.julia"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${JULIA_NUM_THREADS:-1}}"
export FIG2_SKIP_P20000_SERIAL="${FIG2_SKIP_P20000_SERIAL:-false}"

cd "${WORKDIR}"
mkdir -p "${JOBLOG_DIR}" "${RESULT_DIR}"

cmd="julia --project=${PROJECT_DIR} ${SCRIPT_DIR}/fig2_worker.jl ${SLURM_ARRAY_TASK_ID}"
echo "$cmd"
set +e
$cmd
exit_code=$?
set -e

result_file="${RESULT_DIR}/fig2_${FIG2_UPDATE_STRATEGY}_task${SLURM_ARRAY_TASK_ID}.csv"
if [ "$exit_code" -ne 0 ] && [ ! -s "$result_file" ]; then
    now="$(date -Is)"
    msg="worker exited with code ${exit_code}; no Julia result file was written"
    printf 'task_id,timestamp,status,covariance,p,method_name,update_strategy,method,nworkers,robust,rep,seed,julia_threads,elapsed_sec,error_message\n' > "$result_file"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"%s"\n' \
        "${SLURM_ARRAY_TASK_ID}" "$now" "failed_shell" "" "" "" "${FIG2_UPDATE_STRATEGY}" "" "" "" "" "" "${JULIA_NUM_THREADS}" "" "$msg" >> "$result_file"
    echo "Wrote shell failure marker to $result_file"
fi

echo "Job ${SLURM_JOB_ID:-$JOB_ID} task ${SLURM_ARRAY_TASK_ID} (${FIG2_UPDATE_STRATEGY}) ended on: " `hostname -s`
echo "Job ${SLURM_JOB_ID:-$JOB_ID} task ${SLURM_ARRAY_TASK_ID} (${FIG2_UPDATE_STRATEGY}) ended on: " `date `
exit "$exit_code"
