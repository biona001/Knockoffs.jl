#!/bin/bash
#
#SBATCH --job-name=fig3
#
#SBATCH --array=1-3600
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --partition=owners,normal,candes,zihuai
#SBATCH --output=/home/groups/sabatti/.julia/dev/Knockoffs/linear_cholesky_updates/joblogs/fig3-%A_%a.out
#SBATCH --chdir=/home/groups/sabatti/.julia/dev/Knockoffs/linear_cholesky_updates

#save job info on joblog:
echo "Job ${SLURM_JOB_ID:-$JOB_ID} task ${SLURM_ARRAY_TASK_ID} started on:   " `hostname -s`
echo "Job ${SLURM_JOB_ID:-$JOB_ID} task ${SLURM_ARRAY_TASK_ID} started on:   " `date `

# load the job environment:
ml julia/1.11.4 R/4.0.2 java/11.0.11 python/3.9.0 openssl/3.0.7 system
ml julia/1.11.4 R/4.0.2 cmake/3.24.2 harfbuzz/1.4.8 fribidi/1.0.12 libgit2/1.1.0 openssl/3.0.7 
export JULIA_DEPOT_PATH="/home/groups/sabatti/.julia"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export FIG3_P="${FIG3_P:-5000}"
export FIG3_COVARIANCE_STRUCTURES="${FIG3_COVARIANCE_STRUCTURES:-AR1,ER,block,stress}"
export FIG3_WORKERS="${FIG3_WORKERS:-32}"
export WINDOW_CORR_TOL="${WINDOW_CORR_TOL:-0.8}"
export MIN_WINDOW_SIZE="${MIN_WINDOW_SIZE:-150}"
export FACTOR_CHECK="${FACTOR_CHECK:-false}"

# run code (first print the command you run, then run the command)
WORKDIR="${WORKDIR:-/home/groups/sabatti/.julia/dev/Knockoffs/linear_cholesky_updates}"
SCRIPT_DIR="${WORKDIR}/fig3"
PROJECT_DIR="$(dirname "${WORKDIR}")"
mkdir -p "${WORKDIR}/joblogs" "${SCRIPT_DIR}/results"
cmd="julia --project=${PROJECT_DIR} ${SCRIPT_DIR}/fig3_worker.jl ${SLURM_ARRAY_TASK_ID}"
echo "$cmd"
$cmd

#echo job info on joblog:
echo "Job ${SLURM_JOB_ID:-$JOB_ID} task ${SLURM_ARRAY_TASK_ID} ended on:   " `hostname -s`
echo "Job ${SLURM_JOB_ID:-$JOB_ID} task ${SLURM_ARRAY_TASK_ID} ended on:   " `date `
#echo " "
