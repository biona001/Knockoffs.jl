module Knockoffs

using LinearAlgebra
using Statistics
using JuMP
using Hypatia
using SnpArrays
using DelimitedFiles
using ProgressMeter
using Distributions
using CSV
using DataFrames
using fastPHASE
using ElasticArrays
using Random
using PositiveFactorizations
using CovarianceEstimation
using StatsBase: sample, cov2cor, cor2cov, cov2cor!, cor2cov!, countmap
using GLMNet
using BlockDiagonals
using Roots: fzero
using Downloads
using GLM
using Reexport
using LoopVectorization: @turbo # speeding up cholesky updates in utilities.jl
using Ipopt
using SCS
using Optim: optimize, Brent # for group knockoffs
using Clustering: hclust, cutree
using LowRankApprox: id

@reexport using GLM

export 
    # functions that generate knockoffs
    fixed_knockoffs,
    modelX_gaussian_knockoffs,
    ghost_knockoffs,
    modelX_gaussian_group_knockoffs,
    modelX_gaussian_rep_group_knockoffs,
    approx_modelX_gaussian_knockoffs,
    ipad,
    full_knockoffscreen,
    # solvers
    solve_s, 
    solve_s_group,
    # utilities for running knockoff filter
    coefficient_diff, 
    mk_threshold,
    threshold, 
    extract_beta,
    # functions related to fitting lasso
    fit_lasso, 
    debias!, 
    # functions for prediction routine after lasso fit
    predict, R2, auc, 
    # functions for hmm
    get_haplotype_transition_matrix, get_genotype_transition_matrix, 
    get_initial_probabilities, genotype_knockoffs,
    GenotypeState, MarkovChainTable, pair_to_index, index_to_pair,
    hmm_knockoff, forward_backward_sampling, forward_backward_sampling!,
    form_emission_prob_matrix, get_genotype_emission_probabilities,
    get_haplotype_emission_probabilities, markov_knockoffs, markov_knockoffs!,
    sample_markov_chain, sample_markov_chain!,
    update_normalizing_constants!, single_state_dmc_knockoff!,
    rapid, snpknock2, 
    process_fastphase_output, fastphase, 
    # diagnostics
    compare_correlation, compare_pairwise_correlation,
    # utilities
    merge_knockoffs_with_original,
    simulate_AR1,
    download_1000genomes,
    simulate_block_covariance, 
    hc_partition_groups, 
    id_partition_groups,
    normalize_col, 
    normalize_col!

include("struct.jl")
include("fixed.jl")
include("modelX.jl")
include("dmc.jl")
include("feature_stats.jl")
include("threshold.jl")
include("hmm_wrapper.jl")
include("hmm.jl")
include("utilities.jl")
include("knockoffscreen.jl")
include("fit_lasso.jl")
include("approx.jl")
include("ghost.jl")
include("predict.jl")
include("group.jl")
include("ipad.jl")

const SINGLE_KNOCKOFFS = [:mvr, :maxent, :equi, :sdp, :sdp_ccd]
const GROUP_KNOCKOFFS = [:equi, :sdp_subopt, :sdp, :sdp_ccd, :sdp_full, :mvr, :maxent, :maxent_subopt]

# test data directory
datadir(parts...) = joinpath(@__DIR__, "..", "data", parts...)

end
