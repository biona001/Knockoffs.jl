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
using ElasticArrays
using Random
using PositiveFactorizations
using CovarianceEstimation
using StatsBase: sample, cov2cor, cor2cov, cov2cor!, cor2cov!, countmap, zscore!, zscore
using GLMNet
using BlockDiagonals
using Roots: fzero
using Downloads
using GLM
using Reexport
using LoopVectorization: @turbo # speeding up cholesky updates in utilities.jl
using Optim: optimize, Brent # for group knockoffs
using Clustering: hclust, cutree

@reexport using GLM

export 
    # functions that generate knockoffs
    fixed_knockoffs,
    modelX_gaussian_knockoffs,
    modelX_gaussian_group_knockoffs,
    modelX_gaussian_rep_group_knockoffs,
    approx_modelX_gaussian_knockoffs,
    ipad,
    # solvers
    solve_s, 
    solve_s_group,
    solve_s_graphical_group,
    # specific solvers
    solve_equi, solve_max_entropy, solve_sdp_ccd, solve_SDP, solve_sdp_ccd, solve_MVR,
    solve_group_equi, solve_group_max_entropy_hybrid, 
    solve_group_mvr_hybrid, solve_group_sdp_hybrid, 
    # utilities for running knockoff filter
    mk_threshold,
    threshold, 
    MK_statistics,
    # functions related to fitting
    fit_lasso, 
    fit_marginal, 
    debias!, 
    # functions for prediction routine after lasso fit
    predict, R2, auc, 
    # functions for hmm (experimental)
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
    simulate_AR1,
    simulate_ER,
    simulate_block_covariance, 
    download_1000genomes,
    hc_partition_groups, 
    choose_group_reps,
    normalize_col, 
    normalize_col!,
    group_block_objective,
    cond_indep_corr

include("struct.jl")
include("fixed.jl")
include("modelX.jl")
include("threshold.jl")
include("utilities.jl")
include("knockoffscreen.jl")
include("fit_lasso.jl")
include("approx.jl")
include("predict.jl")
include("group.jl")
include("ipad.jl")
include("R.jl")
include("experimental/hmm_wrapper.jl")
include("experimental/hmm.jl")
include("experimental/dmc.jl")

const SINGLE_KNOCKOFFS = [:mvr, :maxent, :equi, :sdp, :sdp_ccd]
const GROUP_KNOCKOFFS = [:equi, :sdp_subopt, :sdp, :sdp_block, :sdp_full, :mvr, :mvr_block, :maxent, :maxent_block]

# test data directory
datadir(parts...) = joinpath(@__DIR__, "..", "data", parts...)

end
