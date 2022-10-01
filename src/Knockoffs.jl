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
using StatsBase
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

@reexport using GLM

export knockoff_filter, 
    fixed_knockoffs, modelX_gaussian_knockoffs, normalize_col, normalize_col!,
    coefficient_diff, threshold, extract_beta,
    partition, rapid, snpknock2, decorrelate_knockoffs,
    process_fastphase_output, fastphase, 
    approx_modelX_gaussian_knockoffs,
    ghost_knockoffs,
    modelX_gaussian_group_knockoffs,
    solve_s, solve_s_group,
    # constructors
    knockoff,
    # functions related to fitting lasso
    fit_lasso, debias!, 
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
    # knockoffscreen knockoffs
    full_knockoffscreen,
    # diagnostics
    compare_correlation, compare_pairwise_correlation,
    # utilities
    merge_knockoffs_with_original, simulate_AR1,
    download_1000genomes,
    simulate_block_covariance

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

# test data directory
datadir(parts...) = joinpath(@__DIR__, "..", "data", parts...)    

end
