module Knockoffs

using LinearAlgebra
using Statistics
using Convex
using SCS
using SnpArrays
using DelimitedFiles
using ProgressMeter
using Distributions
using CSV
using DataFrames
using fastPHASE
using ElasticArrays
using Random

import Base: eltype, getindex, size

export fixed_knockoffs, modelX_gaussian_knockoffs, standardize!,
    coefficient_diff, threshold, extract_beta, extract_combine_beta,
    partition, rapid, snpknock2, decorrelate_knockoffs,
    process_fastphase_output, fastphase, 
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
    merge_knockoffs_with_original

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

# test data directory
datadir(parts...) = joinpath(@__DIR__, "..", "data", parts...)    

end
