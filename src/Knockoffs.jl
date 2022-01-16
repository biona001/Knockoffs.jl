module Knockoffs

using LinearAlgebra
using Statistics
using Convex
using SCS
using SnpArrays
using DelimitedFiles
using ProgressMeter
import Base: eltype, getindex, size

export fixed_knockoffs, modelX_gaussian_knockoffs, normalize_col!,
    coefficient_diff, threshold, extract_beta, extract_combine_beta,
    partition, rapid, snpknock2, decorrelate_knockoffs,
    process_fastphase_output, 
    # functions for hmm
    get_haplotype_transition_matrix, get_genotype_transition_matrix, 
    get_initial_probabilities,
    GenotypeState, MarkovChainTable, pair_to_index, index_to_pair,
    hmm_knockoff, forward_backward_sampling, forward_backward_sampling!,
    form_emission_prob_matrix, get_genotype_emission_probabilities,
    get_haplotype_emission_probabilities

include("struct.jl")
include("fixed.jl")
include("modelX.jl")
include("dmc.jl")
include("feature_stats.jl")
include("threshold.jl")
include("hmm_wrapper.jl")

end
