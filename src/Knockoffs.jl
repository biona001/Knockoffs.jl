module Knockoffs

using LinearAlgebra
using Statistics
using Convex
using SCS
import Base: eltype, getindex, size

export fixed_knockoffs, modelX_gaussian_knockoffs, normalize_col!,
    coefficient_diff, threshold, extract_beta, extract_combine_beta,
    partition, rapid, snpknock2

include("struct.jl")
include("fixed.jl")
include("modelX.jl")
include("dmc.jl")
include("feature_stats.jl")
include("threshold.jl")
include("hmm_wrapper.jl")

end