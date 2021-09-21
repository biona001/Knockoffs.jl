module Knockoffs

using LinearAlgebra
using Statistics
using Convex
using SCS
import Base: eltype, getindex, size

export fixed_knockoffs, modelX_gaussian_knockoffs, normalize_col!

include("struct.jl")
include("fixed.jl")
include("modelX.jl")

end
