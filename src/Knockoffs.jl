module Knockoffs

using LinearAlgebra
using Statistics
using Convex
using SCS
import Base: eltype, getindex, size

export knockoff_equi, knockoff_sdp, normalize_col!

include("struct.jl")
end
