module Knockoffs

using LinearAlgebra
using Statistics
using Convex
using SCS

export knockoff_equi, knockoff_sdp, normalize_col!

include("struct.jl")
end
