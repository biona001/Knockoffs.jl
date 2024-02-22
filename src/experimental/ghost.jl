# This file contains experimental code implementing the GhostKnockoff methodology 
# described in this paper: https://www.nature.com/articles/s41467-022-34932-z.
# An improved version of GhostKnockoffs is implemented in the package 
# GhostKnockoffGWAS https://github.com/biona001/GhostKnockoffGWAS, thus the 
# code here is archived and shouldn't be used.

"""
    ghost_knockoffs(Zscores, D, Σinv; [m=1])
    ghost_knockoffs(Zscores, Z_pos, H_pos, H, method; [windowsize], [covariance_approximator], [kwargs])

Generate Ghost knockoffs given a list of z-scores (GWAS summary statistic). 

# Inputs
+ `Zscores`: List of z-score statistics
+ `D`: Matrix obtained from solving the knockoff problem satisfying 
    `(m+1)/m*Σ - D ⪰ 0`
+ `Σinv`: Inverse of the covariance matrix
+ `Z_pos`: A sorted list of SNP position for each SNP in `Zscores`
+ `H_pos`: A sorted list of SNP position in the reference panel `H`
+ `H`: A haplotype reference panel. Each row is a sample and each column is a variant.
+ `method`: Can be any of the method in [`approx_modelX_gaussian_knockoffs`](@ref)
+ `windowsize`: Number of covariates to be included in a block. Each block consists of
    adjacent variables. The last block could contain less than `windowsize` variables. 
+ `covariance_approximator`: A covariance estimator, defaults to 
    `LinearShrinkage(DiagonalUnequalVariance(), :lw)`. See 
    CovarianceEstimation.jl for more options.
+ `kwargs...`: Possible optional inputs to solvers specified in `method`, see 
    [`solve_MVR`](@ref), [`solve_max_entropy`](@ref), and [`solve_sdp_ccd`](@ref)

# optional inputs
+ `m`: Number of knockoffs

# Reference
He, Z., Liu, L., Belloy, M. E., Le Guen, Y., Sossin, A., Liu, X., ... & Ionita-Laza, I. (2021). 
Summary statistics knockoff inference empowers identification of putative causal variants in 
genome-wide association studies. 
"""
function ghost_knockoffs(Zscores::AbstractVector{T}, D::AbstractMatrix{T}, 
    Σinv::AbstractMatrix{T}; m::Int = 1) where T
    p = size(D, 1)
    length(Zscores) == size(Σinv, 1) == size(Σinv, 2) == p || 
        error("Dimension mismatch")
    DΣinv = D * Σinv
    C = 2D - DΣinv * D
    v = sample_mvn_efficient(C, D, m) # Jiaqi's trick
    P = repeat(I - DΣinv, m)
    return P*Zscores + v
end

"""
    sample_mvn_efficient(C::AbstractMatrix{T}, D::AbstractMatrix{T}, m::Int)

Efficiently samples from `N(0, A)` where
```math
\\begin{aligned}
A &= \\begin{pmatrix}
    C & C-D & \\cdots & C-D\\\\
    C-D & C & \\cdots & C-D\\\\
    \\vdots & & \\ddots & \\vdots\\\\
    C-D & C-D & & C
\\end{pmatrix}
\\end{aligned}
```
Note there are `m` blocks per row/col
"""
function sample_mvn_efficient(C::AbstractMatrix{T}, D::AbstractMatrix{T}, m::Int) where T
    p = size(C, 1)
    L = cholesky(Symmetric(C - (m-1)/m * D))
    e1 = randn(p)
    e2 = Vector{T}[]
    d = MvNormal(Symmetric(D))
    for i in 1:m
        push!(e2, rand(d))
    end
    e2_avg = 1/m * sum(e2)
    Zko = T[]
    for i in 1:m
        append!(Zko, L.L*e1 + e2[i] - e2_avg)
    end
    return Zko
end

function ghost_knockoffs(
    Zscores,
    Z_pos,
    H_pos,
    H,
    method::Symbol; 
    windowsize::Int = 500,
    covariance_approximator=LinearShrinkage(DiagonalUnequalVariance(), :lw),
    kwargs...
    )
    # first match SNPs in Z to those in H
    Z2H_idx = match_Z_to_H(Z_pos, H_pos)
    # preallocated variables
    p = length(Z_pos)
    windows = ceil(Int, p / windowsize)
    block_covariances = Vector{Matrix{Float64}}(undef, windows)
    block_s = Vector{Vector{Float64}}(undef, windows)
    Hstorage = zeros(size(H, 1), windowsize)
    pmeter = Progress(windows, 1, "Approximating covariance by blocks...")
    # solve for s in each block of Σ
    for window in 1:windows
        # current window range
        cur_range = window == windows ? 
            ((windows - 1)*windowsize + 1:p) : 
            ((window - 1)*windowsize + 1:window * windowsize)
        copyto!(Hstorage, @view(H[:, Z2H_idx[cur_range]]))
        # approximate a block of Σ
        Σcur = cov(covariance_approximator, @view(Hstorage[:, 1:length(cur_range)]))
        # solve for s vector
        scur = solve_s(Σcur, method; kwargs...)
        # save result
        block_covariances[window] = Σcur
        block_s[window] = scur
        next!(pmeter)
    end
    # assemble block diagonal Σ, s, and other variables
    Σ = BlockDiagonal(block_covariances)
    s = vcat(block_s...)
    Σinv = inv(Σ)
    D = Diagonal(s)
    DΣinv = D * Σinv
    P = I - DΣinv
    μ = P * Zscores
    V = Symmetric(2D - DΣinv * D)
    # generate ghost knockoffs
    Z̃ = rand(MvNormal(μ, V))
    return Z̃
end

function match_Z_to_H(Z_pos::AbstractVector{Int}, H_pos::AbstractVector{Int})
    issorted(Z_pos) || error("Z_pos not sorted!")
    issorted(H_pos) || error("H_pos not sorted!")
    # find all Zj that can be matched to H
    matched_idx = indexin(Z_pos, H_pos)
    # for Zj that can't be mathced, find a SNP in H that is closest to Zj
    for i in eachindex(matched_idx)
        if isnothing(matched_idx[i])
            matched_idx[i] = searchsortednearest(H_pos, Z_pos[i])
        end
    end
    return Vector{Int}(matched_idx)
end

# reference is assumed sorted
# adapted from https://discourse.julialang.org/t/findnearest-function/4143/5
function searchsortednearest(reference::Vector{Int}, x::Int)
    idx = searchsortedfirst(reference, x)
    idx == 1 && return idx
    idx > length(reference) && return length(reference)
    reference[idx]==x && return idx
    return abs(reference[idx]-x) < abs(reference[idx-1]-x) ? idx : idx - 1
end
