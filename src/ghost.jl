# This file contains experimental code implementing the GhostKnockoff methodology 
# described in this paper: https://www.nature.com/articles/s41467-022-34932-z.
# An improved version of GhostKnockoffs is implemented in the package 
# GhostKnockoffGWAS https://github.com/biona001/GhostKnockoffGWAS

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
