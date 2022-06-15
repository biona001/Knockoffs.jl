"""
    approx_modelX_gaussian_knockoffs(X, method; windowsize = 500, kwargs...)

Generates Gaussian knockoffs by approximating the covariance as a block diagonal matrix. 
Each block contains `windowsize` consecutive features. 

# Inputs
+ `X`: A `n × p` numeric matrix, each row is a sample, and each column is covariate.
+ `method`: Can be one of the following
    * `:mvr` for minimum variance-based reconstructability knockoffs (alg 1 in ref 2)
    * `:maxent` for maximum entropy knockoffs (alg 2 in ref 2)
    * `:equi` for equi-distant knockoffs (eq 2.3 in ref 1), 
    * `:sdp` for SDP knockoffs (eq 2.4 in ref 1)
    * `:sdp_fast` for SDP knockoffs via coordiate descent (alg 2.2 in ref 3)
+ `windowsize`: Number of covariates to be included in a block. Each block consists of
    adjacent variables. The last block could contain less than `windowsize` variables. 
+ `kwargs...`: Possible optional inputs to solvers specified in `method`, see 
    [`solve_MVR`](@ref), [`solve_max_entropy`](@ref), and [`solve_sdp_fast`](@ref)

# Multithreading
To enable multiple threads, simply start Julia with >1 threads and this routine
will run with all available threads. 
"""
function approx_modelX_gaussian_knockoffs(
    X::AbstractMatrix{T}, 
    method::Symbol; 
    windowsize::Int = 500,
    kwargs...
    ) where T
    p = size(X, 2)
    windows = ceil(Int, p / windowsize)
    block_covariances = Vector{Matrix{T}}(undef, windows)
    block_s = Vector{Vector{T}}(undef, windows)
    # solve for s in each block of Σ
    Threads.@threads for window in 1:windows
        cur_range = window == windows ? 
            ((windows - 1)*windowsize + 1:p) : 
            ((window - 1)*windowsize + 1:window * windowsize)
        Xcur = @view(X[:, cur_range])
        # approximate a block of Σ by Ledoit-Wolf optimal shrinkage
        Σcur = cov(LinearShrinkage(DiagonalUnequalVariance(), :lw), Xcur)
        # solve for s vector
        scur = solve_s(Σcur, method; kwargs...)
        # save result
        block_covariances[window] = Σcur
        block_s[window] = scur
    end
    # assemble block Σ, s, and mean components
    Σ = BlockDiagonal(block_covariances)
    μ = vec(mean(X, dims=1))
    s = vcat(block_s...)
    # bisection search over γ ∈ [0, 1] to ensure diag(γs) ≤ 2Σ
    γ = fzero(γ -> f(γ, s, Σ), 0, 1)
    s .*= γ
    # generate knockoffs
    X̃ = condition(X, μ, inv(Σ), Diagonal(s))
    return GaussianKnockoff(X, X̃, s, Symmetric(Σ), method)
end

# for bisection search
function f(γ, s, Σ)
    D = Diagonal(γ .* s)
    λ = eigmin(2Σ - D) # can this be more efficient?
    return λ > 0 ? 1 - γ : -Inf
end
