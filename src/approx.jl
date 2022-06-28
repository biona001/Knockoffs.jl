"""
    approx_modelX_gaussian_knockoffs(X, method; [windowsize = 500], [covariance_approximator], kwargs...)
    approx_modelX_gaussian_knockoffs(X, method, window_ranges; [covariance_approximator], kwargs...)

Generates Gaussian knockoffs by approximating the covariance as a block diagonal matrix. 
Each block contains `windowsize` consecutive features. 

# Inputs
+ `X`: A `n × p` numeric matrix or `SnpArray`. Each row is a sample, and each column is covariate.
+ `method`: Can be one of the following
    * `:mvr` for minimum variance-based reconstructability knockoffs (alg 1 in ref 2)
    * `:maxent` for maximum entropy knockoffs (alg 2 in ref 2)
    * `:equi` for equi-distant knockoffs (eq 2.3 in ref 1), 
    * `:sdp` for SDP knockoffs (eq 2.4 in ref 1)
    * `:sdp_fast` for SDP knockoffs via coordiate descent (alg 2.2 in ref 3)
+ `windowsize`: Number of covariates to be included in a block. Each block consists of
    adjacent variables. The last block could contain less than `windowsize` variables. 
+ `window_ranges`: Vector of ranges for each window. e.g. [1:97, 98:200, 201:500]
+ `covariance_approximator`: A covariance estimator, defaults to `LinearShrinkage(DiagonalUnequalVariance(), :lw)`.
    See CovarianceEstimation.jl for more options.
+ `kwargs...`: Possible optional inputs to solvers specified in `method`, see 
    [`solve_MVR`](@ref), [`solve_max_entropy`](@ref), and [`solve_sdp_fast`](@ref)

# Multithreading (todo)
To enable multiple threads, simply start Julia with >1 threads and this routine
will run with all available threads. 

# Covariance Approximation: 
The covariance is approximated by a `LinearShrinkageEstimator` using 
Ledoit-Wolf shrinkage with `DiagonalUnequalVariance` target, 
which seems to perform well for `p>n` cases. We do not simply use `cov(X)` since `isposdef(cov(X))`
is typically false. For comparison of different estimators, see:
https://mateuszbaran.github.io/CovarianceEstimation.jl/dev/man/msecomp/#msecomp
"""
function approx_modelX_gaussian_knockoffs(
    X::AbstractMatrix, 
    method::Symbol; 
    windowsize::Int = 500,
    covariance_approximator=LinearShrinkage(DiagonalUnequalVariance(), :lw),
    kwargs...
    )
    windowsize > 1 || error("windowsize should be > 1 but was $windowsize")
    p = size(X, 2)
    windows = ceil(Int, p / windowsize)
    window_ranges = UnitRange{Int64}[]
    # partition covariates into windows, each with `windowsize` covariates
    for window in 1:windows
        cur_range = window == windows ? 
            ((windows - 1)*windowsize + 1:p) : 
            ((window - 1)*windowsize + 1:window * windowsize)
        push!(window_ranges, cur_range)
    end
    return approx_modelX_gaussian_knockoffs(X, method, window_ranges; 
        covariance_approximator=covariance_approximator, kwargs...)
end

function approx_modelX_gaussian_knockoffs(
    X::AbstractMatrix, 
    method::Symbol,
    window_ranges::Vector{UnitRange{Int64}};
    covariance_approximator=LinearShrinkage(DiagonalUnequalVariance(), :lw),
    kwargs...
    )
    covariates_by_windows = sum(length.(window_ranges))
    covariates_by_windows == size(X, 2) || error("window_ranges have $covariates_by_windows dimensions but X has $(size(X, 2))")
    windows = length(window_ranges)
    block_covariances = Vector{Matrix{Float64}}(undef, windows)
    block_s = Vector{Vector{Float64}}(undef, windows)
    storage = typeof(X) <: AbstractSnpArray ? zeros(size(X, 1), windowsize) : nothing
    pmeter = Progress(windows, 1, "Approximating covariance by blocks...")
    # solve for s in each block of Σ
    for (window, cur_range) in enumerate(window_ranges)
        # grab current window of X
        Xcur = get_X_subset(X, cur_range, storage)
        # approximate a block of Σ
        Σcur = cov(covariance_approximator, Xcur)
        # solve for s vector
        scur = solve_s(Σcur, method; kwargs...)
        # save result
        block_covariances[window] = Σcur
        block_s[window] = scur
        next!(pmeter)
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
    return ApproxGaussianKnockoff(X, X̃, s, Symmetric(Σ), method)
end

# for bisection search
function f(γ, s, Σ)
    D = Diagonal(γ .* s)
    λ = eigmin(2Σ - D) # can this be more efficient?
    return λ > 0 ? 1 - γ : -Inf
end

function get_X_subset(X::AbstractMatrix, cur_range, storage)
    return @view(X[:, cur_range])
end

function get_X_subset(X::AbstractSnpArray, cur_range, storage)
    copyto!(storage, @view(X[:, cur_range]), impute=true, center=true, scale=true)
    return @view(storage[:, 1:length(cur_range)])
end
