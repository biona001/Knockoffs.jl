"""
    modelX_gaussian_knockoffs(X::Matrix, method::Symbol; [covariance_approximator], [kwargs...])
    modelX_gaussian_knockoffs(X::Matrix, method::Symbol, μ::Vector, Σ::Matrix; [kwargs...])

Creates model-free multivariate normal knockoffs by sequentially sampling from 
conditional multivariate normal distributions. The true mean `μ` and covariance
`Σ` is estimated from data if not supplied. 

# Inputs
+ `X`: A `n × p` numeric matrix, each row is a sample, and each column is covariate.
+ `method`: Can be one of the following
    * `:mvr` for minimum variance-based reconstructability knockoffs (alg 1 in ref 2)
    * `:maxent` for maximum entropy knockoffs (alg 2 in ref 2)
    * `:equi` for equi-distant knockoffs (eq 2.3 in ref 1), 
    * `:sdp` for SDP knockoffs (eq 2.4 in ref 1)
    * `:sdp_fast` for SDP knockoffs via coordiate descent (alg 2.2 in ref 3)
+ `μ`: A `p × 1` vector of column mean of `X`
+ `Σ`: A `p × p` matrix of covariance of `X`
+ `covariance_approximator`: A covariance estimator, defaults to `LinearShrinkage(DiagonalUnequalVariance(), :lw)`.
    See CovarianceEstimation.jl for more options.
+ `kwargs...`: Possible optional inputs to solvers specified in `method`, see 
    [`solve_MVR`](@ref), [`solve_max_entropy`](@ref), and [`solve_sdp_fast`](@ref)

# Reference: 
1. "Panning for Gold: Model-X Knockoffs for High-dimensional Controlled
    Variable Selection" by Candes, Fan, Janson, and Lv (2018)
2. "Powerful knockoffs via minimizing reconstructability" by Spector, Asher, and Lucas Janson (2020)
3. "FANOK: Knockoffs in Linear Time" by Askari et al. (2020).

# Covariance Approximation: 
The covariance is approximated by a linear shrinkage estimator using 
Ledoit-Wolf with `DiagonalUnequalVariance` target, 
which seems to perform well for `p>n` cases. We do not simply use `cov(X)`
since `isposdef(cov(X))` is typically false. For comparison of various estimators, see:
https://mateuszbaran.github.io/CovarianceEstimation.jl/dev/man/msecomp/#msecomp
"""
function modelX_gaussian_knockoffs(
    X::Matrix, 
    method::Symbol;
    covariance_approximator=LinearShrinkage(DiagonalUnequalVariance(), :lw),
    kwargs...
    )
    # approximate Σ
    Σapprox = cov(covariance_approximator, X)
    # mean component is just column means
    μ = vec(mean(X, dims=1))
    return modelX_gaussian_knockoffs(X, method, μ, Σapprox; kwargs...)
end

function modelX_gaussian_knockoffs(X::Matrix, method::Symbol, μ::AbstractVector, Σ::AbstractMatrix; kwargs...)
    # compute s vector using the specified method
    s = solve_s(Symmetric(Σ), method; kwargs...)
    # generate knockoffs
    X̃ = condition(X, μ, inv(Σ), Diagonal(s))
    return GaussianKnockoff(X, X̃, s, Symmetric(Σ), method)
end

"""
    condition(x::AbstractVector, μ::AbstractVector, Σinv::AbstractMatrix, D::AbstractMatrix)

Samples a knockoff x̃ from Gaussian x using conditional distribution formulas:

If (x, x̃) ~ N((μ, μ), G) where G = [Σ  Σ - D; Σ - D  Σ], then we sample x̃ from 
x̃|x = N(x - D*inv(Σ)(x - μ), 2D - D*inv(Σ)*D)

# todo: efficiency
"""
function condition(X::AbstractMatrix, μ::AbstractVector, Σinv::AbstractMatrix, D::AbstractMatrix)
    n, p = size(X)
    ΣinvD = Σinv * D
    new_V = Symmetric(2D - D * ΣinvD)
    L = cholesky(PositiveFactorizations.Positive, new_V).L
    return X - (X .- μ') * ΣinvD + randn(n, p) * L
end
