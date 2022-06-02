"""
    modelX_gaussian_knockoffs(X::Matrix, method::Symbol)

Creates model-free multivariate normal knockoffs by sequentially sampling from 
conditional multivariate normal distributions. The true mean `μ` and covariance
`Σ` is estimated from data. 

# Inputs
+ `X`: A `n × p` numeric matrix, each row is a sample, and each column is covariate.
+ `method`: Can be one of the following
    * `:mvr`: Minimum variance-based reconstructability knockoffs (alg 1 in ref 2)
    * `:maxent`: Maximum entropy knockoffs (alg 2 in ref 2)
    * `:equi`: Equi-distant knockoffs (eq 2.3 in ref 1), 
    * `:sdp`: SDP knockoffs (eq 2.4 in ref 1)
    * `:sdp_fast`: SDP knockoffs via coordiate descent (alg 2.2 in ref 3)
+ `kwargs...`: Possible optional inputs to `method`, see [`solve_MVR`](@ref), 
    [`solve_max_entropy`](@ref), and [`solve_sdp_fast`](@ref)

# Reference: 
1. "Panning for Gold: Model-X Knockoffs for High-dimensional Controlled
    Variable Selection" by Candes, Fan, Janson, and Lv (2018)
2. "Powerful knockoffs via minimizing reconstructability" by Spector, Asher, and Lucas Janson (2020)
3. "FANOK: Knockoffs in Linear Time" by Askari et al. (2020).

# Note: 
The covariance is approximated by the Ledoit-Wolf optimal shrinkage, which
is recommended for p>n case. We do not simply use `cov(X)` since `isposdef(cov(X))`
is typically false. For reference, see 
https://mateuszbaran.github.io/CovarianceEstimation.jl/dev/man/methods/
"""
function modelX_gaussian_knockoffs(X::Matrix, method::Symbol; kwargs...)
    # approximate Σ by Ledoit-Wolf optimal shrinkage
    Σapprox = cov(LinearShrinkage(DiagonalUnequalVariance(), :lw), X)
    # mean component is just column means
    μ = vec(mean(X, dims=1))
    return modelX_gaussian_knockoffs(X, method, μ, Σapprox; kwargs...)
end

"""
    modelX_gaussian_knockoffs(X::Matrix, method::Symbol, μ::Vector, Σ::Matrix)

Creates model-free multivariate normal knockoffs by sequentially sampling from 
conditional multivariate normal distributions. 

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
+ `kwargs...`: Possible optional inputs to `method`, see [`solve_MVR`](@ref), 
    [`solve_max_entropy`](@ref), and [`solve_sdp_fast`](@ref)

# Reference: 
1. "Panning for Gold: Model-X Knockoffs for High-dimensional Controlled
    Variable Selection" by Candes, Fan, Janson, and Lv (2018)
2. "Powerful knockoffs via minimizing reconstructability" by Spector, Asher, and Lucas Janson (2020)
3. "FANOK: Knockoffs in Linear Time" by Askari et al. (2020).
"""
function modelX_gaussian_knockoffs(X::Matrix, method::Symbol, μ::AbstractVector, Σ::AbstractMatrix; kwargs...)
    n, p = size(X)
    # create correlation matrix
    σs = sqrt.(diag(Σ))
    Σcor = StatsBase.cov2cor!(Matrix(Σ), σs)
    # compute s vector using the specified method
    if method == :equi
        s = min(1, 2*eigmin(Σcor)) .* ones(p)
    elseif method == :sdp
        s = solve_SDP(Σcor)
    elseif method == :mvr
        s = solve_MVR(Σcor; kwargs...)
    elseif method == :maxent
        s = solve_max_entropy(Σcor; kwargs...)
    elseif method == :sdp_fast
        s = solve_sdp_fast(Σcor; kwargs...)
    else
        error("Method can only be :equi, :sdp, :mvr, :maxent, or :sdp_fast but was $method")
    end
    s .*= σs.^2 # rescale s back to the result for a covariance matrix
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
