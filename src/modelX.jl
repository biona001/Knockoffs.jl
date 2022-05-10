"""
    modelX_gaussian_knockoffs(X::Matrix, method::Symbol)

Creates model-free multivariate normal knockoffs by sequentially sampling from 
conditional multivariate normal distributions. The true mean `μ` and covariance
`Σ` is estimated from data. 

# Inputs
+ `X`: A `n × p` numeric matrix. Each row is a sample, and each column is standardized
to mean 0 variance 1. 
+ `method`: Either `:equi`, `:sdp`, or `:asdp`

# Reference: 
"Panning for Gold: Model-X Knockoffs for High-dimensional Controlled
Variable Selection" by Candes, Fan, Janson, and Lv (2018)

# Note: 
The covariance is approximated by the Ledoit-Wolf optimal shrinkage, which
is recommended for p>n case. We do not simply use `cov(X)` since `isposdef(cov(X))`
is typically false. For reference, see 
https://mateuszbaran.github.io/CovarianceEstimation.jl/dev/man/methods/
"""
function modelX_gaussian_knockoffs(X::Matrix, method::Symbol)
    Σapprox = cov(LinearShrinkage(DiagonalUnequalVariance(), :lw), X)
    # mean component is just column means
    μ = vec(mean(X, dims=1))
    return modelX_gaussian_knockoffs(X, method, μ, Σapprox)
end

"""
    modelX_gaussian_knockoffs(X::Matrix, method::Symbol, μ::Vector, Σ::Matrix)

Creates model-free multivariate normal knockoffs by sequentially sampling from 
conditional multivariate normal distributions. 

# Inputs
+ `X`: A `n × p` numeric matrix. Each row is a sample, and each column is standardized
to mean 0 variance 1. 
+ `method`: Either `:equi`, `:sdp`, or `:asdp`
+ `μ`: A `p × 1` vector of (true) mean of `X`
+ `Σ`: A `p × p` matrix of covariance of `X`

# Reference: 
"Panning for Gold: Model-X Knockoffs for High-dimensional Controlled
Variable Selection" by Candes, Fan, Janson, and Lv (2018)
"""
function modelX_gaussian_knockoffs(X::Matrix, method::Symbol, μ::AbstractVector, Σ::AbstractMatrix)
    n, p = size(X)
    # todo: convert covariance matrix to correlation matrix
    # compute s vector using the specified method
    if method == :equi
        λmin = minimum(svdvals(X))^2
        s = min(1, 2λmin) .* ones(p)
    elseif method == :sdp
        s = solve_SDP(Σ)
    elseif method==:asdp
        # todo
        error("ASDP not supported yet! sorry!")
    else
        error("modelX_gaussian: method can only be :equi, or :sdp, or :asdp")
    end
    X̃ = condition(X, μ, inv(Σ), Diagonal(s))
    return knockoff(X, X̃, s)
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
