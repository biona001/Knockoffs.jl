"""
    modelX_gaussian_knockoffs(X::Matrix, method::Symbol; [m], [covariance_approximator], [kwargs...])
    modelX_gaussian_knockoffs(X::Matrix, method::Symbol, μ::Vector, Σ::Matrix; [m], [kwargs...])

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
+ `μ`: A `p × 1` vector of column mean of `X`, defaults to column mean
+ `Σ`: A `p × p` matrix of covariance of `X`, defaults to a shrinkage estimator
    specified by `covariance_approximator`. 
+ `m`: Number of knockoff copies per variable to generate, defaults to 1. 
+ `covariance_approximator`: A covariance estimator, defaults to `LinearShrinkage(DiagonalUnequalVariance(), :lw)`
    which tends to give good empirical performance when p>n. See CovarianceEstimation.jl for more options.
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
    X::AbstractMatrix, 
    method::Symbol;
    m::Int = 1,
    covariance_approximator=LinearShrinkage(DiagonalUnequalVariance(), :lw),
    kwargs...
    )
    # approximate Σ
    Σapprox = cov(covariance_approximator, X)
    # mean component is just column means
    μ = vec(mean(X, dims=1))
    return modelX_gaussian_knockoffs(X, method, μ, Σapprox; m=m, kwargs...)
end

function modelX_gaussian_knockoffs(
    X::AbstractMatrix, 
    method::Symbol, 
    μ::AbstractVector, 
    Σ::AbstractMatrix; 
    m::Int = 1,
    kwargs...
    )
    # compute s vector using the specified method
    s = solve_s(Symmetric(Σ), method; m=m, kwargs...)
    # generate knockoffs
    X̃ = condition(X, μ, Symmetric(Σ), Diagonal(s); m=m)
    return GaussianKnockoff(X, X̃, s, Symmetric(Σ), method, m)
end

"""
    condition(x::AbstractVector, μ::AbstractVector, Σ::AbstractMatrix, S::AbstractMatrix, [m::Int=1])

Samples a knockoff x̃ from Gaussian x using conditional distribution formulas:

If (x, x̃) ~ N((μ, μ), G) where G = [Σ  Σ - S; Σ - S  Σ], then we sample x̃ from 
x̃|x = N(μ+(Σ-S)*inv(Σ)*(x-μ) , 2S-S*inv(Σ)*S). 

If we sample `m` knockoffs, we use the algorithm in 
"Improving the Stability of the Knockoff Procedure: Multiple Simultaneous Knockoffs 
and Entropy Maximization" by Gimenez and Zou.

# Inputs
+ `X`: A `n × p` numeric matrix, each row is a sample, and each column is covariate.
+ `μ`: A `p × 1` vector of column mean of `X`
+ `Σ`: A `p × p` covariance matrix of `X`
+ `S`: A `p × p` matrix solved to satisfy `S ⪰ 0` and `(m+1)/m*Σ - S ⪰ 0`
+ `m`: Number of (simultaneous) knockoffs per variable to generate, default `m=1`

# Output
+ `X̃`: A `n × pm` numeric matrix. The first `p` columns store the first knockoff copy,
    and the next `p` columns store the second knockoff...etc

# Todo
+ When s is the zero vector, X̃ should be identical to X but it isn't
+ Consider changing sampling code to using Distribution's MvNormal
+ For multiple knockoffs, can we avoid storing a pm × pm matrix in memory?
"""
function condition(
    X::AbstractMatrix, 
    μ::AbstractVector, 
    Σ::AbstractMatrix, 
    S::AbstractMatrix;
    m::Int = 1
    )
    n, p = size(X)
    m < 1 && error("m should be 1 or larger but was $m.")
    Σinv = inv(Symmetric(Σ))
    ΣinvS = Σinv * S
    C = 2S - S*ΣinvS
    if m == 1
        Σ̃ = Symmetric(C)
        L = cholesky(PositiveFactorizations.Positive, Σ̃).L
        return X - (X .- μ') * ΣinvS + randn(n, p) * L
    end
    # todo: can we form Σ̃ using SymmetricToeplitz? 
    # So we don't need to actually store a matrix of size pm × pm in memory?
    Σ̃ = repeat(Σ - C, m, m)
    Σ̃ += BlockDiagonal([S for _ in 1:m]) # note S is variable D in Gimenez and Zou
    μi = X - (X .- μ') * ΣinvS # in Gaminez and Zou, μi = Dinv(Σ)μ-(I-Dinv(Σ))X = (algebra..) = X-(X.-μ')*ΣinvS
    μfull = repeat(μi, 1, m)
    L = cholesky(PositiveFactorizations.Positive, Symmetric(Σ̃)).L
    return μfull + randn(n, m*p) * L
end
