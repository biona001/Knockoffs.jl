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
    * `:mvr_fast` for experimental parallel MVR knockoffs
    * `:maxent` for maximum entropy knockoffs (alg 2 in ref 2)
    * `:maxent_fast` for experimental parallel maximum entropy knockoffs
    * `:equi` for equi-distant knockoffs (eq 2.3 in ref 1), 
    * `:sdp` for SDP knockoffs via coordinate descent (alg 2.2 in ref 3)
    * `:sdp_fast` for experimental parallel SDP coordinate descent knockoffs
+ `μ`: A `p × 1` vector of column mean of `X`, defaults to column mean
+ `Σ`: A `p × p` matrix of covariance of `X`, defaults to a shrinkage estimator
    specified by `covariance_approximator`. 
+ `m`: Number of knockoff copies per variable to generate, defaults to 1. 
+ `covariance_approximator`: A covariance estimator, defaults to `LinearShrinkage(DiagonalUnequalVariance(), :lw)`
    which tends to give good empirical performance when p>n. See CovarianceEstimation.jl for more options.
+ `kwargs...`: Possible optional inputs to solvers specified in `method`, see 
    [`solve_MVR`](@ref), [`solve_max_entropy`](@ref), [`solve_SDP`](@ref), and
    [`solve_sdp_fast`](@ref)

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
    method::Union{Symbol,String};
    m::Number = 1,
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
    method::Union{Symbol,String}, 
    μ::AbstractVector, 
    Σ::AbstractMatrix; 
    m::Number = 1,
    kwargs...
    )
    # compute s vector using the specified method
    s = solve_s(Symmetric(Σ), Symbol(method); m=m, kwargs...)
    # generate knockoffs
    X̃ = condition(X, μ, Symmetric(Σ), Diagonal(s); m=m)
    return GaussianKnockoff(X, X̃, s, Symmetric(Σ), Symbol(method), Int(m))
end

"""
    condition(x::AbstractVector, μ::AbstractVector, Σ::AbstractMatrix, S::AbstractMatrix, [m::Number=1])

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
"""
function condition(
    X::AbstractMatrix, 
    μ::AbstractVector, 
    Σ::AbstractMatrix, 
    S::AbstractMatrix;
    m::Number = 1
    )
    n, p = size(X)
    m = Int(m)
    m < 1 && error("m should be 1 or larger but was $m.")
    Σinv = inv(Symmetric(Σ))
    ΣinvS = Σinv * S
    C = 2S - S*ΣinvS
    if m == 1
        Σ̃ = Symmetric(C)
        L = cholesky(PositiveFactorizations.Positive, Σ̃).L
        return X - (X .- μ') * ΣinvS + randn(n, p) * L
    end
    # Efficient sampling using Jiaqi's trick: avoids forming an mp × mp covariance matrix.
    # Each knockoff j is decomposed as: μi + shared_component + (ind_j - ind_avg),
    # where shared_component ~ N(0, C - (m-1)/m * S) and ind_j ~ N(0, S) are i.i.d.
    # This gives the correct block covariance: diag blocks = C, off-diag blocks = C - S.
    # Note: S is variable D in Gimenez and Zou.
    L_shared = cholesky(PositiveFactorizations.Positive, Symmetric(C - (m-1)/m * S)).L
    L_ind = cholesky(PositiveFactorizations.Positive, Symmetric(S)).L
    μi = X - (X .- μ') * ΣinvS # in Gimenez and Zou, μi = Dinv(Σ)μ-(I-Dinv(Σ))X = (algebra..) = X-(X.-μ')*ΣinvS
    E1 = randn(n, p) * L_shared'  # n×p shared component, rows ~ N(0, C - (m-1)/m * S)
    E2 = [randn(n, p) * L_ind' for _ in 1:m]  # each n×p, rows ~ N(0, S) i.i.d.
    E2_avg = sum(E2) / m
    X̃ = similar(X, n, m*p)
    for j in 1:m
        X̃[:, (j-1)*p+1 : j*p] .= μi .+ E1 .+ E2[j] .- E2_avg
    end
    return X̃
end
