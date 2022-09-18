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
+ `m`: Number of knockoff copies per variable to generate, defaults to 1. 
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
    X::Matrix, 
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
    return GaussianKnockoff(X, X̃, s, Symmetric(Σ), method)
end

"""
    condition(x::AbstractVector, μ::AbstractVector, Σ::AbstractMatrix, S::AbstractMatrix, [m::Int=1])

Samples a knockoff x̃ from Gaussian x using conditional distribution formulas:

If (x, x̃) ~ N((μ, μ), G) where G = [Σ  Σ - S; Σ - S  Σ], then we sample x̃ from 
x̃|x = N(μ+(Σ-S)*inv(Σ)*(x-μ) , 2S-S*inv(Σ)*S). 
If we sample `m` knockoffs, G would have `m` copies of Σ-S in each row. 
We will sample the next knockoff from
x̃2 | (x, x̃) = N(μ + [Σ-S Σ-S]*inv([Σ Σ-S; Σ-S Σ])*([x;x̃]-[μ;μ]) , Σ-[Σ-S Σ-S]*inv([Σ Σ-S; Σ-S Σ])*[Σ-S;Σ-S])
and do so recursively until `m` knockoffs have been sampled. 

# Inputs
+ `X`: A `n × p` numeric matrix, each row is a sample, and each column is covariate.
+ `μ`: A `p × 1` vector of column mean of `X`
+ `Σ`: A `p × p` covariance matrix of `X`
+ `S`: A `p × p` matrix solved to satisfy `S ⪰ 0` and `2Σ - S ⪰ 0`

# Optional inputs
+ `m`: Number of (simultaneous) knockoffs per variable to generate, default `m=1`

# Output
+ `X̃`: A `n × pm` numeric matrix. The first `p` columns store the first knockoff copy,
    and the next `p` columns store the second knockoff...etc

# Todo
efficiency
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
    if m == 1
        Σinv = inv(Σ)
        ΣinvS = Σinv * S
        new_V = Symmetric(2S - S * ΣinvS)
        L = cholesky(PositiveFactorizations.Positive, new_V).L
        return X - (X .- μ') * ΣinvS + randn(n, p) * L
    end
    X̃ = Matrix{eltype(X)}(undef, n, p*m)
    for i in 1:m
        # partition the covariance into
        #  [Σ11 Σ12]
        #  [Σ21 Σ22]
        Σ11 = repeat(Σ - S, i, i)
        Σ11_diag = BlockDiagonal([S for _ in 1:i])
        Σ11 += Σ11_diag
        Σinv = inv(Σ11)
        # compute mean of the knockoff
        mean_diff = [X .- μ']
        for j in 1:i-1
            push!(mean_diff, X̃[:, (j - 1) * p + 1 : j * p] .- μ')
        end
        ko_mean = hcat(mean_diff...) * Σinv * repeat(Σ - S, i) .+ μ'
        # compute variance of knockoff
        Σ12 = repeat(Σ - S, i)
        ko_cov = Symmetric(Σ - Σ12' * Σinv * Σ12)
        L = cholesky(PositiveFactorizations.Positive, ko_cov).L
        # sample knockoffs
        X̃i = ko_mean + randn(n, p) * L
        X̃[:, (i - 1) * p + 1 : i * p] .= X̃i
    end
    return X̃
end
