"""
    fixed_knockoffs(X::Matrix{T}; method=:sdp)

Creates fixed-X knockoffs. 

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

# Output
+ `Knockoff`: A struct containing the original (column-normalized) `X` and its knockoff `X̃`, 
    in addition to other variables (e.g. `s`)

# Reference
1. "Controlling the false discovery rate via Knockoffs" by Barber and Candes (2015).
2. "Powerful knockoffs via minimizing reconstructability" by Spector, Asher, and Lucas Janson (2020)
3. "FANOK: Knockoffs in Linear Time" by Askari et al. (2020).
"""
function fixed_knockoffs(X::Matrix{T}, method::Symbol; kwargs...) where T <: AbstractFloat
    n, p = size(X)
    n ≥ 2p || error("fixed_knockoffs: currently only works for n ≥ 2p case! sorry!")
    # use column-normalized X 
    X = normalize_col(X)
    # compute gram matrix using full svd
    U, σ, V = svd(X, full=true)
    Σ = V * Diagonal(σ)^2 * V'
    Σinv = V * inv(Diagonal(σ)^2) * V'
    λmin = typemax(T)
    for σi in σ
        σi^2 < λmin && (λmin = σi^2)
    end
    # compute s vector using the specified method
    if method == :equi
        s = min(1, 2λmin) .* ones(size(Σ, 1))
    elseif method == :sdp
        s = solve_SDP(Σ)
    elseif method == :mvr
        s = solve_MVR(Σ, λmin=λmin; kwargs...)
    elseif method == :maxent
        s = solve_max_entropy(Σ, λmin=λmin; kwargs...)
    elseif method == :sdp_fast
        s = solve_sdp_fast(Σ; kwargs...)
    else
        error("Method can only be :equi, :sdp, :mvr, :maxent, or :sdp_fast but was $method")
    end
    # compute Ũ such that Ũ'X = 0
    Ũ = @view(U[:, p+1:2p])
    # compute C such that C'C = 2D - D*inv(Σ)*D via eigendecomposition (cholesky not stable)
    D = Diagonal(s)
    γ, P = eigen(2D - D*Σinv*D)
    clamp!(γ, 0, typemax(T)) # numerical stability
    C = Diagonal(sqrt.(γ)) * P
    # compute knockoffs
    X̃ = X * (I - Σinv*D) + Ũ * C
    return GaussianKnockoff(X, X̃, s, Symmetric(Σ), method)
end
