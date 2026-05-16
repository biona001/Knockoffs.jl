"""
    fixed_knockoffs(X::Matrix{T}, method::Symbol; [center], [kwargs...])

Creates fixed-X knockoffs. Internally, `X` will be automatically normalized before
computing its knockoff. 

# Inputs
+ `X`: A column-normalized `n × p` numeric matrix, each row is a sample, and
    each column is covariate. We will internally normalized `X` if it is not. 

# Optional inputs
+ `method`: Can be one of the following
    * `:mvr`: Minimum variance-based reconstructability knockoffs (alg 1 in ref 2)
    * `:mvr_fast`: Experimental parallel MVR knockoffs
    * `:maxent`: Maximum entropy knockoffs (alg 2 in ref 2)
    * `:maxent_fast`: Experimental parallel maximum entropy knockoffs
    * `:equi`: Equi-distant knockoffs (eq 2.3 in ref 1), 
    * `:sdp`: SDP knockoffs via coordinate descent (alg 2.2 in ref 3)
    * `:sdp_fast`: Experimental parallel SDP coordinate descent knockoffs
+ `center`: Whether to center the columns of `X` before normalizing, defaults to `false`.
    When `center=true` and `n ≥ 2p + 1`, the knockoff columns are also constructed
    to be centered.
+ `kwargs...`: Possible optional inputs to `method`, see [`solve_MVR`](@ref), 
    [`solve_max_entropy`](@ref), [`solve_SDP`](@ref), and [`solve_sdp_parallel`](@ref)

# Output
+ `GaussianKnockoff`: A struct containing the original (column-normalized) `X`
    and its knockoff `X̃`, in addition to other variables (e.g. `s`)

# Reference
1. "Controlling the false discovery rate via Knockoffs" by Barber and Candes (2015).
2. "Powerful knockoffs via minimizing reconstructability" by Spector, Asher, and Lucas Janson (2020)
3. "FANOK: Knockoffs in Linear Time" by Askari et al. (2020).
"""
function fixed_knockoffs(X::Matrix{T}, method::Union{Symbol, AbstractString}; center::Bool=false, kwargs...) where T <: AbstractFloat
    n, p = size(X)
    n ≥ 2p || error("fixed_knockoffs: currently only works for n ≥ 2p case! sorry!")
    # use column-normalized X 
    X = normalize_col(X, center=center)
    # compute gram matrix using full svd
    U, σ, V = svd(X, full=true)
    Σ = V * Diagonal(σ)^2 * V'
    Σinv = V * inv(Diagonal(σ)^2) * V'
    # λmin = typemax(T)
    # for σi in σ
    #     σi^2 < λmin && (λmin = σi^2)
    # end
    # compute s vector using the specified method
    s = solve_s(Symmetric(Σ), method; kwargs...)
    # compute Ũ such that Ũ'X = 0. If X was centered and there is enough
    # dimension, also enforce Ũ'1 = 0 so the knockoffs are centered.
    if center && n ≥ 2p + 1
        intercept = fill(inv(sqrt(T(n))), n)
        U_centered = svd([X intercept], full=true).U
        Ũ = @view(U_centered[:, p+2:2p+1])
    else
        Ũ = @view(U[:, p+1:2p])
    end
    # compute C such that C'C = 2D - D*inv(Σ)*D via eigendecomposition (cholesky not stable)
    D = Diagonal(s)
    γ, P = eigen(2D - D*Σinv*D)
    clamp!(γ, 0, typemax(T)) # numerical stability
    C = Diagonal(sqrt.(γ)) * P'
    # compute knockoffs
    X̃ = X * (I - Σinv*D) + Ũ * C
    return GaussianKnockoff(X, X̃, s, Symmetric(Σ), Symbol(method), 1)
end
