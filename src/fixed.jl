"""
    fixed_knockoffs(X::Matrix{T}; method=:sdp)

Creates fixed knockoffs. 

# Inputs
+ `X`: A `n × p` numeric matrix, each row is a sample, and each column is normalized to mean 0 variance 1 with unit norm. 
+ `method`: :equi for equi-distant knockoffs (eq 2.3 in ref 1), :sdp for SDP 
    knockoffs (eq 2.4 in ref 1), :mvr for minimum variance-based
    reconstructability knockoffs (alg 1 in ref 2), or :maxent for maximum entropy
    knockoffs (alg 2 in ref 2)

# Output
+ `Knockoff`: A struct containing the original `X` and its knockoff `X̃`, in addition to other variables (e.g. `s`)

# Reference
1. "Controlling the false discovery rate via Knockoffs" by Barber and Candes (2015).
2. "Powerful knockoffs via minimizing reconstructability" by Spector, Asher, and Lucas Janson (2020)
"""
function fixed_knockoffs(X::Matrix{T}, method::Symbol) where T <: AbstractFloat
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
        s = solve_MVR(Σ, λmin=λmin)
    elseif method == :maxent
        s = solve_max_entropy(Σ, λmin=λmin)
    else
        error("fixed_knockoffs: method can only be :equi, :sdp, :mvr, or :maxent")
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
