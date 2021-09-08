struct Knockoff{T}
    X::Matrix{T} # original design matrix
    X̃::Matrix{T} # knockoff of X
    s::Vector{T} # diagonal(s) and 2Σ - diagonal(s) are both psd
    C::Matrix{T} # C'C = 2diagonal(s) - diagonal(s)*inv(Σ)*diagonal(s)
    Ũ::Matrix{T} # Ũ'X = 0
    Σ::Matrix{T} # X'X
    Σinv::Matrix{T} # inv(X'X)
end

function knockoff_equi(X::Matrix{T}) where T <: AbstractFloat
    n, p = size(X)
    # compute gram matrix using full svd
    U, σ, V = svd(X, full=true)
    Σ = V * Diagonal(σ)^2 * V'
    Σinv = V * inv(Diagonal(σ)^2) * V'
    # compute equi-correlated knockoffs
    λ = eigvals(Σ)
    s = min(1, 2minimum(λ)) .* ones(size(Σ, 1))
    # compute Ũ such that Ũ'X = 0
    Ũ = U[:, p+1:2p]
    # compute C such that C'C = 2D - D*inv(Σ)*D via eigendecomposition
    D = Diagonal(s)
    γ, P = eigen(2D - D*Σinv*D)
    clamp!(γ, 0, typemax(T))
    C = Diagonal(sqrt.(γ)) * P
    # compute knockoffs
    X̃ = X * (I - Σinv*D) + Ũ * C
    return Knockoff(X, X̃, s, C, Ũ, Σ, Σinv)
end

function knockoff_sdp(X::Matrix{T}) where T <: AbstractFloat
    # TODO
end

function normalize_col!(X::AbstractMatrix)
    n, p = size(X)
    @inbounds for x in eachcol(X)
        μi = mean(x)
        xnorm = norm(x)
        @simd for i in eachindex(x)
            x[i] = (x[i] - μi) / xnorm
        end
    end
    return X
end
