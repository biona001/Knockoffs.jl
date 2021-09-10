"""
A `Knockoff` is an `AbstractMatrix`, essentially the matrix [X X̃] of
concatenating X̃ to X. It behaves like a regular matrix and can be inputted
into any function that supports inputs of type `AbstractMatrix`. Basic
operations like @view(A[:, 1]) are supported. 
"""
struct Knockoff{T} <: AbstractMatrix{T}
    X::Matrix{T} # original design matrix
    X̃::Matrix{T} # knockoff of X
    s::Vector{T} # diagonal(s) and 2Σ - diagonal(s) are both psd
    C::Matrix{T} # C'C = 2diagonal(s) - diagonal(s)*inv(Σ)*diagonal(s)
    Ũ::Matrix{T} # Ũ'X = 0
    Σ::Matrix{T} # X'X
    Σinv::Matrix{T} # inv(X'X)
end

Base.size(A::Knockoff) = size(A.X, 1), 2size(A.X, 2)
Base.eltype(A::Knockoff) = eltype(A.X)
function Base.getindex(A::Knockoff, i::Int)
    n, p = size(A.X)
    i ≤ n * p ? getindex(A.X, i) : getindex(A.X̃, i - n * p)
end
function Base.getindex(A::Knockoff, i::Int, j::Int)
    n, p = size(A.X)
    j ≤ p ? getindex(A.X, i, j) : getindex(A.X̃, i, j - p)
end
function LinearAlgebra.mul!(C::AbstractMatrix, A::Knockoff, B::AbstractMatrix)
    p = size(A.X, 2)
    mul!(C, A.X, @view(B[1:p, :]))
    mul!(C, A.X̃, @view(B[p+1:end, :]), 1.0, 1.0)
end
function LinearAlgebra.mul!(c::AbstractVector, A::Knockoff, b::AbstractVector)
    p = size(A.X, 2)
    mul!(c, A.X, @view(b[1:p]))
    mul!(c, A.X̃, @view(b[p+1:end]), 1.0, 1.0)
end

function knockoff_equi(X::Matrix{T}) where T <: AbstractFloat
    n, p = size(X)
    n ≥ 2p || error("knockoff_equi: currently only works for n ≥ 2p case! sorry!")
    # compute gram matrix using full svd
    U, σ, V = svd(X, full=true)
    Σ = V * Diagonal(σ)^2 * V'
    Σinv = V * inv(Diagonal(σ)^2) * V'
    # compute equi-correlated knockoffs
    λmin = typemax(T)
    for σi in σ
        σi^2 < λmin && (λmin = σi^2)
    end
    s = min(1, 2λmin) .* ones(size(Σ, 1))
    # compute Ũ such that Ũ'X = 0
    Ũ = U[:, p+1:2p]
    # compute C such that C'C = 2D - D*inv(Σ)*D via eigendecomposition (cholesky not stable)
    D = Diagonal(s)
    γ, P = eigen(2D - D*Σinv*D)
    clamp!(γ, 0, typemax(T)) # numerical stability
    C = Diagonal(sqrt.(γ)) * P
    # compute knockoffs
    X̃ = X * (I - Σinv*D) + Ũ * C
    return Knockoff(X, X̃, s, C, Ũ, Σ, Σinv)
end

function knockoff_sdp(X::Matrix{T}) where T <: AbstractFloat
    n, p = size(X)
    n ≥ 2p || error("knockoff_sdp: currently only works for n ≥ 2p case! sorry!")
    # compute gram matrix using full svd
    U, σ, V = svd(X, full=true)
    Σ = V * Diagonal(σ)^2 * V'
    Σinv = V * inv(Diagonal(σ)^2) * V'
    # setup and solve SDP problem to get s
    s = Variable(p)
    problem = maximize(sum(s), s ≥ 0, 1 ≥ s, 2Σ - Diagonal(s) == Semidefinite(p))
    solve!(problem, () -> SCS.Optimizer(verbose=false))
    sfinal = clamp.(vec(s.value), 0, 1)
    # compute Ũ such that Ũ'X = 0
    Ũ = U[:, p+1:2p]
    # compute C such that C'C = 2D - D*inv(Σ)*D via eigendecomposition (cholesky not stable)
    D = Diagonal(sfinal)
    γ, P = eigen(2D - D*Σinv*D)
    clamp!(γ, 0, typemax(T)) # numerical stability
    C = Diagonal(sqrt.(γ)) * P
    # compute knockoffs
    X̃ = X * (I - Σinv*D) + Ũ * C
    return Knockoff(X, X̃, sfinal, C, Ũ, Σ, Σinv)
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
