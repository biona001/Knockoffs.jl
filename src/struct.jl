"""
A `Knockoff` is an `AbstractMatrix`, essentially the matrix [X X̃] of
concatenating X̃ to X. If `A` is a `Knockoff`, `A` behaves like a regular matrix
and can be inputted into any function that supports inputs of type `AbstractMatrix`.
Basic operations like @view(A[:, 1]) are supported. 
"""
struct Knockoff{T} <: AbstractMatrix{T}
    X::Matrix{T} # n × p design matrix
    X̃::Matrix{T} # n × p knockoff of X
    s::Vector{T} # p × 1 vector. Diagonal(s) and 2Σ - Diagonal(s) are both psd
    Σ::Matrix{T} # p × p gram matrix X'X
    Σinv::Matrix{T} # p × p inv(X'X)
end

function knockoff(X::AbstractMatrix{T}, X̃::AbstractMatrix{T}) where T
    Knockoff(X, X̃, T[], Matrix{T}(undef, 0, 0), Matrix{T}(undef, 0, 0))
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
