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

# 1 state of a markov chain
struct GenotypeState
    a::Int # int between 1 and K
    b::Int # int between 1 and K
end

"""
Genotype states are index pairs (ka, kb) where ka, kb is unordered haplotype 1 and 2. 
If there are K=5 haplotype motifs, then the 15 possible genotype states and their index are

(1, 1) = 1
(1, 2) = 2     (2, 2) = 6
(1, 3) = 3     (2, 3) = 7     (3, 3) = 10
(1, 4) = 4     (2, 4) = 8     (3, 4) = 11     (4, 4) = 13
(1, 5) = 5     (2, 5) = 9     (3, 5) = 12     (4, 5) = 14     (5, 5) = 15
"""
struct MarkovChainTable
    K::Int # number of states
    index_to_pair::Vector{GenotypeState}
end
function MarkovChainTable(K::Int)
    table = GenotypeState[]
    for (a, b) in with_replacement_combinations(1:K, 2)
        push!(table, GenotypeState(a, b))
    end
    return MarkovChainTable(K, table)
end

statespace(mc::MarkovChainTable) = ((mc.K + 1) * mc.K) >> 1
function pair_to_index(mc::MarkovChainTable, a::Int, b::Int)
    statespace(mc) - ((mc.K-a+2)*(mc.K-a+1))>>1 + (b - a + 1)
end
function index_to_pair(mc::MarkovChainTable, i::Int)
    return mc.index_to_pair[i].a, mc.index_to_pair[i].b
end
# mc = MarkovChainTable(5)
# index_to_pair(mc, 10)
