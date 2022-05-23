"""
A `Knockoff` holds the original design matrix `X`, along with its knockoff `X̃`.
"""
abstract type Knockoff end

struct GaussianKnockoff{T} <: Knockoff
    X::Matrix{T} # n × p design matrix
    X̃::Matrix{T} # n × p knockoff of X
    s::Vector{T} # p × 1 vector. Diagonal(s) and 2Σ - Diagonal(s) are both psd
    Σ::Symmetric{T, Matrix{T}} # p × p covariance matrix
    method::Symbol # :sdp or :equi
end

function gaussian_knockoff(X::AbstractMatrix{T}, X̃::AbstractMatrix{T}, method::Symbol) where T
    GaussianKnockoff(X, X̃, T[], Matrix{T}(undef, 0, 0), method)
end

function gaussian_knockoff(X::AbstractMatrix{T}, X̃::AbstractMatrix{T}, s::AbstractVector{T}, method::Symbol) where T
    GaussianKnockoff(X, X̃, s, Matrix{T}(undef, 0, 0), method)
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
    index_to_pair::Vector{GenotypeState} # index_to_pair[1] = (1, 1), index_to_pair[2] = (1, 2)...etc
end
function MarkovChainTable(K::Int)
    table = GenotypeState[]
    for a in 1:K, b in a:K
        push!(table, GenotypeState(a, b))
    end
    return MarkovChainTable(K, table)
end

Base.enumerate(mc::MarkovChainTable) = enumerate(mc.index_to_pair)
statespace(mc::MarkovChainTable) = ((mc.K + 1) * mc.K) >> 1
function pair_to_index(mc::MarkovChainTable, a::Int, b::Int)
    statespace(mc) - ((mc.K-a+2)*(mc.K-a+1))>>1 + (b - a + 1)
end
function index_to_pair(mc::MarkovChainTable, i::Int)
    return mc.index_to_pair[i].a, mc.index_to_pair[i].b
end
# mc = MarkovChainTable(5)
# index_to_pair(mc, 10)

# """
# A `ImportanceStatistic` holds the knockoff importance score `W`s and the computed 
# threshold `τ`. In the knockoff-filter methodology, selected predictors are 
# variable `j` such that `W[j] ≥ τ`, which controls the empirical FDR at level `q`
# """
# struct ImportanceStatistic{T}
#     W :: Vector{T}
#     τ :: T
#     q :: T
# end

# """
# Each `Filter{T}` is the final β vector after applying the knockoff-filter to [XX̃],
# controlling FDR at level `fdr`. The `debiased` variable indicates whether the 
# effect size in `β` have been debiased using a secondary Lasso routine
# """
# struct Filter{T}
#     β :: Vector{T}
#     fdr :: T
#     τ :: T
#     debiased :: Bool
# end

"""
A `KnockoffFilter` is essentially a `Knockoff` that has gone through a feature 
selection procedure, such as the Lasso. It stores, among other things, the final
estimated parameters `β` after applying the knockoff-filter procedure.

The `debiased` variable is a boolean
indicating whether estimated effect size have been debiased with Lasso. The
`W` vector stores the feature importance statistic that satisfies the flip coin 
property. `τ` is the knockoff threshold, which controls the empirical FDR at 
level `q`
"""
struct KnockoffFilter{T}
    XX̃ :: Matrix{T}
    original :: Vector{Int}
    knockoff :: Vector{Int}
    W :: Vector{Vector{T}}
    βs :: Vector{Vector{T}}
    a0 :: Vector{T}   # intercepts for each model in βs
    τs :: Vector{T}   # knockoff threshold for selecting Ws correponding to each FDR
    fdr_target :: Vector{T} # target FDR level for each τs and βs
    debiased :: Bool # whether βs and a0 have been debiased
end
