"""
A `Knockoff` holds the original design matrix `X`, along with its knockoff `Xko`.
"""
abstract type Knockoff end

struct GaussianKnockoff{T<:AbstractFloat, M<:AbstractMatrix, S <: Symmetric} <: Knockoff
    X::M # n × p design matrix
    Xko::Matrix{T} # n × mp knockoff of X
    s::Vector{T} # p × 1 vector. Diagonal(s) and 2Sigma - Diagonal(s) are both psd
    Sigma::S # p × p symmetric covariance matrix. 
    method::Symbol # method for solving s
    m::Int # number of knockoffs per feature generated
end

struct ApproxGaussianKnockoff{T<:AbstractFloat, M<:AbstractMatrix, S<:Symmetric} <: Knockoff
    X::M # n × p design matrix
    Xko::Matrix{T} # n × mp knockoff of X
    s::Vector{T} # p × 1 vector. Diagonal(s) and 2Sigma - Diagonal(s) are both psd
    Sigma::S # p × p block-diagonal covariance matrix. 
    method::Symbol # method for solving s
    m::Int # number of knockoffs per feature generated
end

struct GaussianGroupKnockoff{T<:AbstractFloat, BD<:AbstractMatrix, S<:Symmetric} <: Knockoff
    X::Matrix{T} # n × p design matrix
    Xko::Matrix{T} # n × mp matrix storing knockoffs of X
    groups::Vector{Int} # p × 1 vector of group membership
    S::BD # p × p block-diagonal matrix of the same size as Sigma. S and (m+1)/m*Sigma - S are both psd
    gammas::Vector{T} # for suboptimal group construction only. These are scalars chosen so that S_i = γ_i * Sigma_i
    m::Int # number of knockoffs per feature generated
    Sigma::S # p × p symmetric covariance matrix. 
    method::Symbol # method for solving s
    obj::T # final objective value of group knockoff
end

struct GaussianRepGroupKnockoff{T<:AbstractFloat, S<:Symmetric} <: Knockoff
    X::Matrix{T} # n × p design matrix
    Xko::Matrix{T} # n × mp matrix storing knockoffs of X
    groups::Vector{Int} # p × 1 vector of group membership
    group_reps::Vector{Int} # vector of representative variables (i.e. columns of X)
    S11::Matrix{T} # r × r matrix from running group knockoffs optimization on the representatives. S11 and (m+1)/m*Sigma11 - S11 are both psd
    S::S # p × p matrix equal to [S11  S11*inv(Sigma11)*Sigma12; Sigma21*inv(Sigma11)*S11 Sigma21*inv(Sigma11)S11*inv(Sigma11)*Sigma12]
    m::Int # number of knockoffs per feature generated
    Sigma::S # p × p symmetric covariance matrix. 
    method::Symbol # method for solving s
    obj::T # final objective value of group knockoff
    enforce_cond_indep::Bool
end

struct IPADKnockoff{T<:AbstractFloat} <: Knockoff
    X::Matrix{T} # n × p design matrix
    Xko::Matrix{T} # n × mp knockoff of X
    m::Int # number of knockoffs per feature generated
    r::Int # rank of factor model
    r_method::Symbol # method for choosing r
end

struct MergedKnockoff{T} <: Knockoff
    XXko::Matrix{T} # n × (m+1)p matrix of original+knockoff features. The first m+1 variables are the first feature and its knockoffs shuffled...etc
    original::Vector{Int} # p × 1 vector of indices storing which columns of XXko contains the original features
    knockoff::Vector{Vector{Int}} # Length `p` vector each of length `m`. knockoff[i] store which columns of XXko contains knockoffs for ith feature
    p::Int # number of original features
    m::Int # number of knockoffs per feature
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

"""
A `KnockoffFilter` is essentially a `Knockoff` that has gone through a feature 
selection procedure, such as the Lasso. It stores, among other things, the final
estimated parameters `beta` after applying the knockoff-filter procedure.

The `debiased` variable is a boolean
indicating whether estimated effect size have been debiased with Lasso. The
`W` vector stores the feature importance statistic that satisfies the flip coin 
property. `tau` is the knockoff threshold, which controls the empirical FDR at 
level `q`
"""
abstract type KnockoffFilter end

struct LassoKnockoffFilter{T} <: KnockoffFilter
    y :: Vector{T} # n × 1 response vector
    X :: Matrix{T} # n × p matrix of original features
    ko :: Knockoff # A knockoff struct
    # merged_ko :: Knockoff # A MergedKnockoff struct
    m :: Int # number of knockoffs per feature generated
    betas :: Vector{Vector{T}} # betas[i] is the p × 1 vector of effect sizes corresponding to fdr level fdr_target[i]
    a0 :: Vector{T}   # intercepts for each model in betas
    selected :: Vector{Vector{Int}} # selected[i] includes all variables selected based on target FDR level fdr_target[i]
    W :: Vector{T} # length p vector of feature importance
    taus :: Vector{T} # threshold for significance. For fdr fdr_target[i], tau[i] is threshold, and all W ≥ tau[i] is selected
    fdr_target :: Vector{T} # target FDR level for each taus and betas
    d :: UnivariateDistribution # distribution of y
    debias :: Union{Nothing, Symbol} # how betas and a0 have been debiased (`nothing` for not debiased)
end

struct MarginalKnockoffFilter{T} <: KnockoffFilter
    y :: Vector{T} # n × 1 response vector
    X :: Matrix{T} # n × p matrix of original features
    ko :: Knockoff # A knockoff struct
    W :: Vector{T} # length p vector of feature importance
    taus :: Vector{T} # threshold for significance. For fdr fdr_target[i], tau[i] is threshold, and all W ≥ tau[i] is selected
    m :: Int # number of knockoffs per feature generated
    # T0 :: Vector{T} # marginal correlations y'*X
    # Tk :: Vector{Vector{T}} # marginal correlations [[y'*Xko1], [y'*Xko2], ..., [y'*Xkom]]
    selected :: Vector{Vector{Int}} # selected[i] includes all variables selected based on target FDR level fdr_target[i]
    fdr_target :: Vector{T} # target FDR level for each taus and betas
    d :: UnivariateDistribution # distribution of y
end
