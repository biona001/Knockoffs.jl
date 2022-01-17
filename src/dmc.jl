"""
    markov_knockoffs()

Generates knockoff of variables distributed as a discrete Markov Chain
with `K` states.

# Inputs
+ `Z`: Length `p` vector of `GenotypeState` where Z[i] = (ka, kb) is the 2
    haplotype motifs of state `i`
+ `Q`: `K × K × p - 1` array. `Q[:, :, j]` is a `K × K` matrix of transition
    probabilities for `j`th state, i.e. Q[l, k, j] = P(X_{j} = k | X_{j - 1} = l)
+ `q`: `K × 1` vector of initial probabilities

# Reference
Equations 4-5 of "Gene hunting with hidden Markov model knockoffs" by 
Sesia, Sabatti, and Candes
"""
function markov_knockoffs(
    Z::Vector{Int},
    Q::Array{T, 3},
    q::Vector{T}
    ) where T <: AbstractFloat
    p = length(Z)
    statespace = size(Q, 1)
    # preallocated arrays
    Z̃ = zeros(Int, p)
    N = zeros(p, statespace)
    d = Categorical([1 / statespace for _ in 1:statespace])
    return markov_knockoffs!(Z̃, Z, N, d, Q, q) # algorithm 1 in Sesia et al
end

function markov_knockoffs!(
    Z̃::Vector{Int},
    Z::Vector{Int},
    N::AbstractMatrix,
    d::Categorical, # Categorical distribution from Distributions.jl
    Q::Array{T, 3},
    q::Vector{T}
    ) where T <: AbstractFloat
    for j in 1:length(Z)
        update_normalizing_constants!(N, Z, Z̃, Q, q, j) # equation 5 in Sesia et al
        single_state_dmc_knockoff!(Z̃, Z, d, N, Q, q, j) # sample Z̃j
    end
    return Z̃
end

"""
    update_normalizing_constants!(Q::AbstractMatrix{T}, q::AbstractVector{T})

Computes normalizing constants recursively using equation (5).

# Inputs
+ `Q`: `K × K × p` array. `Q[:, :, j]` is a `K × K` matrix of transition
    probabilities for `j`th state, i.e. Q[l, k, j] = P(X_{j} = k | X_{j - 1} = l).
    The first transition matrix is not used. 
+ `q`: `K × 1` vector of initial probabilities

# todo: efficiency
"""
function update_normalizing_constants!(
    N::AbstractMatrix{T},
    Z::AbstractVector{Int},
    Z̃::AbstractVector{Int},
    Q::Array{T, 3},
    q::AbstractVector{T},
    j::Int
    ) where T <: AbstractFloat
    statespace, p = size(Q, 1), size(Q, 3)
    if j == 1
        mul!(@view(N[1, :]), Transpose(@view(Q[:, :, 2])), q)
    elseif j == p
        val = 0.0
        for l in 1:statespace
            val += Q[Z[p-1], l, p] * Q[Z̃[p-1], l, p] / N[p-1, l]
        end
        N[j, :] .= val
    else
        for k in 1:statespace, l in 1:statespace
            N[j, k] += Q[Z[j-1], l, j] * Q[Z̃[j-1], l, j] * Q[l, k, j + 1] / N[j - 1, l]
        end
    end
    return nothing
end

function single_state_dmc_knockoff!(
    Z̃::AbstractVector{Int},
    Z::AbstractVector{Int},
    d::Categorical, # Categorical distribution from Distributions.jl
    N::AbstractMatrix{T},
    Q::Array{T, 3},
    q::AbstractVector{T},
    j::Int
    ) where T
    statespace, p = size(Q, 1), size(Q, 3)
    if j == 1
        for z̃ in 1:statespace
            d.p[z̃] = q[z̃] * Q[z̃, Z[2], 2] / N[1, Z[2]]
        end
    elseif j == p
        for z̃ in 1:statespace
            d.p[z̃] = Q[Z[p-1], z̃, p] * Q[Z̃[p-1], z̃, p] / N[p-1, z̃] / N[p, 1]
        end
    else
        for z̃ in 1:statespace # todo: numerical error?
            d.p[z̃] = Q[Z[j - 1], z̃, j] * Q[Z̃[j-1], z̃, j] * Q[z̃, Z[j+1], j+1] / N[j-1, z̃] / N[j, Z[j+1]]
        end
    end
    Z̃[j] = rand(d)
    return nothing
end
