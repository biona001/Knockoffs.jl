"""
    markov_knockoffs()

Generates knockoff of variables distributed as a discrete Markov Chain
with `K` states.

# Inputs
+ `X`: `n × p` matrix, each row is a sample
+ `Q`: `p - 1` vector of matrices. `Q[j]` is a `K × K` matrix of transition
    probabilities where Q[j][l, k] = P(X_{j+1} = k | X_{j} = l)
+ `q`: `K × 1` vector of initial probabilities

# Reference
Equations 4-5 of "Gene hunting with hidden Markov model knockoffs" by 
Sesia, Sabatti, and Candes
"""
function markov_knockoffs(
    X::AbstractMatrix{T},
    Q::AbstractMatrix{T},
    q::AbstractVector{T}
    ) where T <: AbstractFloat
    n, p = size(X)
    K = length(q)
    # check problem dimensions
    length(Q) == p - 1 || error("There should be $(p-1) transition matrices in Q, got $(length(Q) - 1)")
    for j in 1:p-1
        size(Q[j]) == (K, K) || error("size(Q[j]) = $(size(Q[j])) but there are $K states in q")
    end
    # allocate matrices
    X̃ = Matrix{T}(undef, n, p)
    N = Matrix{T}(undef, p, K)
    # start algorithm 1
    for j in 1:p
        # equation 5
        update_normalizing_constants!(N, @view(X[j-1, :]), @view(X̃[j-1, :]), Q, q)
        # sample knockoffs
        sample!(X̃, N)
    end
end

"""
    update_normalizing_constants!(Q::AbstractMatrix{T}, q::AbstractVector{T})

Computes normalizing constants recursively using equation (5).

# Inputs
+ `Q`: `p - 1` vector of matrix. `Q[j]` is a `K × K` matrix of transition
    probabilities where Q[j][l, k] = P(X_{j+1} = k | X_{j} = l)
+ `q`: `K × 1` vector of initial probabilities

# todo: efficiency
"""
function update_normalizing_constants!(
    N::AbstractMatrix{T},
    x::AbstractVector{T},
    x̃::AbstractVector{T},
    Q::AbstractMatrix{T},
    q::AbstractVector{T}
    ) where T <: AbstractFloat
    fill!(N, 0)
    for j in 1:p
        if j == 1
            mul!(@view(N[1, :]), q', Q[1])
        elseif j == p
            Qp = Q[end]
            for l in 1:K
                N[j, l] += Qp[l, x[p-1]] * Qp[l, x̃[p-1]] / N[p - 1, l]
            end
        else
            Qcurr, Qnext = Q[j], Q[j + 1]
            N[j, :] .= @view(Qnext[:, k]) # 3rd term of numerator
            for l in 1:K
                N[j, l] *= Qcurr[l, x[j-1]] * Qcurr[l, x̃[j-1]] / N[j - 1, l]
            end
        end
    end
    return nothing
end

function sample!(X̃, N)
    # todo
end
