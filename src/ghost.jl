"""
    ghost_knockoffs(zscores, Σ, s)
"""
function ghost_knockoffs(zscores, Σ, s)
    # assemble needed variables
    Σinv = inv(Σ)
    D = Diagonal(s)
    DΣinv = D * Σinv
    P = I - DΣinv
    # generate ghost knockoffs
    V = 2D - DΣinv * D
    μ = P * zscores
    Z̃ = rand(MvNormal(μ, V))
end

function match_Z_to_H(Z_pos::AbstractVector{Int}, H_pos::AbstractVector{Int})
    issorted(Z_pos) || error("Z_pos not sorted!")
    issorted(H_pos) || error("H_pos not sorted!")
    # find all Zj that can be matched to H
    matched_idx = indexin(Z_pos, H_pos)
    # for Zj that can't be mathced, find a SNP in H that is closest to Zj
    for i in eachindex(matched_idx)
        if isnothing(matched_idx[i])
            matched_idx[i] = searchsortednearest(H_pos, Z_pos[i])
        end
    end
    return Vector{Int}(matched_idx)
end

# reference is assumed sorted
# adapted from https://discourse.julialang.org/t/findnearest-function/4143/5
function searchsortednearest(reference::Vector{Int}, x::Int)
    idx = searchsortedfirst(reference, x)
    idx == 1 && return idx
    idx > length(reference) && return length(reference)
    reference[idx]==x && return idx
    return abs(reference[idx]-x) < abs(reference[idx-1]-x) ? idx : idx - 1
end
# reference = [1, 5, 12, 23]
# searchsortednearest(reference, 4)
