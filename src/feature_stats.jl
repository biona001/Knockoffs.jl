# function extract_beta(β̂_knockoff::AbstractVector{T}, fdrs::::AbstractVector, 
#     original::AbstractVector{Int}, knockoff::AbstractVector{Int},
#     method::Symbol=:knockoff, debias::Bool = false) where T <: AbstractFloat

# end

"""
    coefficient_diff(β::AbstractVector, original::AbstractVector{Int}, knockoff::AbstractVector{Int})

Returns the coefficient difference statistic W[j] = |β[j]| - |β[j + p]| 
from a univariate (single response) regression, where the `j`th variable is stored
in position `original[j]` of `β`, and its knockoff is stored in position `knockoff[j]`

# Inputs
+ `β`: Vector of regression coefficients
+ `original`: The index of original variables in `β`
+ `knockoff`: The index of knockoff variables in `β`
"""
function coefficient_diff(β::AbstractVector, original::AbstractVector{Int}, knockoff::AbstractVector{Int})
    return abs.(β[original]) - abs.(β[knockoff])
end

"""
    coefficient_diff(β::AbstractVector, groups::Vector{Int}, original::Vector{Int}, knockoff::Vector{Int})

Returns the coefficient difference statistic for grouped variables
W[G] = sum_{j in G} |β[j]| - sum_{j in G} |β[j + p]|.

# Inputs
+ `β`: `2p × 1` vector of regression coefficients, including original and knockoff effect sizes
+ `groups`: Vector storing group membership. `groups[i]` is the group of `β[i]`
+ `original`: The index of original variables in `β`
+ `knockoff`: The index of knockoff variables in `β`
"""
function coefficient_diff(β::AbstractVector, groups::AbstractVector{Int},
    original::AbstractVector{Int}, knockoff::AbstractVector{Int})
    length(β) == length(groups) || error("coefficient_diff: length(β) does not equal length(groups)")
    unique_groups = unique(groups)
    β_groups = zeros(length(unique_groups))
    # find which variables are Knockoffs
    knockoff_idx = falses(length(β))
    knockoff_idx[knockoff] .= true
    # loop over each variable
    for i in 1:length(β)
        idx = findfirst(x -> x == groups[i], unique_groups)
        if knockoff_idx[i]
            β_groups[idx] -= abs(β[i])
        else
            β_groups[idx] += abs(β[i])
        end
    end
    return β_groups
end

"""
    extract_beta(β̂_knockoff::Vector, fdr::Number, original::Vector{Int}, knockoff::Vector{Int}, method=:knockoff)
    extract_beta(β̂_knockoff::Vector, fdr::Number, groups::Vector{Int}, original::Vector{Int}, knockoff::Vector{Int}, method=:knockoff)

Given estimated β of original variables and their knockoffs, compute β for the
original design matrix that controls the FDR.
"""
function extract_beta(β̂_knockoff::AbstractVector{T}, fdr::Number, 
    original::AbstractVector{Int}, knockoff::AbstractVector{Int},
    method::Symbol=:knockoff, W::AbstractVector{T} = coefficient_diff(β̂_knockoff, original, knockoff)
    ) where T <: AbstractFloat
    # first handle errors
    p = length(β̂_knockoff) >> 1
    0 ≤ fdr ≤ 1 || error("Target FDR should be between 0 and 1 but got $fdr")
    # find knockoff-filter threshold
    τ = threshold(W, fdr, method)
    # construct the full β, thresholding indices that are not selected
    β = zeros(T, p)
    for i in eachindex(W)
        W[i] ≥ τ && (β[i] = β̂_knockoff[original[i]])
    end
    return β, W, τ
end

function extract_beta(β̂_knockoff::AbstractVector{T}, fdr::Number, groups::Vector{Int},
    original::AbstractVector{Int}, knockoff::AbstractVector{Int}, method=:knockoff,
    W::AbstractVector{T} = coefficient_diff(β̂_knockoff, groups, original, knockoff)
    ) where T <: AbstractFloat
    # first handle errors
    0 ≤ fdr ≤ 1 || error("Target FDR should be between 0 and 1 but got $fdr")
    # find knockoff-filter threshold
    τ = threshold(W, fdr, method)
    # construct the full β, thresholding indices that are not selected
    β = zeros(T, length(β̂_knockoff))
    for g in findall(W .≥ τ)
        group_idx = findall(x -> x == g, groups)
        β[group_idx] .= @view(β̂_knockoff[group_idx])
    end
    return β[original], W, τ
end
