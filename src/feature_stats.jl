"""
    select_features(β, original, knockoff, fdr, [method])

Returns a `Vector{Int}` that includes the selected variables

# Inputs
"""
function select_features(
    β::AbstractVector{T}, 
    original::AbstractVector{Int}, 
    knockoff::AbstractVector{Int},
    fdr::Number;
    filter_method::Symbol=:knockoff_plus
    ) where T
    p = length(original)
    length(β) == p + length(knockoff) || 
        error("β should contain effect sizes of original variables and their knockoffs")
    m = Int(length(knockoff) / p)
    if m == 1 # single knockoff uses coefficient-difference statistic
        W = coefficient_diff(β, original, knockoff)
        τ = threshold(W, fdr, filter_method)
        return findall(x -> x ≥ τ, W)
    end
    # multiple simultaneous knockoffs
    κ, τ = zeros(Int, p), zeros(T, p)
    importance_scores = zeros(T, m + 1) # first entry stores score for the original feature
    for i in 1:p
        importance_scores[1] = abs(β[original[i]])
        for j in 1:m
            importance_scores[j + 1] = abs(β[knockoff[p * (j - 1) + i]])
        end
        κ[i] = argmax(importance_scores)
        τ[i] = maximum(importance_scores) - importance_scores[partialsortperm(importance_scores, 2, rev=true)]
    end
    τ̂ = mk_threshold(τ, κ, m, fdr, filter_method) # multi-knockoff selection threshold
    selected = Int[]
    for i in 1:p
        if τ[i] ≥ τ̂ && κ[i] == 1
            push!(selected, i)
        end
    end
    return selected
end

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
+ `groups`: `2p × 1` vector storing group membership. `groups[i]` is the group of `β[i]`
+ `original`: The index of original variables in `β`
+ `knockoff`: The index of knockoff variables in `β`
"""
function coefficient_diff(β::AbstractVector, groups::AbstractVector{Int},
    original::AbstractVector{Int}, knockoff::AbstractVector{Int})
    length(β) == length(groups) || error("coefficient_diff: length(β) = $(length(β)) does not equal length(groups) = $(length(groups))")
    unique_groups = unique(groups)
    β_groups = zeros(length(unique_groups))
    # find which variables are Knockoffs
    knockoff_idx = falses(length(β))
    knockoff_idx[knockoff] .= true
    # loop over each variable
    for i in eachindex(β)
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

Given estimated β of original variables and their knockoffs in `β̂_knockoff`, 
zeros out the effect of non-selected features. 
"""
function extract_beta(β̂_knockoff::AbstractVector{T}, fdr::Number, 
    original::AbstractVector{Int}, knockoff::AbstractVector{Int},
    filter_method::Symbol=:knockoff_plus
    ) where T <: AbstractFloat
    # first handle errors
    p = length(β̂_knockoff) >> 1
    0 ≤ fdr ≤ 1 || error("Target FDR should be between 0 and 1 but got $fdr")
    # select variables using knockoff filter
    selected_idx = select_features(β̂_knockoff, original, knockoff, fdr, filter_method=filter_method)
    # construct the full β, thresholding indices that are not selected
    β = zeros(T, p)
    for i in selected_idx
        β[i] = β̂_knockoff[original[i]]
    end
    return β
end

# todo: make this work for multiple group knockoffs
function extract_beta(β̂_knockoff::AbstractVector{T}, fdr::Number, groups::Vector{Int},
    original::AbstractVector{Int}, knockoff::AbstractVector{Int}, filter_method=:knockoff_plus
    ) where T <: AbstractFloat
    # first handle errors
    0 ≤ fdr ≤ 1 || error("Target FDR should be between 0 and 1 but got $fdr")
    # extract feature importance statistic
    W = coefficient_diff(β̂_knockoff, groups, original, knockoff)
    # find knockoff-filter threshold
    τ = threshold(W, fdr, filter_method)
    # construct the full β, thresholding indices that are not selected
    β = zeros(T, length(β̂_knockoff))
    for g in findall(W .≥ τ)
        group_idx = findall(x -> x == g, groups)
        β[group_idx] .= @view(β̂_knockoff[group_idx])
    end
    return β[original]
end
