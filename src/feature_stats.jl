"""
    coefficient_diff(β::AbstractVector, method=:concat)

Returns the coefficient difference statistic W[j] = |β[j]| - |β[j + p]| 
from a univariate (single response) regression. 

# Inputs
+ `β`: Vector of regression coefficients
+ `method`: Either `:concatenated` (default) if all knockoffs are concatenated
    at the end (e.g. [XX̃]) or `:interleaved` if each variable is immediately
    followed by its knockoff (e.g. [x₁x̃₁x₂x̃₂...])
"""
function coefficient_diff(β::AbstractVector, method::Symbol=:concatenated)
    iseven(length(β)) || error("length of β should be even but was odd.")
    p = length(β) >> 1
    W = Vector{eltype(β)}(undef, p)
    if method == :concatenated
        for j in 1:p
            W[j] = abs(β[j]) - abs(β[j + p])
        end
    elseif method == :interleaved
        for j in 1:p
            W[j] = abs(β[2j - 1]) - abs(β[2j])
        end
    else
        error("method should be :concatenated or :interleaved but got $method")
    end
    return W
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
    coefficient_diff(β::AbstractVector, original::AbstractVector{Int}, knockoff::AbstractVector{Int})

Returns the coefficient difference statistic for grouped variables
W[G] = sum_{j in G} |β[j]| - sum_{j in G} |β[j + p]|.

# Inputs
+ `β`: Vector of regression coefficients
+ `groups`: Vector storing group membership. `groups[i]` is the group of `β[i]`
+ `original`: The index of original variables in `β`
+ `knockoff`: The index of knockoff variables in `β`
"""
function coefficient_diff(β::AbstractVector, groups::AbstractVector{Int},
    original::AbstractVector{Int}, knockoff::AbstractVector{Int})
    unique_groups = unique(groups)
    β_groups = zeros(length(unique_groups))
    # find which variables are Knockoffs
    knockoff_idx = falses(length(β))
    knockoff_idx[knockoff] .= true
    # loop over each variable
    for i in 1:length(β)
        g = groups[i]
        if knockoff_idx[i]
            β_groups[g] -= abs(β[i])
        else
            β_groups[g] += abs(β[i])
        end
    end
    return β_groups
end

"""
    coefficient_diff(B::AbstractMatrix)

Returns the coefficient difference statistic
W[j] = |B[j, 1]| - |B[j + p, 1]| + ... +  |B[j, r]| - |B[j + p, r]|
from a multivariate (multiple response) regression. 

# Inputs
+ `β`: Matrix of regression coefficients. Each column is a vector of β. 
+ `method`: Either `:concatenated` if all knockoffs are concatenated at the end
    (e.g. [XX̃]) or `:interleaved` if each variable is immediately followed by
    its knockoff (e.g. [x₁x̃₁x₂x̃₂...])
"""
function coefficient_diff(B::AbstractMatrix, method::Symbol=:concatenated)
    iseven(size(B, 1)) || error("Number of covariates in B should be even but was odd.")
    p = size(B, 1) >> 1
    r = size(B, 2)
    T = eltype(B)
    W = Vector{T}(undef, p)
    if method == :concatenated
        for j in 1:p
            Wj = zero(T)
            for i in 1:r
                Wj += abs(B[j, i]) - abs(B[j + p, i])
            end
            W[j] = Wj
        end
    elseif method == :interleaved
        for j in 1:p
            Wj = zero(T)
            for i in 1:r
                Wj += abs(B[2j - 1, i]) - abs(B[2j, i])
            end
            W[j] = Wj
        end
    else
        error("method should be :concatenated or :interleaved but got $method")
    end
    return W
end

# """
#     coefficient_diff(B::AbstractMatrix, original::AbstractVector{Int}, knockoff::AbstractVector{Int})

# Returns the coefficient difference statistic W[j] = |β[j]| - |β[j + p]| 
# from a multivariate (multiple response) regression, where the `j`th variable is stored
# in position `original[j]` of `β`, and its knockoff is stored in position `knockoff[j]`

# # Inputs
# + `β`: Vector of regression coefficients
# + `original`: The index of original variables in `β`
# + `knockoff`: The index of knockoff variables in `β`
# """
# function coefficient_diff(B::AbstractMatrix, original::AbstractVector{Int}, knockoff::AbstractVector{Int})
#     p = size(B, 1) >> 1
#     length(original) == length(knockoff) == p || error("Number of variables in " * 
#         "B should be twice the length of original and knockoff.")
#     W = Vector{eltype(B)}(undef, p)
#     for j in 1:p
#         Wj = zero(T)
#         for i in 1:r
#             Wj += abs(B[original[j], i]) - abs(B[knockoff[j], i])
#         end
#         W[j] = Wj
#     end
#     return W
# end

"""
    extract_beta(β̂_knockoff::AbstractVector, fdr::Number, method::Symbol=:concatenated)

Given estimated β of original variables and their knockoffs, compute β for the
original design matrix that controls the FDR.
"""
function extract_beta(β̂_knockoff::AbstractVector{T}, fdr::Number, 
    method::Symbol=:concatenated
    ) where T <: AbstractFloat
    # first handle errors
    0 ≤ fdr ≤ 1 || error("Target FDR should be between 0 and 1 but got $fdr")
    p = length(β̂_knockoff) >> 1
    length(original) == length(knockoff) == p || error("Length of " * 
        "β should be twice of original and knockoff.")
    # find set of selected predictors
    W = coefficient_diff(β̂_knockoff, method)
    τ = threshold(W, fdr)
    detected = findall(W .≥ τ)
    # construct original β
    β = zeros(T, p)
    if method == :concatenated
        for i in detected
            β[i] = β̂_knockoff[i]
        end
    elseif method == :interleaved
        for i in detected
            β[i] = β̂_knockoff[2i - 1]
        end
    else
        error("method should be :concatenated or :interleaved but got $method")
    end
    return β
end

function extract_beta(β̂_knockoff::AbstractVector{T}, fdr::Number, 
    original::AbstractVector{Int}, knockoff::AbstractVector{Int}, method=:knockoff
    ) where T <: AbstractFloat
    # first handle errors
    p = length(β̂_knockoff) >> 1
    0 ≤ fdr ≤ 1 || error("Target FDR should be between 0 and 1 but got $fdr")
    length(original) == length(knockoff) == p || error("Length of " * 
        "β should be twice of original and knockoff.")
    # find set of selected predictors
    W = coefficient_diff(β̂_knockoff, original, knockoff)
    τ = threshold(W, fdr, method)
    detected = findall(W .≥ τ)
    # construct original β
    β = zeros(T, p)
    for i in detected
        β[i] = β̂_knockoff[original[i]]
    end
    return β
end

function extract_beta(β̂_knockoff::AbstractVector{T}, fdr::Number, groups::Vector{Int},
    original::AbstractVector{Int}, knockoff::AbstractVector{Int}, method=:knockoff
    ) where T <: AbstractFloat
    # first handle errors
    p = length(β̂_knockoff) >> 1
    0 ≤ fdr ≤ 1 || error("Target FDR should be between 0 and 1 but got $fdr")
    length(β̂_knockoff) == length(groups) ||
        error("β̂_knockoff should have same length as groups")
    length(original) == length(knockoff) == p ||
        error("Length of β should be twice of original and knockoff.")
    # find set of selected predictors
    W = coefficient_diff(β̂_knockoff, groups, original, knockoff)
    τ = threshold(W, fdr, method)
    detected_groups = findall(W .≥ τ)
    # construct original β
    β = zeros(T, p)
    for i in 1:p
        g = groups[i]
        if g in detected_groups
            β[i] = β̂_knockoff[original[i]]
        end
    end
    return β
end

function extract_combine_beta(β_full::AbstractVector{T}, 
    original::AbstractVector{Int}, knockoff::AbstractVector{Int}, 
    ) where T <: AbstractFloat
    p = length(β_full) >> 1
    # construct β by summing original β and β_ko
    β = zeros(T, p)
    for i in 1:p
        β[i] = β_full[original[i]] + β_full[knockoff[i]]
    end
    return β
end
