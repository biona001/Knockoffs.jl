"""
    select_features(β, original, knockoff, fdr, [method])
    select_features(β, original, knockoff, groups, fdr, [method])

Returns a `Vector{Int}` that includes the selected variables, the `W` statistic 
that measures each feature's importance score, and the threshold `τ` that
constitutes the threshold for selection. 

# Inputs
+ `β`: (m+1)p × 1 vector of feature importance statistics for the original and knockoff features
+ `original`: p × 1 vector of indices storing which columns of β contains the original features
+ `knockoff`: p × 1 vector of where knockoff[i] is a length m vector storing which 
    columns of β contains the `i`th knockoffs
+ `groups`: mp × 1 vector storing group membership for each column of XX̃
+ `fdr`: Target FDR level, a number between 0 and 1
+ `filter_method`: Choices are `:knockoff` or `:knockoff_plus` (default)
+ `mk_filter`: Choices are `Statistics.median` (default) or `maximum`. The original paper by Gimenez 
    and Zou uses `maximum` but He et al (https://www.nature.com/articles/s41467-021-22889-4)
    propose uses median which seems to be more stable in practice
"""
function select_features(
    β::AbstractVector{T},
    original::AbstractVector{Int}, 
    knockoff::Vector{Vector{Int}},
    fdr::Number;
    filter_method::Symbol=:knockoff_plus,
    mk_filter = Statistics.median
    ) where T
    p = length(knockoff)
    m = length(knockoff[1])
    length(original) == p || error("Expected length(original) == length(knockoff)")
    length(β) == (m + 1) * p || 
        error("β should contain effect sizes of original variables and their knockoffs")
    if m == 1 # single knockoff uses coefficient-difference statistic
        W = coefficient_diff(β, original, vcat(knockoff...))
        τ = threshold(W, fdr, filter_method)
        selected = findall(x -> x ≥ τ, W)
        return W, selected, τ
    end
    # multiple simultaneous knockoffs
    κ = zeros(Int, p) # κ[i] stores which of m knockoffs has largest importance score (κ[i]==0 if original variable has largest score)
    τ = zeros(T, p)   # τ[i] stores (T0 - mk_filter(T1,...,Tm)) where T0,...,Tm are ordered statistics
    W = zeros(T, p)   # W[i] stores (original_effect - mk_filter(importance scores of knockoffs)) * I(original beta has largest effect compared to all its knockoffs)
    T̃ = zeros(T, m)   # preallocated vector storing feature importance score for knockoff
    ordered = zeros(T, m + 1) # preallocated vector storing ordered statistics of the m+1 variables
    for i in 1:p
        # compute importance score of original feature and its m knockoffs
        original_effect = abs(β[original[i]])
        for j in 1:m
            T̃[j] = abs(β[knockoff[i][j]])
        end
        # find index of largest importance score among m+1 (original + m knockoff) features
        T̃max, max_idx = findmax(T̃)
        if T̃max > original_effect
            κ[i] = max_idx
        end
        # compute ordered statistic among the original feature and its m knockoffs
        ordered[1] = original_effect
        ordered[2:end] .= T̃
        sort!(ordered, rev=true)
        # compute importance statistic for current feature
        T0 = ordered[1]
        τ[i] = T0 - mk_filter(@view(ordered[2:end]))
        W[i] = (original_effect - mk_filter(T̃)) * (original_effect ≥ T̃max)
    end
    # compute multi-knockoff selection threshold
    τ̂ = mk_threshold(τ, κ, m, fdr, filter_method)
    selected = findall(x -> x ≥ τ̂, W)
    return W, selected, τ̂
end

function select_features(
    β::AbstractVector{T}, # length (m+1)×p  (first m+1 entries are feature 1 and its m knockoffs...etc)
    original::AbstractVector{Int}, # length p
    knockoff::Vector{Vector{Int}}, # length p where each knockoff[i] is length m
    groups::AbstractVector, # length (m+1)×p
    fdr::Number;
    filter_method::Symbol=:knockoff_plus,
    mk_filter = Statistics.median
    ) where T
    unique_groups = unique(groups)
    p = length(knockoff) # number of features
    m = length(knockoff[1]) # number of knockoffs per feature
    g = length(unique_groups) # number of groups
    length(original) == p || error("Expected length(original) == length(knockoff)")
    length(β) == (m + 1) * p || 
        error("β should contain effect sizes of original variables and their knockoffs")
    if m == 1 # single knockoff uses coefficient-difference statistic
        W = coefficient_diff(β, groups, original, vcat(knockoff...))
        τ = threshold(W, fdr, filter_method)
        selected = findall(x -> x ≥ τ, W)
        return W, selected, τ
    end
    # multiple simultaneous knockoffs Tg = mean(sum(β[g]))
    κ = zeros(Int, g) # κ[i] stores which of m knockoff groups has largest importance score (κ[i]==0 if original group has largest score)
    τ = zeros(T, g)   # τ[i] stores (T0 - median(T1,...,Tm)) where T0,...,Tm are ordered statistics
    W = zeros(T, g)   # W[i] stores (original_effect - mk_filter(importance scores of knockoffs)) * I(original beta has largest effect compared to all its knockoffs) / group size
    T̃ = zeros(T, m)   # preallocated vector storing feature importance score for knockoff
    ordered = zeros(T, m + 1) # preallocated vector storing ordered statistics of the m+1 variables
    original_variable_groups = groups[original]
    for (i, grp) in enumerate(unique_groups)
        group_idx = findall(x -> x == grp, groups)
        group_size = length(group_idx) / (m+1) # i and its m knockoffs
        # compute importance score of group i's original features
        original_effect = zero(T)
        for j in group_idx ∩ original
            original_effect += abs(β[j])
        end
        original_effect /= group_size
        # compute importance score of group i's m knockoffs
        fill!(T̃, 0)
        group_members = findall(x -> x == grp, original_variable_groups) # group_members are original variables that belong to current group
        for j in group_members
            for (idx, jj) in enumerate(knockoff[j]) # knockoff[j] stores indices of the jth variable's m knockoff
                T̃[idx] += abs(β[jj])
            end
        end
        T̃ ./= group_size
        # find index of largest importance score among m+1 (original + m knockoff) features
        T̃max, max_idx = findmax(T̃)
        if T̃max > original_effect
            κ[i] = max_idx
        end
        # compute ordered statistic among the original feature and its m knockoffs
        ordered[1] = original_effect
        ordered[2:end] .= T̃
        sort!(ordered, rev=true)
        # compute importance statistic for current feature
        T0 = ordered[1]
        τ[i] = T0 - mk_filter(@view(ordered[2:end]))
        W[i] = (original_effect - mk_filter(T̃)) * (original_effect ≥ T̃max)
    end
    # multi-knockoff selection
    τ̂ = mk_threshold(τ, κ, m, fdr, filter_method)
    selected = findall(x -> x ≥ τ̂, W)
    return W, selected, τ̂
end

function MK_statistics(T0::Vector{T}, Tk::Vector{Vector{T}}) where T
    p, m = length(T0), length(Tk)
    all(p .== length.(Tk)) || error("Length of T0 should equal all vectors in Tk")
    κ = zeros(Int, p) # index of largest importance score
    τ = zeros(p)      # difference between largest importance score and median of remaining
    W = zeros(p)      # importance score of each feature
    storage = zeros(m + 1)
    for i in 1:p
        storage[1] = abs(T0[i])
        for k in 1:m
            if abs(Tk[k][i]) > abs(T0[i])
                κ[i] = k
            end
            storage[k+1] = abs(Tk[k][i])
        end
        W[i] = (storage[1] - median(@view(storage[2:end]))) * (κ[i] == 0)
        sort!(storage, rev=true)
        τ[i] = storage[1] - median(@view(storage[2:end]))
    end
    return κ, τ, W
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
    coefficient_diff(β::AbstractVector, groups::Vector, original::Vector{Int}, knockoff::Vector{Int})

Returns the coefficient difference statistic for grouped variables. If
`compute_avg=true`, we compute the average beta for each group, otherwise
we compute the sum.

# Inputs
+ `β`: `2p × 1` vector of regression coefficients, including original and knockoff effect sizes
+ `groups`: `2p × 1` vector storing group membership. `groups[i]` is the group of `β[i]`
+ `original`: The index of original variables in `β`
+ `knockoff`: The index of knockoff variables in `β`
+ `compute_avg`: If true, feature importance for each group will average over
    absolute values of the betas. If false, we compute the sum instead.
"""
function coefficient_diff(β::AbstractVector, groups::AbstractVector,
    original::AbstractVector{Int}, knockoff::AbstractVector{Int}; compute_avg::Bool=true)
    length(β) == length(groups) || 
        error("coefficient_diff: length(β) = $(length(β)) does not equal length(groups) = $(length(groups))")
    unique_groups = unique(groups)
    W = zeros(length(unique_groups))
    # find which variables are Knockoffs
    knockoff_idx = falses(length(β))
    knockoff_idx[knockoff] .= true
    # loop over each variable
    for i in eachindex(β)
        idx = findfirst(x -> x == groups[i], unique_groups)
        if knockoff_idx[i]
            W[idx] -= abs(β[i])
        else
            W[idx] += abs(β[i])
        end
    end
    # average over group size
    if compute_avg
        for (i, g) in enumerate(unique_groups)
            W[i] /= count(x -> x == g, groups) / 2 # divide by 2 since groups include both original and knockoff variable
        end
    end
    return W
end

"""
    extract_beta(β̂_knockoff::Vector, fdr::Number, original::Vector{Int}, knockoff::Vector{Vector{Int}}, method=:knockoff)
    extract_beta(β̂_knockoff::Vector, fdr::Number, groups::Vector, original::Vector{Int}, knockoff::Vector{Vector{Int}}, method=:knockoff)

Given estimated β of original variables and their knockoffs in `β̂_knockoff`, 
zeros out the effect of non-selected features. 
"""
function extract_beta(β̂_knockoff::AbstractVector{T}, fdr::Number, 
    original::AbstractVector{Int}, knockoff::Vector{Vector{Int}},
    filter_method::Symbol=:knockoff_plus
    ) where T <: AbstractFloat
    # first handle errors
    p = length(original)
    0 ≤ fdr ≤ 1 || error("Target FDR should be between 0 and 1 but got $fdr")
    # select variables using knockoff filter
    W, selected_idx, τ = select_features(β̂_knockoff, original, knockoff, fdr, 
        filter_method=filter_method)
    # construct the full β, thresholding indices that are not selected
    β = zeros(T, p)
    for i in selected_idx
        β[i] = β̂_knockoff[original[i]]
    end
    return β, W, τ
end

function extract_beta(β̂_knockoff::AbstractVector{T}, fdr::Number, groups::Vector,
    original::AbstractVector{Int}, knockoff::Vector{Vector{Int}}, filter_method=:knockoff_plus
    ) where T <: AbstractFloat
    # first handle errors
    0 ≤ fdr ≤ 1 || error("Target FDR should be between 0 and 1 but got $fdr")
    # select variables using knockoff filter
    W, selected_groups, τ = select_features(β̂_knockoff, original, knockoff, groups, 
        fdr, filter_method=filter_method)
    # construct the full β, thresholding indices that are not selected
    β = zeros(T, length(β̂_knockoff))
    for g in selected_groups
        group_idx = findall(x -> x == g, groups)
        β[group_idx] .= @view(β̂_knockoff[group_idx])
    end
    return β[original], W, τ
end
