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

"""
    extract_beta(β̂_knockoff::AbstractVector, fdr::Number, method::Symbol=:concatenated)

Given estimated β of original variables and their knockoffs, compute β for the
original design matrix that controls the FDR.
"""
function extract_beta(β̂_knockoff::AbstractVector{T}, fdr::Number, 
    method::Symbol=:concatenated
    ) where T <: AbstractFloat
    0 ≤ fdr ≤ 1 || error("Target FDR should be between 0 and 1 but got $fdr")
    p = length(β̂_knockoff) >> 1
    # find set of selected predictors
    W = coefficient_diff(β̂_knockoff, method)
    τ = threshold(W, fdr)
    detected = findall(W .> τ)
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
