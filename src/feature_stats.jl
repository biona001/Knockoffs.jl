"""
    coefficient_diff(β::AbstractVector)

Returns the coefficient difference statistic W[j] = |β[j]| - |β[j + p]| 
from a univariate (single response) regression. 
"""
function coefficient_diff(β::AbstractVector)
    iseven(length(β)) || error("length of β should be even but was odd.")
    p = length(β) >> 1
    W = Vector{eltype(β)}(undef, p)
    for j in 1:p
        W[j] = abs(β[j]) - abs(β[j + p])
    end
    return W
end

"""
    coefficient_diff(B::AbstractMatrix)

Returns the coefficient difference statistic
W[j] = |B[j, 1]| - |B[j + p, 1]| + ... +  |B[j, r]| - |B[j + p, r]|
from a multivariate (multiple response) regression. 
"""
function coefficient_diff(B::AbstractMatrix)
    iseven(size(B, 1)) || error("Number of covariates in B should be even but was odd.")
    p = size(B, 1) >> 1
    r = size(B, 2)
    W = Vector{eltype(B)}(undef, p)
    for j in 1:p
        Wj = zero(eltype(B))
        for i in 1:r
            Wj += abs(B[j, i]) - abs(B[j + p, i])
        end
        W[j] = Wj
    end
    return W
end
