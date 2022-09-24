"""
    threshold(w::AbstractVector, q::Number, [method=:knockoff], [m::Int=1])

Chooses a threshold τ > 0 by setting
τ = min{ t > 0 : {#j: w[j] ≤ -t} / {#j: w[j] ≥ t} ≤ q }.

# Inputs
+ `w`: Vector of feature important statistics
+ `q`: target FDR (between 0 and 1)
+ `method`: either `:knockoff` or `:knockoff_plus` (default)
+ `m`: Number of knockoff variables to generate, default `m=1`

# Reference: 
Equation 3.10 (`method=:knockoff`) or 3.11 (`method=:knockoff_plus`) of 
"Panning for Gold: Model-X Knockoffs for High-dimensional
Controlled Variable Selection" by Candes, Fan, Janson, and Lv (2018)
"""
function threshold(w::AbstractVector{T}, q::Number, method=:knockoff_plus) where T <: AbstractFloat
    0 ≤ q ≤ 1 || error("Target FDR should be between 0 and 1 but got $q")
    offset = method == :knockoff ? 0 : method == :knockoff_plus ? 1 :
        error("method should be :knockoff or :knockoff_plus but was $method.")
    τ = typemax(T)
    for t in sort!(abs.(w), rev=true) # t starts from largest |W|
        ratio = (offset + count(x -> x ≤ -t, w)) / count(x -> x ≥ t, w)
        ratio ≤ q && t > 0 && (τ = t)
    end
    return τ
end

"""
    mk_threshold(feature_scores::Vector{Vector{T}}, q::Number)

Chooses the multiple knockoff threshold `τ̂ > 0`. 

# Inputs
+ `feature_scores`: Vector of vector of feature important statistics. `feature_scores[i]`
    is a vector of length `m + 1` that stores feature importance score of variable `i`
    and its `m` knockoffs.
+ `q`: target FDR (between 0 and 1)

# Reference: 
Algorithm 1 of "Improving the Stability of the Knockoff Procedure: Multiple 
Simultaneous Knockoffs and Entropy Maximization" by Gimenez and Zou.
"""
function mk_threshold(τ::Vector{T}, κ::Vector{Int}, m::Int, q::Number, method=:knockoff_plus) where T <: AbstractFloat
    0 ≤ q ≤ 1 || error("Target FDR should be between 0 and 1 but got $q")
    method == :knockoff_plus || error("Multiple knockoffs needs to use :knockoff_plus filtering method")
    length(τ) == length(κ) || error("Length of τ and κ should be the same")
    p = length(τ) # number of features
    τ̂ = typemax(T)
    offset = 1 / m
    for t in τ
        numer_counter, demon_counter = 0, 0
        for i in 1:p
            κ[i] ≥ 2 && τ[i] ≥ t && (numer_counter += 1)
            κ[i] == 1 && τ[i] ≥ t && (demon_counter += 1)
        end
        ratio = (offset + offset * numer_counter) / demon_counter
        ratio ≤ q && 0 < t < τ̂ && (τ̂ = t)
    end
    return τ̂
end
