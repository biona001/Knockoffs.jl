"""
    threshold(w::AbstractVector, q::Number, method=:knockoff)

Chooses a threshold τ > 0 by setting
τ = min{ t > 0 : {#j: w[j] ≤ -t} / {#j: w[j] ≥ -t} ≤ q }.

# Inputs
+ `w`: Vector of feature important statistics
+ `q`: target FDR (between 0 and 1)
+ `method`: either `:knockoff` or `:knockoff_plus`

# Reference: 
Equation 3.10 (`method=:knockoff`) or 3.11 (`method=:knockoff_plus`) of 
"Panning for Gold: Model-X Knockoffs for High-dimensional
Controlled Variable Selection" by Candes, Fan, Janson, and Lv (2018)
"""
function threshold(w::AbstractVector{T}, q::Number, method=:knockoff) where T <: AbstractFloat
    0 ≤ q ≤ 1 || error("Target FDR should be between 0 and 1 but got $q")
    offset = method == :knockoff ? 0 : method == :knockoff_plus ? 1 :
        error("method should be :knockoff or :knockoff_plus but was $method.")
    τ = typemax(T)
    for t in sort!(abs.(w), rev=true) # t starts from largest |W|
        ratio = (offset + count(x -> x ≤ -t, w)) / count(x -> x ≥ t, w)
        ratio ≤ q && (τ = t)
    end
    return τ
end
