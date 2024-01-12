"""
    threshold(w::AbstractVector, q::Number, [method=:knockoff], [m::Int=1])

Chooses a threshold τ > 0 by choosing `τ` to be one of the following
τ = min{ t > 0 : {#j: w[j] ≤ -t} / {#j: w[j] ≥ t} ≤ q }        (`method=:knockoff`)
τ = min{ t > 0 : (1 + {#j: w[j] ≤ -t}) / {#j: w[j] ≥ t} ≤ q }  (`method=:knockoff`)

# Inputs
+ `w`: Vector of feature important statistics
+ `q`: target FDR (between 0 and 1)
+ `method`: either `:knockoff` or `:knockoff_plus` (default)
+ `rej_bounds`: Number of values of top W to consider (default = 10000)

# Reference: 
Equation 3.10 (`method=:knockoff`) or 3.11 (`method=:knockoff_plus`) of 
"Panning for Gold: Model-X Knockoffs for High-dimensional
Controlled Variable Selection" by Candes, Fan, Janson, and Lv (2018)
"""
function threshold(w::AbstractVector{T}, q::Number,
    method=:knockoff_plus, rej_bounds::Int=10000) where T <: AbstractFloat
    0 ≤ q ≤ 1 || error("Target FDR should be between 0 and 1 but got $q")
    offset = method == :knockoff ? 0 : method == :knockoff_plus ? 1 :
        error("method should be :knockoff or :knockoff_plus but was $method.")
    τ = typemax(T)
    for (i, t) in enumerate(sort!(abs.(w), rev=true)) # t starts from largest |W|
        ratio = (offset + count(x -> x ≤ -t, w)) / count(x -> x ≥ t, w)
        ratio ≤ q && t > 0 && (τ = t)
        i > rej_bounds && break
    end
    return τ
end

"""
    mk_threshold(τ::Vector{T}, κ::Vector{Int}, m::Int, q::Number)

Chooses the multiple knockoff threshold `τ̂ > 0` by setting
τ̂ = min{ t > 0 : (1/m + 1/m * {#j: κ[j] ≥ 1 and W[j] ≥ t}) / {#j: κ[j] == 0 and W[j] ≥ τ̂} ≤ q }.

# Inputs
+ `τ`: τ[i] stores the feature importance score for the ith feature, i.e. the value
    T0 - median(T1,...,Tm). Note in Gimenez and Zou, the max function is used 
    instead of median
+ `κ`: κ[i] stores which of m knockoffs has largest importance score. When original 
    variable has largest score, κ[i] == 0.
+ `m`: Number of knockoffs per variable generated
+ `q`: target FDR (between 0 and 1)
+ `rej_bounds`: Number of values of top τ to consider (default = 10000)

# Reference: 
+ Equations 8 and 9 in supplement of "Identification of putative causal loci in 
    wholegenome sequencing data via knockoff statistics" by He et al. 
+ Algorithm 1 of "Improving the Stability of the Knockoff Procedure: Multiple 
    Simultaneous Knockoffs and Entropy Maximization" by Gimenez and Zou.
"""
function mk_threshold(τ::Vector{T}, κ::Vector{Int}, m::Int, q::Number,
    method=:knockoff_plus, rej_bounds::Int=10000
    ) where T <: AbstractFloat
    0 ≤ q ≤ 1 || error("Target FDR should be between 0 and 1 but got $q")
    method == :knockoff_plus || error("Multiple knockoffs needs to use :knockoff_plus filtering method")
    length(τ) == length(κ) || error("Length of τ and κ should be the same")
    p = length(τ) # number of features
    τ̂ = typemax(T)
    offset = 1 / m
    for (i, t) in enumerate(sort(τ, rev=true))
        numer_counter, demon_counter = 0, 0
        for i in 1:p
            κ[i] ≥ 1 && τ[i] ≥ t && (numer_counter += 1)
            κ[i] == 0 && τ[i] ≥ t && (demon_counter += 1)
        end
        ratio = (offset + offset * numer_counter) / max(1, demon_counter)
        ratio ≤ q && 0 < t < τ̂ && (τ̂ = t)
        i > rej_bounds && break
    end
    return τ̂
end

"""
    MK_statistics(T0::Vector, Tk::Vector{Vector}; filter_method)

Computes the multiple knockoff statistics kappa, tau, and W. 

# Inputs
+ `T0`: p-vector of importance score for original variables
+ `Tk`: Vector storing T1, ..., Tm, where Ti is importance scores for 
    the `i`th knockoff copy
+ `filter_method`: Either `Statistics.median` (default) or max (original 
    function used in 2019 Gimenez and Zou)

# output
+ `κ`: Index of the most significant feature (`κ[i] = 0` if original feature most 
    important, otherwise `κ[i] = k` if the `k`th knockoff is most important)
+ `τ`: `τ[i]` stores the most significant statistic among original and knockoff
    variables minus `filter_method()` applied to the remaining statistics. 
+ `W`: coefficient difference statistic `W[i] = abs(T0[i]) - abs(Tk[i])`     
"""
function MK_statistics(
    T0::Vector{T}, 
    Tk::Vector{Vector{T}};
    filter_method::Function = Statistics.median
    ) where T
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
        W[i] = (storage[1] - filter_method(@view(storage[2:end]))) * (κ[i] == 0)
        sort!(storage, rev=true)
        τ[i] = storage[1] - filter_method(@view(storage[2:end]))
    end
    return κ, τ, W
end

"""
    MK_statistics(T0::Vector, Tk::Vector)

Compute regular knockoff statistics tau and W. 

# Inputs
+ `T0`: p-vector of importance score for original variables
+ `Tk`: p-vector of importance score for knockoff variables

# output
+ `W`: coefficient difference statistic `W[i] = abs(T0[i]) - abs(Tk[i])`
"""
function MK_statistics(T0::Vector{T}, Tk::Vector{T}) where T
    p = length(T0)
    p == length(Tk) || error("Length of T0 should equal length of Tk")
    W = T[] # importance score of each feature
    for (t0, tk) in zip(T0, Tk)
        push!(W, abs(t0) - abs(tk))
    end
    return W
end
