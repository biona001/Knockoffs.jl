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
    method = Symbol(method)
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
    get_knockoff_qvalue(w; method=:knockoff_plus, rej_bounds=10000)

Compute knockoff q-values for the single-knockoff filter. The q-value for each
feature is the minimum target FDR at which that feature is selected.
"""
function get_knockoff_qvalue(
    w::AbstractVector{T};
    method::Union{Symbol, String}=:knockoff_plus,
    rej_bounds::Int=10000,
    ) where T <: AbstractFloat
    method = Symbol(method)
    rej_bounds > 0 || error("rej_bounds should be positive but got $rej_bounds")
    offset = method == :knockoff ? 0 : method == :knockoff_plus ? 1 :
        error("method should be :knockoff or :knockoff_plus but was $method.")
    thresholds = sort(abs.(w), rev=true)
    max_index = min(length(thresholds), rej_bounds)
    ratio = zeros(Float64, max_index)
    for i in 1:max_index
        t = thresholds[i]
        ratio[i] = (offset + count(x -> x ≤ -t, w)) / count(x -> x ≥ t, w)
    end
    qvalues = ones(Float64, length(w))
    for j in eachindex(w)
        w[j] > 0 || continue
        qj = 1.0
        for i in 1:max_index
            thresholds[i] <= w[j] || continue
            qj = min(qj, ratio[i])
        end
        qvalues[j] = min(1.0, qj)
    end
    return qvalues
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
    method = Symbol(method)
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
    get_knockoff_qvalue(κ, τ, m; groups=collect(1:length(τ)), rej_bounds=10000)

Compute knockoff q-values for multiple knockoffs. The q-value for a variable is
the minimum target FDR at which that variable is selected.

# Inputs
+ `κ`: Index of the most significant feature (`κ[i] = 0` if original feature is
    most important)
+ `τ`: Multiple-knockoff feature statistic
+ `m`: Number of knockoffs per feature generated
+ `groups`: Group labels for each statistic in `τ` (defaults to all singleton groups)
+ `rej_bounds`: Number of top `τ` values considered when computing q-values

# References
+ Eq. 19 of "Powerful statistical method to detect disease associated genes using
  multiple knockoffs" (Nature Communications, 2022)
+ Within-group adaptation in Algorithm 2 of Gu and He (2024)
"""
function get_knockoff_qvalue(
    κ::AbstractVector{Int},
    τ::AbstractVector{T},
    m::Int;
    groups::AbstractVector=collect(1:length(τ)),
    rej_bounds::Int=10000,
    ) where T <: AbstractFloat
    m > 0 || error("Number of knockoffs m should be positive but got $m")
    length(τ) == length(κ) || error("Length of τ and κ should be the same")
    length(groups) == length(τ) || error("Length of groups and τ should be the same")
    rej_bounds > 0 || error("rej_bounds should be positive but got $rej_bounds")
    b = sortperm(τ, rev=true)
    κ_sorted = @view(κ[b])
    τ_sorted = @view(τ[b])
    groups_sorted = groups[b]
    c_0 = κ_sorted .== 0
    offset = 1 / m

    # calculate ratios for top rej_bounds τ values
    max_index = min(length(b), rej_bounds)
    ratio = zeros(Float64, max_index)
    temp_0 = 0
    for i in 1:max_index
        temp_0 += c_0[i]
        temp_1 = i - temp_0
        G_factor = maximum(values(countmap(@view(groups_sorted[1:i]))))
        ratio[i] = (offset * G_factor + offset * temp_1) / max(1, temp_0)
    end

    # calculate q values for top rej_bounds positive τ values
    qvalues = ones(Float64, length(τ))
    positive_bound = findlast(>(zero(T)), τ_sorted)
    if !isnothing(positive_bound)
        index_bound = min(positive_bound, max_index)
        for i in 1:index_bound
            c_0[i] || continue
            qvalues[b[i]] = minimum(@view(ratio[i:index_bound]))
        end
        qvalues[qvalues .> 1] .= 1
    end
    return qvalues
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
