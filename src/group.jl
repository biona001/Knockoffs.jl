"""
    modelX_gaussian_group_knockoffs(X, method, groups, μ, Σ; [m], [covariance_approximator], [kwargs])
    modelX_gaussian_group_knockoffs(X, method, groups; [m], [covariance_approximator], [kwargs])

Constructs Gaussian model-X group knockoffs. If the covariance `Σ` and mean `μ` 
are not specified, they will be estimated from data, i.e. we will make second-order
group knockoffs. To incorporate group structure, the (true or estimated) covariance 
matrix is block-diagonalized according to `groups` membership to solve a relaxed 
optimization problem. See reference paper and Knockoffs.jl docs for more details. 

# Inputs
+ `X`: A `n × p` design matrix. Each row is a sample, each column is a feature.
+ `method`: Method for constructing knockoffs. Options include
    * `:maxent`: (recommended) for fully general maximum entropy group knockoffs
    * `:mvr`: for fully general minimum variance-based reconstructability (MVR) group 
        knockoffs
    * `:equi`: for equi-correlated knockoffs. This is the methodology proposed in
        `Dai R, Barber R. The knockoff filter for FDR control in group-sparse and multitask regression. 
        International conference on machine learning 2016 Jun 11 (pp. 1851-1859). PMLR.`
    * `:sdp`: Fully general SDP group knockoffs based on coodinate descent
+ `groups`: Vector of group membership
+ `μ`: A length `p` vector storing the true column means of `X`
+ `Σ`: A `p × p` covariance matrix for columns of `X`
+ `m`: Number of knockoffs per variable, defaults to 1. 
+ `covariance_approximator`: A covariance estimator, defaults to 
    `LinearShrinkage(DiagonalUnequalVariance(), :lw)`. See CovarianceEstimation.jl 
    for more options.
+ `kwargs`: Extra keyword arguments for `solve_s_group`

# How to define groups
The exported functions `hc_partition_groups` can be used to build a group 
membership vector. 

# A note on compute time
The computational complexity of group knockoffs scales quadratically with group size.
Thus, very large groups (e.g. >100 members per group) dramatically slows down 
parameter estimation. In such cases, one can consider running the routine 
`modelX_gaussian_rep_group_knockoffs` which constructs group knockoffs by choosing
top representatives from each group. 

# Reference
Dai & Barber 2016, The knockoff filter for FDR control in group-sparse and multitask regression
"""
function modelX_gaussian_group_knockoffs(
    X::AbstractMatrix{T}, 
    method::Union{Symbol,String},
    groups::AbstractVector{Int};
    m::Number = 1,
    covariance_approximator=LinearShrinkage(DiagonalUnequalVariance(), :lw),
    kwargs... # extra arguments for solve_s_group
    ) where T
    # approximate covariance matrix
    Σapprox = cov(covariance_approximator, X)
    # mean component is just column means
    μ = vec(mean(X, dims=1))
    return modelX_gaussian_group_knockoffs(X, method, groups, μ, Σapprox; m=m, kwargs...)
end

function modelX_gaussian_group_knockoffs(
    X::AbstractMatrix{T}, 
    method::Union{Symbol,String},
    groups::AbstractVector{Int},
    μ::AbstractVector{T},
    Σ::AbstractMatrix{T};
    m::Number = 1,
    kwargs...
    ) where T
    # first check errors
    length(groups) == size(X, 2) || 
        error("Expected length(groups) == size(X, 2). Each variable in X needs a group membership.")
    typeof(method) <: String && (method = Symbol(method))
    # compute S matrix using the specified knockoff method
    S, γs, obj = solve_s_group(Symmetric(Σ), groups, method; m=m, kwargs...)
    # generate knockoffs
    X̃ = condition(X, μ, Σ, S; m=m)
    return GaussianGroupKnockoff(X, X̃, groups, S, γs, Int(m), Symmetric(Σ), method, obj)
end

"""
    modelX_gaussian_rep_group_knockoffs(X, method, groups; [m], [covariance_approximator], [kwargs...])
    modelX_gaussian_rep_group_knockoffs(X, method, groups, μ, Σ; [m], [kwargs...])

Constructs group knockoffs by choosing representatives from each group and
solving a smaller optimization problem based on the representatives only. Remaining
knockoffs are generated based on a conditional independence assumption similar to
a graphical model (details to be given later). The representatives are computed
by [`choose_group_reps`](@ref)

# Inputs
+ `X`: A `n × p` design matrix. Each row is a sample, each column is a feature.
+ `method`: Method for constructing knockoffs. Options are the same as 
    `modelX_gaussian_group_knockoffs`
+ `groups`: Vector of `Int` denoting group membership. `groups[i]` is the group 
    of `X[:, i]`
+ `covariance_approximator`: A covariance estimator, defaults to 
    `LinearShrinkage(DiagonalUnequalVariance(), :lw)`. See CovarianceEstimation.jl 
    for more options.
+ `μ`: A length `p` vector storing the true column means of `X`
+ `Σ`: A `p × p` covariance matrix for columns of `X`
+ `rep_threshold`: Value between 0 and 1 that controls the number of 
    representatives per group. Larger means more representatives (default 0.5)
+ `m`: Number of knockoffs per variable, defaults to 1. 
+ `kwargs`: Extra keyword arguments for `solve_s_group`
"""
function modelX_gaussian_rep_group_knockoffs(
    X::AbstractMatrix{T}, 
    method::Union{Symbol, String},
    groups::AbstractVector{Int};
    covariance_approximator=LinearShrinkage(DiagonalUnequalVariance(), :lw),
    m::Number = 1,
    rep_threshold::T = 0.5,
    kwargs... # extra arguments for solve_s_group
    ) where T
    Σapprox = cov(covariance_approximator, X) # approximate covariance matrix
    μ = vec(mean(X, dims=1)) # empirical column means
    return modelX_gaussian_rep_group_knockoffs(X, method, groups, μ, Σapprox;
        m=m, rep_threshold=rep_threshold, kwargs...)
end

# todo: Efficient sampling of knockoffs when `m>1` using conditional independence
function modelX_gaussian_rep_group_knockoffs(
    X::AbstractMatrix{T}, # n × p
    method::Union{Symbol, String},
    groups::AbstractVector{Int}, # p × 1 Vector{Int} of group membership
    μ::AbstractVector, # p × 1
    Σ::AbstractMatrix; # p × p
    m::Number = 1,
    rep_threshold::T = 0.5,
    verbose::Bool = false,
    enforce_cond_indep::Bool = false,
    kwargs... # extra arguments for solve_s_group
    ) where T
    size(X, 2) == length(groups)  || error("Dimensions of X and groups doesn't match")

    # compute group representatives
    group_reps = choose_group_reps(Symmetric(Σ), groups, threshold=rep_threshold)

    # decide which sigma to use
    sigma = enforce_cond_indep ? cond_indep_corr(Σ, groups, group_reps) : Σ

    # compute (block-diagonal) S on representatives and form larger (dense) D
    S, D, obj = solve_s_graphical_group(Symmetric(sigma), groups, group_reps, 
        method, m=m, verbose=verbose; kwargs...)

    # sample multiple knockoffs (todo: sample each independently)
    X̃ = condition(X, μ, sigma, Symmetric(D); m=m)

    return GaussianRepGroupKnockoff(X, X̃, groups, group_reps, S, 
        Symmetric(D), Int(m), Symmetric(sigma), method, obj, enforce_cond_indep)
end

"""
Returns `Σnew` as a covariance matrix that strictly satisfies the conditional
independence assumption. 
"""
function cond_indep_corr(
    Σ::AbstractMatrix{T}, 
    groups::AbstractVector{Int}, # group membership for each variable in Σ
    group_reps::AbstractVector{Int} # index of group representatives
    ) where T
    p = size(Σ, 1)
    Σnew = zeros(T, p, p)
    non_reps = setdiff(1:p, group_reps) # variables that are not representatives
    groups_of_reps = groups[group_reps] # groups membership of representatives
    # form group-block-diagonal matrices needed later
    Σblock1, Σblock2 = zeros(T, p, p), zeros(T, p, p)
    for g in unique(groups)
        g_rep_idx = group_reps[findall(x -> x == g, groups_of_reps)] # reps that belong to group g
        g_nonrep_idx = setdiff(findall(x -> x == g, groups), g_rep_idx) # non-reps that belong to group g
        Σg_RR_inv = inv(Σ[g_rep_idx,g_rep_idx])
        Σg_RRc = Σ[g_rep_idx, g_nonrep_idx]
        Σblock1[g_rep_idx, g_nonrep_idx] .= Σg_RR_inv * Σg_RRc
        Σblock2[g_nonrep_idx, g_nonrep_idx] .= @views Σ[g_nonrep_idx, g_nonrep_idx]
        Σblock2[g_nonrep_idx, g_nonrep_idx] .-= Σg_RRc' * Σg_RR_inv * Σg_RRc
    end
    # Σnew_11
    Σ11 = Σ[group_reps, group_reps]
    Σnew[group_reps, group_reps] .= Σ11
    # Σnew_12 and Σnew_21
    Σ12_diag = Σblock1[group_reps, non_reps]
    Σnew[group_reps, non_reps] .= Σ11 * Σ12_diag
    Σnew[non_reps, group_reps] .= @views Transpose(Σnew[group_reps, non_reps])
    # Σnew_22
    Σnew[non_reps, non_reps] .= @views Σblock2[non_reps, non_reps]
    Σnew[non_reps, non_reps] .+= Σ12_diag' * Σ11 * Σ12_diag
    return Σnew
end

"""
    hc_partition_groups(X::AbstractMatrix; [cutoff], [min_clusters], [force_contiguous])
    hc_partition_groups(Σ::Symmetric; [cutoff], [min_clusters], [force_contiguous])

Computes a group partition based on individual level data `X` or correlation 
matrix `Σ` using hierarchical clustering with specified linkage. 

# Inputs
+ `X`: `n × p` data matrix. Each row is a sample
+ `Σ`: `p × p` correlation matrix. Must be wrapped in the `Symmetric` argument,
    otherwise we will treat it as individual level data
+ `cutoff`: Height value for which the clustering result is cut, between 0 and 1
    (default 0.5). This ensures that no variables between 2 groups have correlation
    greater than `cutoff`. 1 recovers ungrouped structure, 0 corresponds to 
    everything in a single group. 
+ `min_clusters`: The desired number of clusters. 
+ `linkage`: *cluster linkage* function to use (when `force_contiguous=true`, 
    `linkage` must be `:single`). `linkage` defines how the 
    distances between the data points are aggregated into the distances between 
    the clusters. Naturally, it affects what clusters are merged on each 
    iteration. The valid choices are:
    + `:single` (default): use the minimum distance between any of the cluster members
    + `:average`: use the mean distance between any of the cluster members
    + `:complete`: use the maximum distance between any of the members
    + `:ward`: the distance is the increase of the average squared distance of a
        point to its cluster centroid after merging the two clusters
    + `:ward_presquared`: same as `:ward`, but assumes that the distances in d 
        are already squared.
+ `rep_method`: Method for selecting representatives for each group. Options are
    `:id` (tends to select roughly independent variables) or `:rss` (tends to
    select more correlated variables)

If `force_contiguous = false` and both `min_clusters` and `cutoff` are specified, 
it is guaranteed that the number of clusters is not less than `min_clusters` and
their height is not above `cutoff`. If `force_contiguous = true`, `min_clusters`
keyword is ignored. 

# Outputs
+ `groups`: Length `p` vector of group membership for each variable
+ `group_reps`: Columns of X selected as representatives. Each group have at 
    most `nrep` representatives. These are typically used to construct smaller
    group knockoff for extremely large groups
"""
function hc_partition_groups(
    Σ::Symmetric;
    cutoff = 0.5,
    min_clusters = 1,
    linkage::Union{String, Symbol}=:complete,
    force_contiguous = false
    )
    all(x -> x ≈ 1, diag(Σ)) || 
        error("Σ must be scaled to a correlation matrix first.")
    force_contiguous && linkage != :single &&
        error("When force_contiguous = true, linkage must be :single")
    typeof(linkage) <: String && (linkage = Symbol(linkage))
    # convert correlation matrix to a distance matrix
    distmat = copy(Matrix(Σ))
    @inbounds @simd for i in eachindex(distmat)
        distmat[i] = 1 - abs(distmat[i])
    end
    # hierarchical clustering
    if force_contiguous
        groups = adj_constrained_hclust(distmat, h=1-cutoff)
    else
        cluster_result = hclust(distmat; linkage=linkage)
        groups = cutree(cluster_result, h=1-cutoff, k=min_clusters)
    end
    return groups
end

function hc_partition_groups(X::AbstractMatrix; cutoff = 0.5, min_clusters = 1, 
    linkage=:complete, force_contiguous=false)
    return hc_partition_groups(Symmetric(cor(X)), cutoff=cutoff, 
        linkage=linkage,min_clusters=min_clusters, 
        force_contiguous=force_contiguous)
end

"""
    adj_constrained_hclust(distmat::AbstractMatrix, h::Number)

Performs (single-linkage) hierarchical clustering, forcing groups to be contiguous.
After clustering, variables in different group is guaranteed to have distance 
less than `h`. 

Note: this is a custom (bottom-up) implementation because `Clustering.jl` does not 
support adjacency constraints, see https://github.com/JuliaStats/Clustering.jl/issues/230
"""
function adj_constrained_hclust(distmat::AbstractMatrix{T}; 
    h::Number=0.3) where T
    0 ≤ h ≤ 1 || error("adj_constrained_hclust: expected 0 ≤ h ≤ 1 but got $h")
    p = size(distmat, 2)
    clusters = [[i] for i in 1:p] # initially all variables is its own cluster
    @inbounds for iter in 1:p-1
        remaining_clusters = length(clusters)
        min_d, max_d = typemax(T), typemin(T)
        merge_left, merge_right = 0, 0 # clusters to be merged
        # find min between-cluster distance
        for left in 1:remaining_clusters, right in left+1:remaining_clusters
            d = Knockoffs.single_linkage_distance(distmat, clusters[left], clusters[right])
            if d < min_d
                merge_left, merge_right = left, right
                min_d = d
            end
            d > max_d && (max_d = d)
        end
        # merge 2 clusters (and all those in between) with min distance
        for c in merge_left+1:merge_right
            for i in clusters[c]
                push!(clusters[merge_left], i)
            end
        end
        deleteat!(clusters, merge_left+1:merge_right)
        # check for convergence
        min_d ≥ h && break
    end
    # let each cluster be its own group
    groups = zeros(Int, p)
    for (i, cluster) in enumerate(clusters), g in cluster
        groups[g] = i
    end
    issorted(groups) || error("adj_constrained_hclust did not produce contiguous groups")
    return groups
end

"""
    single_linkage_distance(distmat, left, right)

Computes the minimum distance (i.e. single-linkage distance) between members
in `left` and members in `right`. Member distances are precomputed in `distmat`
"""
function single_linkage_distance(distmat::AbstractMatrix{T}, left::Vector{Int}, right::Vector{Int}) where T
    d = typemax(T)
    @inbounds for j in left, i in right
        new_d = distmat[i, j]
        new_d < d && (d = new_d)
    end
    return d
end

"""
    choose_group_reps(Σ::Symmetric, groups::AbstractVector; [threshold=0.5], [prioritize_idx], [Σinv])

Chooses group representatives. Returns indices of `Σ` that are representatives.
If R is the set of selected variables within a group and O is the set of variables
outside the group, then we keep adding variables to R until the proportion of
variance explained by R divided by the proportion of variance explained by R and
O exceeds `threshold`. 

# Inputs
+ `Σ`: Correlation matrix wrapped in the `Symmetric` argument.
+ `groups`: Vector of group membership. 

# Optional inputs
+ `threshold`: Value between 0 and 1 that controls the number of 
    representatives per group. Larger means more representatives (default 0.5)
+ `prioritize_idx`: Variable indices that should receive priority to be chosen
    as representatives, defaults to `nothing`
+ `Σinv`: Precomputed `inv(Σ)` (it will be computed if not supplied)
"""
function choose_group_reps(Σ::Symmetric{T}, groups::Vector{Int}; threshold=0.5, 
    prioritize_idx::Union{Vector{Int}, Nothing}=nothing, Σinv=nothing
    ) where T
    0 < threshold < 1 || error("threshold should be in (0, 1) but was $threshold")
    length(groups) == size(Σ, 1) ||
        error("Expected length(groups) == size(Σ, 1)")

    # Boundary case: remove duplicated (linearly dependent) columns first.
    independent_cols, dependent_cols = _split_dependent_columns(Σ, prioritize_idx)
    if !isempty(dependent_cols)
        Σsub = Symmetric(Σ[independent_cols, independent_cols])
        groups_sub = groups[independent_cols]
        prioritize_sub = _remap_indices(prioritize_idx, independent_cols)
        group_reps_sub = choose_group_reps(Σsub, groups_sub, threshold=threshold, 
            prioritize_idx=prioritize_sub)
        group_reps = independent_cols[group_reps_sub]
        return group_reps
    end

    # Main algorithm.
    isnothing(Σinv) && (Σinv = inv(Σ))
    unique_groups = unique(groups)
    group_reps = Int[]
    p = size(Σ, 1)
    storage1 = zeros(T, p, p)
    storage2 = zeros(T, p)
    @inbounds for g in unique_groups
        group_idx = findall(x -> x == g, groups)
        group_size = length(group_idx)
        if group_size == 1
            push!(group_reps, group_idx[1])
            continue
        end
        O = findall(x -> x != g, groups)

        # Compute an ordering of within-group importance, then increase the
        # number of representatives until the A1 stopping criterion is met.
        Σg = @view(Σ[group_idx, group_idx])
        index = select_best_rss_subset(Σg, group_size)
        if !isnothing(prioritize_idx)
            priority_pos = filter(!isnothing, indexin(prioritize_idx, group_idx[index]))
            if !isempty(priority_pos)
                index = prioritize_variants(index, index[priority_pos])
            end
        end

        indexΣ = group_idx[index]
        R = [indexΣ[1]]
        push!(group_reps, indexΣ[1])
        while length(R) < group_size
            ratio = _mean_explained_variance_ratio!(
                Σ, Σinv, R, O, indexΣ, storage1, storage2
            )
            if ratio > threshold
                break
            end
            next_rep = indexΣ[length(R) + 1]
            push!(R, next_rep)
            push!(group_reps, next_rep)
        end
    end
    return sort!(group_reps)
end

function _split_dependent_columns(
    Σ::Symmetric{T},
    prioritize_idx::Union{Vector{Int}, Nothing}
    ) where T
    p = size(Σ, 1)
    prioritized = isnothing(prioritize_idx) ? Set{Int}() : Set(prioritize_idx)
    dependent_cols = Int[]
    @inbounds for i in 1:p-1
        for j in i+1:p
            _columns_match(Σ, i, j) || continue
            keep_i = i in prioritized
            keep_j = j in prioritized
            if keep_i && !keep_j
                push!(dependent_cols, j)
            elseif keep_j && !keep_i
                push!(dependent_cols, i)
            else
                push!(dependent_cols, j)
            end
        end
    end
    sort!(unique!(dependent_cols))
    independent_cols = setdiff(1:p, dependent_cols)
    return independent_cols, dependent_cols
end

function _columns_match(
    Σ::Symmetric{T},
    i::Int,
    j::Int;
    atol::Real = 1e-10,
    rtol::Real = 1e-8
    ) where T
    @inbounds for k in axes(Σ, 1)
        isapprox(Σ[k, i], Σ[k, j], atol=atol, rtol=rtol) || return false
    end
    return true
end

function _remap_indices(
    prioritize_idx::Union{Vector{Int}, Nothing},
    kept_cols::Vector{Int}
    )
    isnothing(prioritize_idx) && return nothing
    old_to_new = Dict(col => i for (i, col) in pairs(kept_cols))
    remapped = Int[]
    for idx in prioritize_idx
        haskey(old_to_new, idx) && push!(remapped, old_to_new[idx])
    end
    return isempty(remapped) ? nothing : remapped
end

function _mean_explained_variance_ratio!(
    Σ::Symmetric{T},
    Σinv::AbstractMatrix,
    R::Vector{Int},
    O::Vector{Int},
    indexΣ::Vector{Int},
    storage1::AbstractMatrix{T},
    storage2::AbstractVector{T}
    ) where T
    RO = union(R, O)
    Rc = setdiff(indexΣ, R)
    ROc = setdiff(1:size(Σ, 1), RO)
    Σ_RR_inv = inv(Σ[R, R])

    # Compute inv(Σ[RO, RO]) from Σinv using block matrix inverse identities.
    L = cholesky(Symmetric(Σinv[ROc, ROc]))
    X = inv(L.L) * Σinv[ROc, RO]
    Σ_RORO_inv = @view(storage1[1:length(RO), 1:length(RO)])
    Σ_RORO_inv .= @view(Σinv[RO, RO])
    BLAS.syrk!('U', 'T', -one(T), X, one(T), Σ_RORO_inv)

    ratio = zero(T)
    for j in Rc
        Σ_Rj = Σ[R, j]
        Σ_ROj = Σ[RO, j]
        R2_R = _dot(Σ_Rj, Σ_RR_inv, Σ_Rj, storage2)
        R2_RO = _dot(Σ_ROj, Symmetric(Σ_RORO_inv), Σ_ROj, storage2)
        ratio += R2_R / R2_RO
    end
    return ratio / length(Rc)
end

"""
    prioritize_variants!(index::AbstractVector, priority_vars::AbstractVector)

Given (unsorted) `index`, we make variables in `priority_vars` appear first 
in `index`, preserving the original order in `index` and those not in 
`priority_vars`. 

# Example
```julia
index = [11, 4, 5, 9, 7]
priority_vars = [4, 9]
result = prioritize_variants(index, priority_vars)
result == [4, 9, 11, 5, 7]
```
"""
function prioritize_variants(index::AbstractVector, priority_vars::AbstractVector)
    first_idx = indexin(priority_vars, index)
    all(!isnothing, first_idx) || 
        error("Expected all variables in priority_vars to exist in index")
    second_idx = setdiff(1:length(index), first_idx)
    return [index[first_idx]; index[second_idx]]
end

# computes x'*A*y without allocation
function _dot(x, A, y, storage=zeros(size(A, 1)))
    p = size(A, 1)
    store = @views storage[1:p]
    mul!(store, A, y)
    return dot(x, store)
end

# faithful re-implementation of Trevor's R code. Probably not the most Julian/efficient Julia code
# select_one and select_best_rss_subset will help us choose k representatives from each group
# such that the RSS of the non-represented variables are minimized. Earlier 
# returned values are more important
function select_one(C::AbstractMatrix, vlist, RSS0, tol=1e-12)
    dC = diag(C)
    rs = vec(sum(C.^2, dims=1)) ./ dC
    v, imax = findmax(rs)
    vmin = sum(dC) - rs[imax]
    residC = C - (C[:,imax] * C[:,imax]' ./ C[imax, imax])
    index = vlist[imax]
    nzero = findall(x -> x > tol, diag(residC))
    R2 = 1 - vmin/RSS0
    return index, R2, residC[nzero, nzero], vlist[nzero]
end
function select_best_rss_subset(C::AbstractMatrix, k::Int, r2_threshold=1-1e-12)
    p = size(C, 2)
    # p ≤ k && return collect(1:p) # quick return
    indices = zeros(Int, k)
    RSS0 = p
    R2 = zeros(k)
    vlist = collect(1:p)
    for i in 1:k
        idx, r2, Cnew, vnew = select_one(C, vlist, RSS0)
        indices[i] = idx
        R2[i] = r2
        C = Cnew
        vlist = vnew
        r2 > r2_threshold && break # terminate alg when explained r2 ≈ 1 
    end
    # return non-0 indices
    return indices[findall(x -> x > 0, indices)]
end

"""
    search_rank(A::AbstractMatrix, sk::Vector{Int}, target=0.25, verbose=false)

Finds the rank (number of columns of A) that best approximates the remaining columns
such that regressing each remaining variable on those selected has RSS less than some
target. 

+ `Σ`: Original (p × p) correlation matrix
+ `A`: The (upper triangular) cholesky factor of Σ
+ `sk`: The (unsorted) columns of A, earlier ones are more important
+ `target`: Target residual level

note: we cannot do binary search because large ranks can increase residuals
"""
function search_rank(Σ::AbstractMatrix, A::AbstractMatrix, sk::Vector{Int}, target=0.25)
    p = size(A, 1)
    rk = 0
    invΣ = inv(Σ[sk[1], sk[1]])
    for k in 1:p
        selected = @view(sk[1:k])
        not_selected = @view(sk[k+1:end])
        # compute inv(Σ_SS) using block matrix inverse trick
        # https://math.stackexchange.com/questions/182309/block-inverse-of-symmetric-matrices
        if k > 1
            δ = @view(Σ[sk[1:k-1], sk[k]])
            Z = inv(Σ[sk[k], sk[k]])
            μ = Z - dot(δ, invΣ, δ)
            invΣδ = invΣ * δ
            invΣ = [ invΣ.+(invΣδ*invΣδ')/μ  -invΣδ/μ;
                        -invΣδ'/μ                1/μ  ]
            # invΣ_correct = inv(Σ[selected, selected])
            # @show all(invΣ .≈ invΣ_correct)
        end
        # check if residuals of remaining columns are lower than threshold
        success = test_residuals(invΣ, Σ, not_selected, selected, target)
        if success
            rk = k
            break
        end
    end
    return rk
end

function test_residuals(invΣ, Σ::AbstractMatrix{T}, not_selected, selected, target=0.25) where T
    S = selected
    k = length(S)
    success = true
    storage = zeros(T, k)
    for j in not_selected
        @views begin
            mul!(storage, invΣ, Σ[S, j])
            rss = Σ[j, j] - dot(Σ[j, S], storage)
        end
        if rss > target
            success = false
            break
        end
    end
    return success
end
