"""
    group_block_objective(Σ, S, groups, m, method)

Evaluate the objective for SDP/MVR/ME. This is not an efficient function, so it
should only be called at the start of each algorithm. 

# Inputs
+ `Σ`: Covariance or correlation matrix for original data
+ `S`: Optimization variable (group-block-diagonal)
+ `groups`: Vector of group membership. Variable `i` belongs to group `groups[i]`
+ `m`: Number of knockoffs to generate for each variable
+ `method`: The optimization method for group knockoffs
"""
function group_block_objective(Σ::AbstractMatrix{T}, S::AbstractMatrix{T}, 
    groups::Vector{Int}, m::Number, method) where T
    size(Σ) == size(S) || error("expected size(Σ) == size(S)")
    if occursin("sdp", string(method)) || occursin("equi", string(method))
        obj = zero(eltype(Σ))
        for g in unique(groups)
            idx = findall(x -> x == g, groups)
            obj += _sdp_block_objective(@view(Σ[idx, idx]), @view(S[idx, idx]))
        end
    elseif occursin("maxent", string(method))
        obj = logdet((m+1)/m*Σ - S + 1e-8I) + m*logdet(S + 1e-8I)
    elseif occursin("mvr", string(method))
        # obj += m^2*tr(inv(S + 1e-8I)) + tr(inv((m+1)/m*Σ - S + 1e-8I))
        obj = m^2*tr(inv(S)) + tr(inv((m+1)/m*Σ - S))
    else
        error("unrecognized method: method should be one of $GROUP_KNOCKOFFS")
    end
    return obj
end

# helper function to evaluate the SDP objective for a single block
function _sdp_block_objective(Σg, Sg)
    size(Σg) == size(Sg) || error("Expected size(Σg) == size(Sg)")
    obj = zero(eltype(Σg))
    @inbounds for j in axes(Σg, 2)
        @simd for i in axes(Σg, 1)
            obj += abs(Σg[i, j] - Sg[i, j])
        end
    end
    g = size(Σg, 1)
    return obj / g^2
end

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
    * `:sdp_block`: Fully general SDP group knockoffs where each block is solved exactly 
        using an interior point solver. 
    * `:sdp_subopt`: Chooses each block `S_{i} = γ_i * Σ_{ii}`. This slightly 
        generalizes the equi-correlated group knockoff idea proposed in Dai and Barber 2016.
+ `groups`: Vector of group membership
+ `μ`: A length `p` vector storing the true column means of `X`
+ `Σ`: A `p × p` covariance matrix for columns of `X`
+ `m`: Number of knockoffs per variable, defaults to 1. 
+ `covariance_approximator`: A covariance estimator, defaults to 
    `LinearShrinkage(DiagonalUnequalVariance(), :lw)`. See CovarianceEstimation.jl 
    for more options.
+ `kwargs`: Extra keyword arguments for `solve_s_group`

# How to define groups
The exported functions `hc_partition_groups` and `id_partition_groups` can be used
to build a group membership vector. 

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
    X̃ = condition(X, μ, Σ, Symmetric(D); m=m)

    return GaussianRepGroupKnockoff(X, X̃, groups, group_reps, S, 
        Symmetric(D), Int(m), Symmetric(Σ), method, obj, enforce_cond_indep)
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
    solve_s_graphical_group(Σ::Symmetric, groups::Vector{Int}, group_reps::Vector{Int},
    method; [m], [verbose])

Solves the group knockoff problem but the convex optimization problem only runs
on the representatives. The non-representative variables are assumed to be 
independent by groups when conditioning on the reprensetatives. 

# Inputs
+ `Σ`: Symmetric `p × p` covariance matrix
+ `groups`: `p` dimensional vector of group membership
+ `group_reps`: Indices for the representatives. 
+ `method`: Method for solving group knockoff problem
+ `m`: Number of knockoffs to generate per feature
+ `verbose`: Whether to print informative intermediate results
+ `kwargs...`: extra arguments for [`solve_s_group`](@ref)

# Outputs
+ `S`: Matrix obtained from solving the optimization problem on the representatives.
+ `D`: A `p × p` (dense) matrix corresponding to the S matrix for both the
    representative and non-representative variables. Knockoff sampling should 
    use this matrix. If the graphical conditional independent assumption is 
    satisfied exactly, this matrix should be sparse, but it is always never sparse
    unless we use `cond_indep_corr` to force the covariance matrix to satisify it. 
+ `obj`: Objective value for solving the optimization problem on the representatives. 
"""
function solve_s_graphical_group(
    Σ::Symmetric{T}, # p × p
    groups::AbstractVector{Int}, # p × 1 Vector{Int} of group membership
    group_reps::AbstractVector{Int}, # Vector{Int} of representatives
    method::Union{Symbol, String};
    m::Number = 1,
    verbose::Bool = false,
    kwargs... # extra arguments for solve_s_group
    ) where T
    p = size(Σ, 1)
    group_size = countmap(groups[group_reps]) |> values |> collect
    r = length(group_reps)
    verbose && println("$r representatives for $p variables, " * 
        "$(sum(abs2, group_size)) optimization variables"); flush(stdout)

    # Compute S matrix on the representatives
    non_reps = setdiff(1:p, group_reps)
    Σ11 = Σ[group_reps, group_reps] # no view because Σ11 needs to be inverted later
    Σ12 = @views Σ[group_reps, non_reps]
    Σ22 = @views Σ[non_reps, non_reps]
    S, _, obj = solve_s_group(Symmetric(Σ11), groups[group_reps], method; 
        m=m, verbose=verbose, kwargs...)

    # form full S matrix (call it D) using conditional independence assumption
    Σ11inv = inv(Σ11)
    Σ11inv_Σ12 = Σ11inv * Σ12
    S_Σ11inv_Σ12 = S * Σ11inv_Σ12 # r × (p-r)
    D = Matrix{T}(undef, p, p)
    D[group_reps, group_reps] .= S
    D[group_reps, non_reps] .= S_Σ11inv_Σ12
    D[non_reps, group_reps] .= S_Σ11inv_Σ12'
    D[non_reps, non_reps] .= Σ22 - 
        (Σ12' * Σ11inv * Σ12) + (Σ11inv_Σ12' * S * Σ11inv_Σ12)

    # threshold small values to 0
    D[findall(x -> abs(x) < 1e-10, D)] .= 0

    return S, D, obj
end

"""
    solve_s_group(Σ, groups, method; [m=1], kwargs...)

Solves the group knockoff problem, returns block diagonal matrix S
satisfying `(m+1)/m*Σ - S ⪰ 0` where `m` is number of knockoffs per feature. 

# Inputs 
+ `Σ`: A general covariance matrix wrapped by `Symmetric` keyword
+ `groups`: Vector of group membership, does not need to be contiguous
+ `method`: Method for constructing knockoffs. Options include
    * `:maxent`: (recommended) for fully general maximum entropy group knockoffs
    * `:mvr`: for fully general minimum variance-based reconstructability (MVR) group 
        knockoffs
    * `:equi`: for equi-correlated knockoffs. This is the methodology proposed in
        `Dai R, Barber R. The knockoff filter for FDR control in group-sparse and multitask regression. 
        International conference on machine learning 2016 Jun 11 (pp. 1851-1859). PMLR.`
    * `:sdp`: Fully general SDP group knockoffs based on coodinate descent
    * `:sdp_subopt`: Chooses each block `S_{i} = γ_i * Σ_{ii}`. This slightly 
        generalizes the equi-correlated group knockoff idea proposed in Dai and Barber 2016.
    * `:sdp_block`: Fully general SDP group knockoffs where each block is solved exactly 
        using an interior point solver. 
+ `m`: Number of knockoffs per variable, defaults to 1. 
+ `kwargs`: Extra arguments available for specific methods. For example, to use 
    less stringent convergence tolerance, specify `tol = 0.001`.
    For a list of available options, see [`solve_group_mvr_hybrid`](@ref),
    [`solve_group_max_entropy_hybrid`](@ref), [`solve_group_sdp_hybrid`](@ref), or
    [`solve_group_equi`](@ref)

# Output
+ `S`: A matrix solved so that `(m+1)/m*Σ - S ⪰ 0` and `S ⪰ 0`
+ `γ`: A vector that is only non-empty for equi and suboptimal knockoff constructions. 
    They correspond to values of γ where `S_{gg} = γΣ_{gg}`. So for equi, the
    vector is length 1. For SDP, the vector has length equal to number of groups
+ `obj`: Final SDP/MVR/ME objective value given `S`. Equi-correlated group knockoffs
    and singleton (non-grouped knockoffs) returns 0 because they either no objective 
    value or it is not necessary to evaluate the objectives

# Warning
This function potentially permutes the columns/rows of `Σ`, and puts them back
at the end. Thus one should NOT call `solve_s_group` on the same `Σ` simultaneously,
e.g. in a multithreaded for loop. Permutation does not happen when groups are
contiguous. 
"""
function solve_s_group(
    Σ::Symmetric{T}, 
    groups::Vector{Int},
    method::Union{Symbol, String};
    m::Number=1,
    kwargs...
    ) where T
    # check for errors
    length(groups) == size(Σ, 1) || 
        error("Length of groups should be equal to dimension of Σ")
    max_group_size = countmap(groups) |> values |> collect |> maximum
    if max_group_size > 50 && method != :equi && !occursin("pca", string(method))
        @warn "Maximum group size is $max_group_size, optimization may be slow. " * 
            "Consider running `modelX_gaussian_rep_group_knockoffs` to speed up convergence."
        flush(stdout)
    end
    method = Symbol(method)
    # Scale covariance to correlation matrix
    σs = sqrt.(diag(Σ))
    iscor = all(x -> x ≈ 1, σs)
    Σcor = Symmetric(cov2cor(Σ.data, σs))
    # if groups not contiguous, permute columns/rows of Σ so that they are contiguous
    perm = sortperm(groups)
    group_permuted = copy(groups)
    permuted = false
    if !issorted(groups)
        permute!(group_permuted, perm)
        Σcor.data .= @view(Σcor.data[perm, perm])
        permuted = true
    end
    if length(unique(groups)) == length(groups)
        # solve ungroup knockoff problem (todo: delete kwargs unique to solve_s_group)
        s = solve_s(Symmetric(Σcor), 
            method == :sdp_subopt ? :sdp : method;
            m=m, kwargs...
        )
        S = Diagonal(s) |> Matrix
        γs = T[]
        obj = zero(T)
    else
        # solve group knockoff optimization problem
        if method == :equi
            S, γs, obj = solve_group_equi(Σcor, group_permuted; m=m)
        elseif method == :sdp_subopt
            S, γs, obj = solve_group_SDP_subopt(Σcor, group_permuted; m=m)
        elseif method == :sdp_subopt_correct
            S, γs, obj = solve_group_SDP_subopt_correct(Σcor, group_permuted; m=m)
        elseif method == :sdp_block
            S, γs, obj = solve_group_block_update(Σcor, group_permuted, method; m=m, kwargs...)
        elseif method == :mvr_block
            S, γs, obj = solve_group_block_update(Σcor, group_permuted, method; m=m, kwargs...)
        elseif method == :maxent_block
            S, γs, obj = solve_group_block_update(Σcor, group_permuted, method; m=m, kwargs...)
        elseif method == :sdp_full
            S, γs, obj, _, _ = solve_group_SDP_full(Σcor, group_permuted; m=m)
        elseif method == :sdp
            S, γs, obj = solve_group_sdp_hybrid(Σcor, group_permuted; m=m, kwargs...)
        elseif method == :mvr
            S, γs, obj, _, _ = solve_group_mvr_hybrid(Σcor, group_permuted; m=m, kwargs...)
        elseif method == :maxent
            S, γs, obj, _, _ = solve_group_max_entropy_hybrid(Σcor, group_permuted; m=m, kwargs...)
        else
            error("Method must be one of $GROUP_KNOCKOFFS but was $method")
        end
    end
    # permuate S and Σ back to the original noncontiguous group structure
    if permuted
        iperm = invperm(perm)
        S .= @view(S[iperm, iperm])
        Σcor.data .= @view(Σcor.data[iperm, iperm])
    end
    # rescale S back to the result for a covariance matrix   
    iscor || cor2cov!(S, σs)
    return S, γs, obj
end

"""
    initialize_S(Σ, groups, m, method, verbose)

Internal function to help initialize `S` to a good starting value, returns the
final `S` matrix as well as the cholesky factorizations `L` and `C` where
+ L.L*L.U = cholesky((m+1)/m*Σ - S)
+ C.L*C.U = cholesky(S)
"""
function initialize_S(Σ, groups::Vector{Int}, m::Number, method, ϵ=1e-8)
    S, _, _ = solve_group_equi(Σ, groups, m=m)
    # make minimum eigenvalue ϵ
    evals, evecs = eigen(S)
    evals[findall(x -> x < ϵ, evals)] .= ϵ
    S = evecs * Diagonal(evals) * evecs'
    # do not start at boundary condition
    S ./= 2
    L = cholesky(Symmetric((m+1)/m * Σ - S))
    C = cholesky(Symmetric(S))
    return S, L, C
end

"""
Computes A^{-1/2} via eigen-decomposition
"""
function inverse_mat_sqrt(A::Symmetric; tol=1e-6)
    λ, ϕ = eigen(A)
    for i in eachindex(λ)
        λ[i] < tol && (λ[i] = tol)
    end
    return ϕ * Diagonal(1 ./ sqrt.(λ)) * ϕ'
end

"""
    block_diagonalize(Σ, groups)

Internal function to block-diagonalize the covariance `Σ` according to groups. 
"""
function block_diagonalize(Σ::AbstractMatrix, groups::Vector{Int})
    Σblocks = Matrix{eltype(Σ)}[]
    for g in unique(groups)
        idx = findall(x -> x == g, groups)
        push!(Σblocks, Σ[idx, idx])
    end
    return BlockDiagonal(Σblocks)
end

"""
Solves the equi-correlated group knockoff problem. Here
`Σ` is the true covariance matrix (scaled so that it has 1 on its diagonal)
and `Σblocks` is the block-diagonal covariance matrix where each 
block corresponds to groups.

Details can be found in
Dai & Barber 2016, The knockoff filter for FDR control in group-sparse and multitask regression
"""
function solve_group_equi(
    Σ::AbstractMatrix, 
    groups::Vector{Int};
    m::Number = 1 # number of knockoffs per feature to generate
    )
    Σblocks = block_diagonalize(Σ, groups)
    Db = Matrix{eltype(Σ)}[]
    for Σbi in Σblocks.blocks
        push!(Db, inverse_mat_sqrt(Symmetric(Σbi)))
    end
    Db = BlockDiagonal(Db)
    λmin = Symmetric(Db * Σ * Db) |> eigmin
    γ = min(1, (m+1)/m * λmin)
    S = BlockDiagonal(γ .* Σblocks.blocks) |> Matrix
    obj = group_block_objective(Σ, S, groups, m, :equi)
    return S, [γ], obj
end

"""
Solves the SDP group knockoff problem using analogy to the equi-correlated
group knockoffs. Basically, the idea is to optimize a vector `γ` where `γ[j]` 
multiplies Σ_jj. In the equi-correlated setting, all `γ[j]` is forced to be equal.

Details can be found in
Dai & Barber 2016, The knockoff filter for FDR control in group-sparse and multitask regression
"""
function solve_group_SDP_subopt(
    Σ::AbstractMatrix, 
    groups::Vector{Int}; 
    m::Number = 1,
    verbose=false
    )
    model = Model(() -> Hypatia.Optimizer(verbose=verbose))
    # model = Model(() -> SCS.Optimizer())
    Σblocks = block_diagonalize(Σ, groups)
    n = nblocks(Σblocks)
    block_sizes = size.(Σblocks.blocks, 1)
    @variable(model, 0 <= γ[1:n] <= 1)
    blocks = BlockDiagonal([γ[i] * Σblocks.blocks[i] for i in 1:n]) |> Matrix
    @objective(model, Max, block_sizes' * γ)
    @constraint(model, Symmetric((m+1)/m*Σ - blocks) in PSDCone())
    JuMP.optimize!(model)
    success = check_model_solution(model)
    if !success
        @warn "Optimization unsuccessful, solution may be inaccurate"
    end
    # return solution
    γs = clamp!(JuMP.value.(γ), 0, 1)
    S = BlockDiagonal(γs .* Σblocks.blocks) |> Matrix
    obj = group_block_objective(Σ, S, groups, m, :sdp_subopt)
    return S, γs, obj
end

function solve_group_SDP_subopt_correct(
    Σ::AbstractMatrix, 
    groups::Vector{Int}; 
    m::Number = 1,
    verbose=false
    )
    model = Model(() -> Hypatia.Optimizer(verbose=verbose))
    Σblocks = block_diagonalize(Σ, groups)
    n = nblocks(Σblocks)
    block_sizes = size.(Σblocks.blocks, 1)
    @variable(model, γ[1:n])
    blocks = BlockDiagonal([γ[i] * Σblocks.blocks[i] for i in 1:n]) |> Matrix
    @constraint(model, Symmetric((m+1)/m*Σ - blocks) in PSDCone())
    @constraint(model, Symmetric(blocks) in PSDCone())
    # slack variables
    @variable(model, U[1:sum(block_sizes.^2)])
    # loop over each block
    offset = 0 # allows indexing over blocks of S
    Uidx = 1   # index of U if U were treated as a matrix, i.e index of U[i, j]
    for g in 1:n
        G = block_sizes[g] # g is group idx, G is size of group g
        cur_idx = offset + 1:offset + G
        for i in cur_idx, j in cur_idx
            @constraint(model, Σ[i, j] - γ[g]*Σ[i, j] ≤ U[Uidx])
            @constraint(model, -U[Uidx] ≤ Σ[i, j] - γ[g]*Σ[i, j])
            Uidx += 1
        end
        offset += G
    end
    @objective(model, Min, sum(U))
    JuMP.optimize!(model)
    success = check_model_solution(model)
    if !success
        @warn "Optimization unsuccessful, solution may be inaccurate"
    end
    # return solution
    γs = JuMP.value.(γ)
    S = BlockDiagonal(γs .* Σblocks.blocks) |> Matrix
    obj = group_block_objective(Σ, S, groups, m, :sdp)
    return S, γs, obj
end

"""
    solve_group_SDP_single_block(Σ11, ub)

Solves a single block of the fully general group SDP problem. The objective is
    min  sum_{i,j} |Σ[i,j] - S[i,j]|
    s.t. 0 ⪯ S ⪯ A11 - [A12 A13]*inv(A22-S2 A32; A23 A33-S3)*[A21; A31]

# Inputs
+ `Σ11`: The block corresponding to the current group. Must be a correlation matrix. 
+ `ub`: The matrix defined as A11 - [A12 A13]*inv(A22-S2 A32; A23 A33-S3)*[A21; A31]
+ `optm`: Any solver compatible with JuMP.jl
"""
function solve_group_SDP_single_block(
    Σ11::AbstractMatrix,
    ub::AbstractMatrix; # this is upper bound, equals [A12 A13]*inv(A22-S2 A32; A23 A33-S3)*[A21; A31]
    optm=Hypatia.Optimizer(verbose=false, iter_limit=100) # Any solver compatible with JuMP
    # optm=Hypatia.Optimizer(verbose=false, iter_limit=100, tol_rel_opt=1e-4, tol_abs_opt=1e-4) # Any solver compatible with JuMP
    )
    # quick return for singleton groups
    if size(ub) == (1, 1)
        if Σ11[1] ≤ ub[1]
            return Σ11, true
        else
            return ub .- 1e-6, true
        end
    end
    # Build model via JuMP
    p = size(Σ11, 1)
    model = Model(() -> optm)
    @variable(model, -1 ≤ S[1:p, 1:p] ≤ 1, Symmetric)
    # slack variables to handle absolute value in obj 
    @variable(model, U[1:p, 1:p], Symmetric)
    for i in 1:p, j in i:p
        @constraint(model, Σ11[i, j] - S[i, j] ≤ U[i, j])
        @constraint(model, -U[i, j] ≤ Σ11[i, j] - S[i, j])
    end
    @objective(model, Min, sum(U)) # equivalent to @objective(model, Min, sum(abs.(Σ11 - S)))
    # SDP constraints
    @constraint(model, S in PSDCone())
    @constraint(model, ub - S in PSDCone())
    # solve and return
    JuMP.optimize!(model)
    success = check_model_solution(model)
    return JuMP.value.(S), success
end

function solve_group_maxent_single_block(
    Σ11::AbstractMatrix,
    ub::AbstractMatrix, # this is upper bound, equals [A12 A13]*inv(A22-S2 A32; A23 A33-S3)*[A21; A31]
    m::Number; # number of knockoffs to generate
    optm=Hypatia.Optimizer(verbose=false, iter_limit=100) # Any solver compatible with JuMP
    # optm=Hypatia.Optimizer(verbose=false, iter_limit=100, tol_rel_opt=1e-4, tol_abs_opt=1e-4) # Any solver compatible with JuMP
    )
    # todo: quick return for singleton groups
    # Build model via JuMP
    p = size(Σ11, 1)
    model = Model(() -> optm)
    @variable(model, -1 ≤ S[1:p, 1:p] ≤ 1, Symmetric)
    # SDP constraints
    @constraint(model, S in PSDCone())
    @constraint(model, ub - S in PSDCone())
    # logdet objective needs to be handled by converting it to conic problem:
    #     max log(det(x))
    # is equivalent to
    #     max t
    #     s.t. t <= log(det(X))
    # see: https://discourse.julialang.org/t/log-determinant-objective/23927/6
    @variable(model, t)
    @variable(model, u)
    @constraint(model, [t; 1; vec(ub - S)] in MOI.LogDetConeSquare(p))
    @constraint(model, [u; 1; vec(S)] in MOI.LogDetConeSquare(p))
    @objective(model, Max, t + u)
    # solve and return
    JuMP.optimize!(model)
    success = check_model_solution(model)
    return JuMP.value.(S), success
end

function solve_group_MVR_single_block(
    Σ11::AbstractMatrix,
    ub::AbstractMatrix, # this is upper bound, equals [A12 A13]*inv(A22-S2 A32; A23 A33-S3)*[A21; A31]
    A21::AbstractMatrix,
    D22inv::AbstractMatrix,
    m::Number; # number of knockoffs to generate
    optm=Hypatia.Optimizer(verbose=false, iter_limit=100) # Any solver compatible with JuMP
    # optm=Hypatia.Optimizer(verbose=false, iter_limit=100, tol_rel_opt=1e-4, tol_abs_opt=1e-4) # Any solver compatible with JuMP
    )
    # Build model via JuMP
    p = size(Σ11, 1)
    q = size(D22inv, 1)
    model = Model(() -> optm)
    @variable(model, -1 ≤ S[1:p, 1:p] ≤ 1, Symmetric)
    # SDP constraints
    @constraint(model, S in PSDCone())
    @constraint(model, ub - S in PSDCone())
    # convert tr(inv(X)) terms into linear matrix inequalities using slack variable trick
    # https://discourse.julialang.org/t/how-to-optimize-trace-of-matrix-inverse-with-jump-or-convex/94167/4
    @variable(model, X[1:p, 1:p])
    @variable(model, Y[1:p, 1:p])
    @variable(model, Z[1:q]) # force Z to be diagonal matrix rather than q by q matrix
    @constraint(model, [X I; I S] in PSDCone())
    @constraint(model, [Y I; I ub-S] in PSDCone())
    C = D22inv * A21
    @constraint(model, [Diagonal(Z) C; Transpose(C) ub-S] in PSDCone())
    # objective
    @objective(model, Min, m^2*tr(X) + tr(Y) + sum(Z))
    # solve and return
    JuMP.optimize!(model)
    # success = check_model_solution(model)
    success = eigmin(JuMP.value.(S)) ≥ 0 ? true : false
    return JuMP.value.(S), success
end

"""
# Todo
+ somehow avoid reallocating ub every iteration
+ When solving each individual block,
    - warmstart
    - avoid reallocating S1_new
    - allocate vector of models
    - use loose convergence criteria
+ For singleton groups, don't use JuMP and directly update
+ Currently all objective values are computed based on SDP case. 
    Need to display objective values for ME/MVR objective
"""
function solve_group_block_update(
    Σ::AbstractMatrix{T}, 
    groups::Vector{Int},
    method::Union{Symbol, String};
    ϵ::T = 1e-8, # small constant added to the matrix inverse in the constraint to enforce full rank
    m::Number = 1,
    tol=0.01, # converges when changes in s are all smaller than tol
    niter = 100, # max number of cyclic block updates
    verbose::Bool = false,
    ) where T
    method ∈ [:sdp_block, :maxent_block, :mvr_block] ||
        error("Expected method to be :sdp_block, :maxent_block, or :mvr_block")
    p = size(Σ, 1)
    unique_groups = unique(groups)
    blocks = length(unique_groups)
    group_sizes = [count(x -> x == g, groups) for g in unique_groups]
    perm = collect(1:p)
    # initialize S/A/D matrices
    S, _, _ = initialize_S(Σ, groups, m, method)
    A = (m+1)/m * Σ
    D = A - S
    # compute initial objective value
    obj = group_block_objective(Σ, S, groups, m, method)
    verbose && println("Init obj = $obj, with $blocks unique blocks to optimze")
    # begin block updates
    for l in 1:niter
        offset = 0
        max_delta = zero(eltype(Σ))
        for b in 1:blocks
            g = group_sizes[b]
            # permute current block into upper left corner
            cur_idx = offset + 1:offset + g
            @inbounds @simd for i in 1:offset
                perm[g+i] = i
            end
            perm[1:g] .= cur_idx
            S      .= @view(S[perm, perm])
            A.data .= @view(A.data[perm, perm])
            D      .= @view(D[perm, perm])
            Σ.data .= @view(Σ.data[perm, perm])
            # update constraints
            S11 = @view(S[1:g, 1:g])
            Σ11 = @view(Σ[1:g, 1:g])
            A11 = @view(A[1:g, 1:g])
            D12 = @view(D[1:g, g + 1:end])
            D21 = @view(D[g + 1:end, 1:g])
            D22 = @view(D[g + 1:end, g + 1:end])
            D22inv = inv(D22 + ϵ*I)
            ub = Symmetric(A11 - D12 * D22inv * D21)
            # solve SDP/MVR/ME problem for current block
            if method == :sdp_block
                S11_new, opt_success = solve_group_SDP_single_block(Σ11, ub)
            elseif method == :maxent_block
                S11_new, opt_success = solve_group_maxent_single_block(Σ11, ub, m)
            elseif method == :mvr_block
                S11_new, opt_success = solve_group_MVR_single_block(
                    Σ11, ub, D21, D22inv, m)
            end
            # only update if optimization was successful
            if opt_success
                # find max difference between previous block S
                for i in eachindex(S11_new)
                    if abs(S11_new[i] - S11[i]) > max_delta
                        max_delta = abs(S11_new[i] - S11[i])
                    end
                end
                # update relevant blocks
                S11 .= S11_new
                D[1:g, 1:g] .= A11 .- S11_new
            end
            # repermute columns/rows of S back
            iperm = invperm(perm)
            S      .= @view(S[iperm, iperm])
            A.data .= @view(A.data[iperm, iperm])
            D      .= @view(D[iperm, iperm])
            Σ.data .= @view(Σ.data[iperm, iperm])
            sort!(perm)
            offset += g
        end
        if verbose
            obj = group_block_objective(Σ, S, groups, m, method)
            println("Iter $l: obj = $obj, δ = $max_delta")
            flush(stdout)
        end
        max_delta < tol && break 
    end
    return S, T[], obj
end

function group_sdp_objective_single_block(Σg::AbstractMatrix{T}, Sg::AbstractMatrix{T}) where T
    p = size(Σg, 1)
    size(Σg) == size(Sg) || error("group_sdp_objective_single_block: Expected size of Σg and Sg to be equal")
    obj = zero(T)
    for j in 1:p, i in 1:p
        obj += abs(Σg[i, j] - Sg[i, j])
    end
    return obj
end

# this code solves every variable in S simultaneously, i.e. not fixing any block 
function solve_group_SDP_full(
    Σ::AbstractMatrix, 
    groups::Vector{Int}; 
    m::Number = 1,
    optm=Hypatia.Optimizer(verbose=false), # Any solver compatible with JuMP
    )
    model = Model(() -> optm)
    T = eltype(Σ)
    p = size(Σ, 1)
    group_sizes = [count(x -> x == g, groups) for g in unique(groups)]
    # in full SDP, every non-zero entry in S (group-block diagonal matrix) can vary
    @variable(model, S[1:p, 1:p], Symmetric)
    # fix everything
    for j in 1:p, i in j:p
        fix(S[i, j], zero(T))
    end
    # free pertinent variables
    idx = 0
    for g in group_sizes
        for j in 1:g, i in j:g
            unfix(S[i + idx, j + idx])
            set_lower_bound(S[i + idx, j + idx], zero(T))
            set_upper_bound(S[i + idx, j + idx], one(T))
        end
        idx += g
    end
    @constraint(model, Symmetric((m+1)/m * Σ - S) in PSDCone())
    @constraint(model, Symmetric(S) in PSDCone())
    # slack variables to handle absolute value in obj 
    @variable(model, U[1:p, 1:p], Symmetric)
    for i in 1:p, j in i:p
        @constraint(model, Σ[i, j] - S[i, j] ≤ U[i, j])
        @constraint(model, -U[i, j] ≤ Σ[i, j] - S[i, j])
    end
    @objective(model, Min, sum(U)) # equivalent to @objective(model, Min, sum(abs.(Σ - S)))
    JuMP.optimize!(model)
    check_model_solution(model)
    obj = group_block_objective(Σ, S, groups, m, method)
    return JuMP.value.(S), T[], obj
end

"""
    solve_group_max_entropy_hybrid(Σ, groups, [outer_iter=100], [inner_pca_iter=1],
        [inner_ccd_iter=1], [tol=0.0001], [ϵ=1e-6], [m=1], [robust=false], [verbose=false])

Solves the group-knockoff optimization problem based on Maximum Entropy objective.
Users should call `solve_s_group` instead of this function. 

# Inputs
+ `Σ`: Correlation matrix
+ `groups`: Group membership vector 

# Optional inputs
+ `outer_iter`: Maximum number of outer iterations. Each outer iteration will
    perform `inner_pca_iter` PCA updates `inner_ccd_iter` full optimization 
    updates (default = 100).
+ `inner_pca_iter`: Number of full PCA updates before changing to fully
    general coordinate descent updates (default = 1)
+ `inner_ccd_iter`: Number of full general coordinate descent updates before changing
    to PCA updates (default = 1)
+ `tol`: convergence tolerance. Algorithm converges when 
    `abs((obj_new-obj_old)/obj_old) < tol` OR when changes in `S` matrix falls 
    below 1e-4
+ `ϵ`: tolerance added to the lower and upper bound, prevents numerical issues
    (default = `1e-6`)
+ `m`: Number of knockoffs per variable (defaults `1`)
+ `robust`: whether to use "robust" Cholesky updates. If `robust=true`, alg will
    be ~10x slower, only use this if `robust=false` causes cholesky updates to fail.
    (default `false`)
+ `verbose`: Whether to print intermediate results (default `false`)
"""
function solve_group_max_entropy_hybrid(
    Σ::AbstractMatrix{T}, 
    groups::Vector{Int};
    outer_iter::Int = 100,
    inner_pca_iter::Int = 1,
    inner_ccd_iter::Int = 1,
    tol=0.0001, # converges when abs((obj_new-obj_old)/obj_old) fall below tol
    ϵ=1e-6, # tolerance added to the lower and upper bound, prevents numerical issues
    m::Number = 1, # number of knockoffs per variable
    robust::Bool = false, # whether to use "robust" Cholesky updates (if robust=true, CCD alg will be ~10x slower, only use this if the default causes cholesky updates to fail)
    verbose::Bool = false
    ) where T
    p = size(Σ, 1)
    group_sizes = [count(x -> x == g, groups) for g in unique(groups)]
    # whether to use robust cholesky updates or not
    cholupdate! = robust ? lowrankupdate! : lowrankupdate_turbo!
    choldowndate! = robust ? lowrankdowndate! : lowrankdowndate_turbo!
    # initialize S matrix, initial cholesky factors, and constants
    S, L, C = initialize_S(Σ, groups, m, :maxent)
    obj = group_maxent_obj(L, C, m)
    verbose && println("Maxent initial obj = $obj")
    # compute vectors for PCA updates
    V = get_PCA_vectors(Σ, groups)
    # some timers
    t1 = zero(T) # time for updating cholesky factors
    t2 = zero(T) # time for forward/backward solving
    t3 = zero(T) # time for solving offdiag 1D optimization problems
    # preallocated vectors for efficiency
    u, w, ei, ej = zeros(p), zeros(p), zeros(p), zeros(p)
    iter = 1
    for i in 1:outer_iter
        # PCA iterations
        converged1, obj, t1, t2, t3, iter = _maxent_pca_ccd_iter!(
            S, L, C, V, 
            obj, m, inner_pca_iter, tol, ϵ, t1, t2, t3, iter, 
            cholupdate!, choldowndate!,
            u, w; verbose=verbose
        )
        # Full CCD iterations
        converged2, obj, t1, t2, t3, iter = _maxent_ccd_iter!(
            S, L, C, 
            obj, m, group_sizes, inner_ccd_iter, tol, ϵ, t1, t2, t3, iter, 
            cholupdate!, choldowndate!,
            u, w, ei, ej; verbose=verbose
        )
        # check convergence
        converged1 && converged2 && break
    end
    return S, T[], obj, L, C
end

"""
    solve_group_sdp_hybrid(Σ, groups, [outer_iter=100], [inner_pca_iter=1],
        [inner_ccd_iter=1], [tol=0.0001], [ϵ=1e-6], [m=1], [robust=false], [verbose=false])

Solves the group-knockoff optimization problem based on SDP objective.
Users should call `solve_s_group` instead of this function. 

# Inputs
+ `Σ`: Correlation matrix
+ `groups`: Group membership vector 

# Optional inputs
+ `outer_iter`: Maximum number of outer iterations. Each outer iteration will
    perform `inner_pca_iter` PCA updates `inner_ccd_iter` full optimization 
    updates (default = 100).
+ `inner_pca_iter`: Number of full PCA updates before changing to fully
    general coordinate descent updates (default = 1)
+ `inner_ccd_iter`: Number of full general coordinate descent updates before changing
    to PCA updates (default = 1)
+ `tol`: convergence tolerance. Algorithm converges when 
    `abs((obj_new-obj_old)/obj_old) < tol` OR when changes in `S` matrix falls 
    below 1e-4
+ `ϵ`: tolerance added to the lower and upper bound, prevents numerical issues
    (default = `1e-6`)
+ `m`: Number of knockoffs per variable (defaults `1`)
+ `robust`: whether to use "robust" Cholesky updates. If `robust=true`, alg will
    be ~10x slower, only use this if `robust=false` causes cholesky updates to fail.
    (default `false`)
+ `verbose`: Whether to print intermediate results (default `false`)
"""
function solve_group_sdp_hybrid(
    Σ::AbstractMatrix{T}, 
    groups::Vector{Int};
    outer_iter::Int = 100,
    inner_pca_iter::Int = 1,
    inner_ccd_iter::Int = 1,
    tol=0.0001, # converges when abs((obj_new-obj_old)/obj_old) fall below tol
    ϵ=1e-6, # tolerance added to the lower and upper bound, prevents numerical issues
    m::Number = 1, # number of knockoffs per variable
    robust::Bool = false, # whether to use "robust" Cholesky updates (if robust=true, CCD alg will be ~10x slower, only use this if the default causes cholesky updates to fail)
    verbose::Bool = false
    ) where T
    p = size(Σ, 1)
    group_sizes = [count(x -> x == g, groups) for g in unique(groups)]
    # whether to use robust cholesky updates or not
    cholupdate! = robust ? lowrankupdate! : lowrankupdate_turbo!
    choldowndate! = robust ? lowrankdowndate! : lowrankdowndate_turbo!
    # compute vectors for PCA updates
    V = get_PCA_vectors(Σ, groups)
    # initialize S matrix and initial cholesky factors
    S, L, C = initialize_S(Σ, groups, m, :sdp)
    # intial objective for each group
    group_objectives, group_idx = T[], Vector{Int}[]
    for g in unique(groups)
        idx = findall(x -> x == g, groups)
        obj_g = _sdp_block_objective(@view(Σ[idx, idx]), @view(S[idx, idx]))
        push!(group_objectives, obj_g)
        push!(group_idx, idx)
    end
    obj = sum(group_objectives)
    verbose && println("SDP initial obj = $obj")
    if obj < ϵ
        return S, T[], obj, L, C # quick return
    end
    # for each v, find which group v updates
    v_groups = Int[]
    for v in eachcol(V)
        nz_idx = findfirst(!iszero, v) |> something
        g = findfirst(x -> nz_idx in x, group_idx) |> something
        push!(v_groups, g)
    end
    # some timers
    t1 = zero(T) # time for updating cholesky factors
    t2 = zero(T) # time for forward/backward solving
    t3 = zero(T) # time for solving offdiag 1D optimization problems
    # preallocated vectors for efficiency
    u, w, ei, ej = zeros(p), zeros(p), zeros(p), zeros(p)
    iter = 1
    for i in 1:outer_iter
        # PCA iterations
        converged1, obj, t1, t2, t3, iter = _sdp_pca_ccd_iter!(
            S, L, C, V, Σ,
            obj, inner_pca_iter, tol, ϵ, t1, t2, t3, iter, 
            group_idx, v_groups, group_objectives,
            cholupdate!, choldowndate!,
            u, w, groups, m, verbose=verbose
        )
        # Full CCD iterations
        converged2, obj, t1, t2, t3, iter = _sdp_ccd_iter!(
            S, L, C, Σ, groups,
            obj, m, group_sizes, inner_ccd_iter, tol, ϵ, t1, t2, t3, iter, 
            cholupdate!, choldowndate!,
            u, w, ei, ej, verbose=verbose
        )
        if inner_pca_iter > 0 # update block objectives
            for (g, idx) in enumerate(group_idx)
                group_objectives[g] = 
                    _sdp_block_objective(@view(Σ[idx, idx]), @view(S[idx, idx]))
            end
        end
        # check convergence
        converged1 && converged2 && break
    end
    return S, T[], obj, L, C
end

"""
    solve_group_mvr_hybrid(Σ, groups, [outer_iter=100], [inner_pca_iter=1],
        [inner_ccd_iter=1], [tol=0.0001], [ϵ=1e-6], [m=1], [robust=false], [verbose=false])

Solves the group-knockoff optimization problem based on MVR objective.
Users should call `solve_s_group` instead of this function. 

# Inputs
+ `Σ`: Correlation matrix
+ `groups`: Group membership vector 

# Optional inputs
+ `outer_iter`: Maximum number of outer iterations. Each outer iteration will
    perform `inner_pca_iter` PCA updates `inner_ccd_iter` full optimization 
    updates (default = 100).
+ `inner_pca_iter`: Number of full PCA updates before changing to fully
    general coordinate descent updates (default = 1)
+ `inner_ccd_iter`: Number of full general coordinate descent updates before changing
    to PCA updates (default = 1)
+ `tol`: convergence tolerance. Algorithm converges when 
    `abs((obj_new-obj_old)/obj_old) < tol` OR when changes in `S` matrix falls 
    below 1e-4
+ `ϵ`: tolerance added to the lower and upper bound, prevents numerical issues
    (default = `1e-6`)
+ `m`: Number of knockoffs per variable (defaults `1`)
+ `robust`: whether to use "robust" Cholesky updates. If `robust=true`, alg will
    be ~10x slower, only use this if `robust=false` causes cholesky updates to fail.
    (default `false`)
+ `verbose`: Whether to print intermediate results (default `false`)
"""
function solve_group_mvr_hybrid(
    Σ::AbstractMatrix{T}, 
    groups::Vector{Int};
    outer_iter::Int = 100,
    inner_pca_iter::Int = 1,
    inner_ccd_iter::Int = 1,
    tol=0.0001, # converges when abs((obj_new-obj_old)/obj_old) fall below tol
    ϵ=1e-6, # tolerance added to the lower and upper bound, prevents numerical issues
    m::Number = 1, # number of knockoffs per variable
    robust::Bool = false, # whether to use "robust" Cholesky updates (if robust=true, CCD alg will be ~10x slower, only use this if the default causes cholesky updates to fail)
    verbose::Bool = false
    ) where T
    p = size(Σ, 1)
    group_sizes = [count(x -> x == g, groups) for g in unique(groups)]
    # whether to use robust cholesky updates or not
    cholupdate! = robust ? lowrankupdate! : lowrankupdate_turbo!
    choldowndate! = robust ? lowrankdowndate! : lowrankdowndate_turbo!
    # compute vectors for PCA updates
    V = get_PCA_vectors(Σ, groups)
    # initialize S matrix and initial cholesky factors
    S, L, C = initialize_S(Σ, groups, m, :mvr)
    obj = group_block_objective(Σ, S, groups, m, :mvr)
    verbose && println("MVR initial obj = $obj")
    # some timers
    t1 = zero(T) # time for updating cholesky factors
    t2 = zero(T) # time for forward/backward solving
    t3 = zero(T) # time for solving offdiag 1D optimization problems
    # preallocated vectors for efficiency
    u, w, ei, ej, storage = zeros(p), zeros(p), zeros(p), zeros(p), zeros(p)
    iter = 1
    for i in 1:outer_iter
        # PCA iterations
        converged1, obj, t1, t2, t3, iter = _mvr_pca_ccd_iter!(
            S, L, C, V, Σ, 
            obj, m, inner_pca_iter, tol, ϵ, t1, t2, t3, iter, 
            cholupdate!, choldowndate!,
            u, w, storage, verbose=verbose
        )
        # Full CCD iterations
        converged2, obj, t1, t2, t3, iter = _mvr_ccd_iter!(
            S, L, C, Σ,
            obj, m, group_sizes, inner_ccd_iter, tol, ϵ, t1, t2, t3, iter, 
            cholupdate!, choldowndate!,
            u, w, ei, ej, storage, verbose=verbose
        )
        # check convergence
        converged1 && converged2 && break
    end
    return S, T[], obj, L, C
end

function _sdp_ccd_iter!(
    S, L, C, Σ, groups, # main matrix variables
    obj, m, group_sizes, niter, tol, ϵ, t1, t2, t3, print_iter, # constants
    cholupdate!, choldowndate!, # cholesky update functions
    u, v, ei, ej; verbose=false # storages
    )
    T = eltype(S)
    blocks = length(group_sizes)
    converged = niter == 0 ? true : false
    for l in 1:niter
        max_delta = zero(T)
        obj_new = obj
        offset = 0
        for b in 1:blocks
            group_size = group_sizes[b]
            #
            # optimize diagonal entries
            #
            for idx in 1:group_sizes[b]
                j = idx + offset
                # compute feasible region
                fill!(ej, 0)
                ej[j] = 1
                t2 += @elapsed ldiv!(u, UpperTriangular(L.factors)', ej)
                t2 += @elapsed ldiv!(v, UpperTriangular(C.factors)', ej)
                ub = 1 / sum(abs2, u) - ϵ
                lb = -1 / sum(abs2, v) + ϵ
                lb ≥ ub && continue
                # compute new δ, making sure it is in feasible region
                δj = clamp(Σ[j, j] - S[j, j], lb, ub)
                change_obj = (abs(Σ[j, j]-S[j, j]-δj) - abs(Σ[j, j]-S[j, j])) / group_size^2
                if abs(δj) < 1e-15 || isnan(δj) || isinf(δj) || change_obj > 0.01
                    continue
                end
                # update S
                S[j, j] += δj
                obj_new += change_obj
                # rank 1 update to cholesky factors
                t1 += @elapsed rank1_cholesky_update!(
                    L, C, j, δj, ej, u, choldowndate!, cholupdate!
                )
                # update convergence tol
                abs(δj) > max_delta && (max_delta = abs(δj))
            end
            #
            # optimize off-diagonal entries
            #
            for idx1 in 1:group_sizes[b], idx2 in idx1+1:group_sizes[b]
                i, j = idx2 + offset, idx1 + offset
                fill!(ej, 0); fill!(ei, 0)
                ej[j], ei[i] = 1, 1
                # compute aii, ajj, aij, bii, bjj, bij
                t2 += @elapsed begin
                    ldiv!(u, UpperTriangular(L.factors)', ei)
                    ldiv!(v, UpperTriangular(L.factors)', ej)
                    aij, aii, ajj = dot(u, v), dot(u, u), dot(v, v)
                    ldiv!(u, UpperTriangular(C.factors)', ei)
                    ldiv!(v, UpperTriangular(C.factors)', ej)
                    bij, bii, bjj = dot(u, v), dot(u, u), dot(v, v)
                end
                # compute (mathematical) feasible region
                s1 = (aij - sqrt(aii*ajj)) / (aij^2 - aii * ajj)
                s2 = (aij + sqrt(aii*ajj)) / (aij^2 - aii * ajj)
                d1 = (-bij - sqrt(bii*bjj)) / (bij^2 - bii * bjj)
                d2 = (-bij + sqrt(bii*bjj)) / (bij^2 - bii * bjj)
                s1 > s2 && ((s1, s2) = (s2, s1))
                d1 > d2 && ((d1, d2) = (d2, d1))
                # feasible region criteria due to computational reasons
                lb = max(s1, d1, -2 / (bii + 2bij + bjj)) + ϵ
                ub = min(s2, d2, 2 / (aii + 2aij + ajj)) - ϵ
                lb ≥ ub && continue
                # find δ ∈ [lb, ub] that maximizes objective
                δ = clamp(Σ[i, j] - S[i, j], lb, ub)
                change_obj = (2*abs(Σ[i, j]-S[i, j]-δ) - 2*abs(Σ[i, j]-S[i, j])) / group_size^2
                if abs(δ) < 1e-15 || isnan(δ) || isinf(δ) || change_obj > 0.01
                    continue
                end
                # update S
                S[i, j] += δ
                S[j, i] += δ
                obj_new += change_obj
                # rank 2 update to cholesky factors
                t1 += @elapsed rank2_cholesky_update!(
                    L, C, i, j, δ, u, v, choldowndate!, cholupdate!
                )
                # update convergence tol
                abs(δ) > max_delta && (max_delta = abs(δ))
            end
            offset += group_sizes[b]
        end
        if verbose
            # obj_true = group_block_objective(Σ, S, groups, m, :sdp)
            # @show obj_true
            println("Iter $print_iter (CCD): obj = $obj_new, δ = $max_delta, " * 
                "t1 = $(round(t1, digits=2)), t2 = $(round(t2, digits=2)), " * 
                "t3 = $(round(t3, digits=2))")
            print_iter += 1
            flush(stdout)
        end
        change_obj = abs((obj_new - obj) / obj)
        obj = obj_new
        if change_obj < tol || max_delta < 1e-4
            converged = true
            break 
        end
    end
    return converged, obj, t1, t2, t3, print_iter
end

function _mvr_ccd_iter!(
    S, L, C, Σ, # main matrix variables
    obj, m, group_sizes, niter, tol, ϵ, t1, t2, t3, print_iter, # constants
    cholupdate!, choldowndate!, # cholesky update functions
    u, v, ei, ej, storage; verbose=false # storages
    )
    T = eltype(S)
    blocks = length(group_sizes)
    converged = niter == 0 ? true : false
    for l in 1:niter
        max_delta = zero(T)
        obj_new = obj
        offset = 0
        for b in 1:blocks
            #
            # optimize diagonal entries. Note: cannot reuse code from ungrouped
            # knockoff case because S is no longer diagonal, need new alg
            #
            for idx in 1:group_sizes[b]
                j = idx + offset
                fill!(ej, 0)
                ej[j] = 1
                # compute ajj, bjj, cjj, djj which defines the feasible region
                t2 += @elapsed begin
                    ldiv!(v, UpperTriangular(L.factors)', ej)
                    ldiv!(u, UpperTriangular(C.factors)', ej)
                    ajj, bjj = dot(v, v), dot(u, u)
                    forward_backward!(v, C, ej, storage)
                    forward_backward!(u, L, ej, storage)
                    cjj, djj = dot(v, v), dot(u, u)
                end
                # compute δ that is within feasible region
                ub = 1 / ajj - ϵ
                lb = -1 / bjj + ϵ
                lb ≥ ub && continue
                x1, x2 = diag_mvr_obj_root(m, ajj, bjj, cjj, djj)
                δ = lb < x1 < ub ? x1 : lb < x2 < ub ? x2 : NaN
                # update S if objective improves
                change_obj = -m^2*δ*cjj/(1+δ*bjj) + δ*djj/(1-δ*ajj)
                if change_obj > 0 || abs(δ) < 1e-15 || isnan(δ) || isinf(δ)
                    continue
                end
                S[j, j] += δ
                obj_new += change_obj
                # rank 1 update to cholesky factors
                t1 += @elapsed rank1_cholesky_update!(
                    L, C, j, δ, ej, u, choldowndate!, cholupdate!
                )
                # update convergence tol
                abs(δ) > max_delta && (max_delta = abs(δ))
            end
            #
            # optimize off-diagonal entries
            #
            for idx1 in 1:group_sizes[b], idx2 in idx1+1:group_sizes[b]
                i, j = idx2 + offset, idx1 + offset
                fill!(ej, 0); fill!(ei, 0)
                ej[j], ei[i] = 1, 1
                # compute aii, ajj, aij, bii, bjj, bij
                t2 += @elapsed begin
                    ldiv!(u, UpperTriangular(L.factors)', ei)
                    ldiv!(v, UpperTriangular(L.factors)', ej)
                    aij, aii, ajj = dot(u, v), dot(u, u), dot(v, v)
                    ldiv!(u, UpperTriangular(C.factors)', ei)
                    ldiv!(v, UpperTriangular(C.factors)', ej)
                    bij, bii, bjj = dot(u, v), dot(u, u), dot(v, v)
                    # compute cii, cjj, cij, dii, djj, dij
                    forward_backward!(u, C, ei, storage)
                    forward_backward!(v, C, ej, storage)
                    cij, cii, cjj = dot(u, v), dot(u, u), dot(v, v)
                    forward_backward!(u, L, ei, storage)
                    forward_backward!(v, L, ej, storage)
                    dij, dii, djj = dot(u, v), dot(u, u), dot(v, v)
                end
                # compute (mathematical) feasible region
                s1 = (aij - sqrt(aii*ajj)) / (aij^2 - aii * ajj)
                s2 = (aij + sqrt(aii*ajj)) / (aij^2 - aii * ajj)
                d1 = (-bij - sqrt(bii*bjj)) / (bij^2 - bii * bjj)
                d2 = (-bij + sqrt(bii*bjj)) / (bij^2 - bii * bjj)
                s1 > s2 && ((s1, s2) = (s2, s1))
                d1 > d2 && ((d1, d2) = (d2, d1))
                # feasible region criteria due to computational reasons
                lb = max(s1, d1, -2 / (bii + 2bij + bjj) + ϵ)
                ub = min(s2, d2, 2 / (aii + 2aij + ajj) - ϵ)
                lb ≥ ub && continue
                # find δ ∈ [lb, ub] that maximizes objective
                t3 += @elapsed opt = optimize(
                    δ -> offdiag_mvr_obj(
                        δ, m, aij, aii, ajj, bij, bii, bjj,
                              cij, cii, cjj, dij, dii, djj,
                    ),
                    lb, ub, Brent(), show_trace=false, abs_tol=0.0001
                )
                δ = clamp(opt.minimizer, lb, ub)
                change_obj = opt.minimum
                if change_obj > 0 || abs(δ) < 1e-15 || isnan(δ) || isinf(δ)
                    continue
                end
                # update S
                obj_new += change_obj
                S[i, j] += δ
                S[j, i] += δ
                # update cholesky factors
                t1 += @elapsed rank2_cholesky_update!(
                    L, C, i, j, δ, u, v, choldowndate!, cholupdate!
                )
                # update convergence tol
                abs(δ) > max_delta && (max_delta = abs(δ))
            end
            offset += group_sizes[b]
        end
        if verbose
            println("Iter $print_iter (CCD): obj = $obj_new, δ = $max_delta, " * 
                "t1 = $(round(t1, digits=2)), t2 = $(round(t2, digits=2))," * 
                "t3 = $(round(t3, digits=2))")
            print_iter += 1
            flush(stdout)
        end
        change_obj = abs((obj_new - obj) / obj)
        obj = obj_new
        if change_obj < tol || max_delta < 1e-4
            converged = true
            break 
        end
    end
    return converged, obj, t1, t2, t3, print_iter
end

function _maxent_ccd_iter!(
    S, L, C, # main matrix variables
    obj, m, group_sizes, niter, tol, ϵ, t1, t2, t3, print_iter,  # constants
    cholupdate!, choldowndate!, # cholesky update functions
    u, v, ei, ej; verbose = false # storages
    )
    T = eltype(S)
    blocks = length(group_sizes)
    converged = niter == 0 ? true : false
    for l in 1:niter
        max_delta = zero(T)
        obj_new = obj
        offset = 0
        for b in 1:blocks
            #
            # optimize diagonal entries. Note: cannot reuse code from ungrouped
            # knockoff case because S is no longer diagonal, need new alg
            #
            for idx in 1:group_sizes[b]
                j = idx + offset
                # compute new S[j, j]
                fill!(ej, 0)
                ej[j] = 1
                t2 += @elapsed ldiv!(u, UpperTriangular(L.factors)', ej)
                t2 += @elapsed ldiv!(v, UpperTriangular(C.factors)', ej)
                ajj, bjj = dot(u, u), dot(v, v)
                sj_new = (m*bjj-ajj) / ((m+1)*ajj*bjj)
                # ensure feasibility
                ub = 1 / ajj - ϵ
                lb = -1 / bjj + ϵ
                lb ≥ ub && continue
                δ = clamp(sj_new - S[j, j], lb, ub)
                # update S if objective improves
                change_obj = log(1 - δ*ajj) + m*log(1 + δ*bjj)
                if change_obj < 0 || abs(δ) < 1e-15 || isnan(δ) || isinf(δ)
                    continue
                end
                S[j, j] += δ
                obj_new += change_obj
                # rank 1 update to cholesky factors
                t1 += @elapsed rank1_cholesky_update!(
                    L, C, j, δ, ej, u, choldowndate!, cholupdate!
                )
                # update convergence tol
                abs(δ) > max_delta && (max_delta = abs(δ))
            end
            #
            # optimize off-diagonal entries
            #
            for idx1 in 1:group_sizes[b], idx2 in idx1+1:group_sizes[b]
                i, j = idx2 + offset, idx1 + offset
                fill!(ej, 0); fill!(ei, 0)
                ej[j], ei[i] = 1, 1
                # compute aii, ajj, aij, bii, bjj, bij
                t2 += @elapsed begin
                    ldiv!(u, UpperTriangular(L.factors)', ei)
                    ldiv!(v, UpperTriangular(L.factors)', ej)
                    aij, aii, ajj = dot(u, v), dot(u, u), dot(v, v)
                    ldiv!(u, UpperTriangular(C.factors)', ei)
                    ldiv!(v, UpperTriangular(C.factors)', ej)
                    bij, bii, bjj = dot(u, v), dot(u, u), dot(v, v)
                end
                # compute (mathematical) feasible region
                s1 = (aij - sqrt(aii*ajj)) / (aij^2 - aii * ajj)
                s2 = (aij + sqrt(aii*ajj)) / (aij^2 - aii * ajj)
                d1 = (-bij - sqrt(bii*bjj)) / (bij^2 - bii * bjj)
                d2 = (-bij + sqrt(bii*bjj)) / (bij^2 - bii * bjj)
                s1 > s2 && ((s1, s2) = (s2, s1))
                d1 > d2 && ((d1, d2) = (d2, d1))
                # feasible region criteria due to computational reasons
                lb = max(s1, d1, -2 / (bii + 2bij + bjj) + ϵ)
                ub = min(s2, d2, 2 / (aii + 2aij + ajj) - ϵ)
                lb ≥ ub && continue
                # find δ ∈ [lb, ub] that maximizes objective
                t3 += @elapsed opt = optimize(
                    δ -> offdiag_maxent_obj(δ, m, aij, aii, ajj, bij, bii, bjj),
                    lb, ub, Brent(), show_trace=false, abs_tol=0.0001
                )
                δ = clamp(opt.minimizer, lb, ub)
                change_obj = -opt.minimum
                if change_obj < 0 || abs(δ) < 1e-15 || isnan(δ) || isinf(δ)
                    continue
                end
                obj_new += change_obj
                # update S
                S[i, j] += δ
                S[j, i] += δ
                # update cholesky factors
                t1 += @elapsed rank2_cholesky_update!(
                    L, C, i, j, δ, u, v, choldowndate!, cholupdate!
                )
                # update convergence tol
                abs(δ) > max_delta && (max_delta = abs(δ))
            end
            offset += group_sizes[b]
        end
        if verbose
            # true_obj = group_maxent_obj(L, C, m)
            # @show true_obj
            println("Iter $print_iter (CCD): obj = $obj_new, δ = $max_delta, t1 = " * 
                "$(round(t1, digits=2)), t2 = $(round(t2, digits=2)), " * 
                "t3 = $(round(t3, digits=2))")
            print_iter += 1
            flush(stdout)
        end
        change_obj = abs((obj_new - obj) / obj)
        obj = obj_new
        if change_obj < tol || max_delta < 1e-4
            converged = true
            break 
        end
    end
    return converged, obj, t1, t2, t3, print_iter
end

function _maxent_pca_ccd_iter!(
    S, L, C, evecs, # main matrix variables
    obj, m, niter, tol, ϵ, t1, t2, t3, print_iter, # constants
    cholupdate!, choldowndate!, # cholesky update functions 
    u, w; verbose=false # storages
    )
    T = eltype(S)
    converged = niter == 0 ? true : false
    for l in 1:niter
        max_delta = zero(T)
        obj_new = obj
        for v in eachcol(evecs)
            # get necessary constants
            t2 += @elapsed begin
                ldiv!(w, UpperTriangular(L.factors)', v)
                ldiv!(u, UpperTriangular(C.factors)', v)
                vt_Sinv_v = dot(u, u)
                vt_Dinv_v = dot(w, w)
            end
            # compute δ ∈ [lb, ub]
            lb = -1 / vt_Sinv_v + ϵ
            ub = 1 / vt_Dinv_v - ϵ
            lb ≥ ub && continue
            δ = (m*vt_Sinv_v - vt_Dinv_v) / ((m+1)*vt_Sinv_v*vt_Dinv_v)
            δ = clamp(δ, lb, ub)
            # compute new objective
            change_obj = log(1 - δ*vt_Dinv_v) + m*log(1 + δ*vt_Sinv_v)
            if change_obj < 0 || abs(δ) < 1e-15 || isnan(δ) || isinf(δ)
                continue
            end
            # update S_new = S + δ*v*v'
            t1 += @elapsed BLAS.ger!(δ, v, v, S)
            obj_new += change_obj
            # update cholesky factors
            u .= sqrt(abs(δ)) .* v
            w .= sqrt(abs(δ)) .* v
            t1 += @elapsed begin
                if δ > 0
                    choldowndate!(L, u)
                    cholupdate!(C, w)
                else
                    cholupdate!(L, u)
                    choldowndate!(C, w)
                end
            end
            # track convergence
            abs(δ) > max_delta && (max_delta = abs(δ))
        end
        if verbose
            println("Iter $(print_iter) (PCA): obj = $obj_new, δ = $max_delta, t1 = " * 
                "$(round(t1, digits=2)), t2 = $(round(t2, digits=2))")
            print_iter += 1
            flush(stdout)
        end
        # check convergence
        change_obj = abs((obj_new - obj) / obj)
        obj = obj_new
        if change_obj < tol || max_delta < 1e-4
            converged = true
            break 
        end
    end
    return converged, obj, t1, t2, t3, print_iter
end

function _mvr_pca_ccd_iter!(
    S, L, C, evecs, Σ, # main matrix variables
    obj, m, niter, tol, ϵ, t1, t2, t3, print_iter, # constants
    cholupdate!, choldowndate!, # cholesky update functions 
    u, w, storage; verbose=false # storages
    )
    T = eltype(S)
    converged = niter == 0 ? true : false
    for l in 1:niter
        max_delta = zero(T)
        obj_new = obj
        for v in eachcol(evecs)
            # get necessary constants
            t2 += @elapsed begin
                ldiv!(w, UpperTriangular(L.factors)', v)
                ldiv!(u, UpperTriangular(C.factors)', v)
                vt_Sinv_v = dot(u, u) # bjj
                vt_Dinv_v = dot(w, w) # ajj
                forward_backward!(w, L, v, storage)
                forward_backward!(u, C, v, storage)
                vt_Dinv2_v = dot(w, w) # djj
                vt_Sinv2_v = dot(u, u) # cjj
            end
            # compute δ that is within feasible region
            lb = -1 / vt_Sinv_v + ϵ
            ub = 1 / vt_Dinv_v - ϵ
            lb ≥ ub && continue
            x1, x2 = diag_mvr_obj_root(m, vt_Dinv_v, vt_Sinv_v, 
                vt_Sinv2_v, vt_Dinv2_v)
            δ = lb < x1 < ub ? x1 : lb < x2 < ub ? x2 : NaN
            # update S_new = S + δ*v*v' if objective improves
            change_obj = -m^2*δ*vt_Sinv2_v/(1+δ*vt_Sinv_v) + 
                δ*vt_Dinv2_v/(1-δ*vt_Dinv_v)
            if change_obj > 0 || abs(δ) < 1e-15 || isnan(δ) || isinf(δ)
                continue
            end
            obj_new += change_obj
            t1 += @elapsed BLAS.ger!(δ, v, v, S)
            # update cholesky factors
            u .= sqrt(abs(δ)) .* v
            w .= sqrt(abs(δ)) .* v
            t1 += @elapsed begin
                if δ > 0
                    choldowndate!(L, u)
                    cholupdate!(C, w)
                else
                    cholupdate!(L, u)
                    choldowndate!(C, w)
                end
            end
            # track convergence
            abs(δ) > max_delta && (max_delta = abs(δ))
        end
        if verbose
            println("Iter $print_iter (PCA): obj = $obj_new, δ = $max_delta, t1 = " * 
                "$(round(t1, digits=2)), t2 = $(round(t2, digits=2))")
            print_iter += 1
            flush(stdout)
        end
        # check convergence
        change_obj = abs((obj_new - obj) / obj)
        obj = obj_new
        if change_obj < tol || max_delta < 1e-4
            converged = true
            break
        end
    end
    return converged, obj, t1, t2, t3, print_iter
end

function _sdp_pca_ccd_iter!(
    S, L, C, evecs, Σ, # main matrix variables
    obj, niter, tol, ϵ, t1, t2, t3, print_iter, # constants
    group_indices, v_groups, group_objectives, # some precomputed variables
    cholupdate!, choldowndate!, # cholesky update functions 
    u, w, groups, m; verbose=false # storages
    )
    T = eltype(S)
    converged = niter == 0 ? true : false
    for l in 1:niter
        max_delta = zero(T)
        obj_new = obj
        for (j, v) in enumerate(eachcol(evecs))
            v_group = v_groups[j]
            group_idx = group_indices[v_group]
            # get necessary constants
            t2 += @elapsed begin
                ldiv!(w, UpperTriangular(L.factors)', v)
                ldiv!(u, UpperTriangular(C.factors)', v)
                vt_Sinv_v = dot(u, u)
                vt_Dinv_v = dot(w, w)
            end
            # compute feasible region
            lb = -1 / vt_Sinv_v + ϵ
            ub = 1 / vt_Dinv_v - ϵ
            lb ≥ ub && continue
            # compute δ numerically
            Σg, Sg = @view(Σ[group_idx, group_idx]), @view(S[group_idx, group_idx])
            vg = @view(v[group_idx])
            t3 += @elapsed opt = optimize(
                δ -> pca_sdp_obj(δ, Σg, Sg, vg),
                lb, ub, Brent(), show_trace=false, abs_tol=0.0001
            )
            δ = clamp(opt.minimizer, lb, ub)
            # find difference in objective (requiring objective to strictly
            # improve causes algorithm to not move much, not really sure why,
            # so I allow an update as long as objective doesn't get much worse)
            change_obj = opt.minimum - group_objectives[v_group]
            if abs(δ) < 1e-15 || isnan(δ) || isinf(δ) || change_obj > 0.01
                continue
            end
            # update S_new = S + δ*v*v'
            t1 += @elapsed BLAS.ger!(δ, v, v, S)
            obj_new += change_obj
            group_objectives[v_group] = opt.minimum
            # update cholesky factors
            u .= sqrt(abs(δ)) .* v
            w .= sqrt(abs(δ)) .* v
            t1 += @elapsed begin
                if δ > 0
                    choldowndate!(L, u)
                    cholupdate!(C, w)
                else
                    cholupdate!(L, u)
                    choldowndate!(C, w)
                end
            end
            # track convergence
            abs(δ) > max_delta && (max_delta = abs(δ))
        end
        if verbose
            # obj_true = group_block_objective(Σ, S, groups, m, :sdp)
            # @show obj_true
            println("Iter $print_iter (PCA): obj = $obj_new, δ = $max_delta, t1 = " * 
                "$(round(t1, digits=2)), t2 = $(round(t2, digits=2)), " * 
                "t3 = $(round(t3, digits=2))")
            print_iter += 1
            flush(stdout)
        end
        # check convergence
        change_obj = abs((obj_new - obj) / obj)
        obj = obj_new
        if change_obj < tol || max_delta < 1e-4
            converged = true
            break
        end
    end
    return converged, obj, t1, t2, t3, print_iter
end

# efficient and numerically stable way to evaluate max entropy objective 
# logdet((m+1)/m*Σ-S) + m*logdet(S) where
# C is cholesky factor of S and L is cholesky factor of (m+1)/m*Σ-S
function group_maxent_obj(L::Cholesky, C::Cholesky, m::Number)
    return logdet(L) + m*logdet(C)
end

function group_mvr_obj(L::Cholesky, C::Cholesky, m::Number, 
    storage::LowerTriangular{T}=LowerTriangular(zeros(size(L)))) where T
    copyto!(storage, I)
    ldiv!(C.L, storage)
    obj = m^2 * sum(abs2, storage)
    copyto!(storage, I)
    ldiv!(L.L, storage)
    obj += sum(abs2, storage)
    return obj
end

# objective functions to minimize when optimizing diagonal or offdiagnal entries
# in max entropy, MVR, or SDP group knockoffs
function offdiag_maxent_obj(δ, m, aij, aii, ajj, bij, bii, bjj)
    in1 = (1 - δ*aij)^2 - δ^2*aii*ajj
    in2 = (1 + δ*bij)^2 - δ^2*bjj*bii
    in1 ≤ 0 || in2 ≤ 0 && return typemin(δ)
    return -log(in1) - m*log(in2)
end
function offdiag_mvr_obj(δ, m, aij, aii, ajj, bij, bii, bjj, cij, cii, cjj, dij, dii, djj)
    denom1 = (1 + δ*bij)^2 - δ^2*bii*bjj
    denom2 = (1 - δ*aij)^2 - δ^2*aii*ajj
    numer1 = -m^2 * δ * ((cij*bij - cjj*bii - cii*bjj + cij*bij)*δ + 2cij)
    numer2 = δ * ((-dij*aij + djj*aii + dii*ajj - dij*aij)*δ + 2dij)
    return numer1 / denom1 + numer2 / denom2
end
function diag_mvr_obj_root(m, ajj, bjj, cjj, djj)
    a = (-ajj^2*m^2*cjj + bjj^2*djj)
    b = 2ajj*m^2*cjj + 2bjj*djj
    c = djj - m^2*cjj
    a == c == 0 && return 0, 0
    x1 = (-b + sqrt(b^2 - 4a*c)) / (2a)
    x2 = (-b - sqrt(b^2 - 4a*c)) / (2a)
    return x1, x2
end
function pca_sdp_obj(δ, Σg, Sg, v)
    g = size(Σg, 1)
    g == size(Sg, 1) == length(v) || error("Dimension mismatch!")
    obj = zero(eltype(v))
    @inbounds for j in eachindex(v)
        @simd for i in eachindex(v)
            obj += abs(Σg[i, j] - Sg[i, j] - δ*v[i]*v[j])
        end
    end
    return obj / g^2
end

function rank1_cholesky_update!(L, C, j, δ, store1, store2, 
    choldowndate!, cholupdate!)
    fill!(store1, 0); fill!(store2, 0)
    store1[j] = store2[j] = sqrt(abs(δ))
    if δ > 0
        choldowndate!(L, store1)
        cholupdate!(C, store2)
    else
        cholupdate!(L, store1)
        choldowndate!(C, store2)
    end
    return nothing
end

function rank2_cholesky_update!(
    L, C, i, j, δ, store1, store2, choldowndate!, cholupdate!)
    # update cholesky factor L
    fill!(store1, 0); fill!(store2, 0)
    store1[j] = store1[i] = store2[j] = sqrt(abs(δ/2))
    store2[i] = -sqrt(abs(δ/2))
    if δ > 0
        choldowndate!(L, store1)
        cholupdate!(L, store2)
    else 
        cholupdate!(L, store1)
        choldowndate!(L, store2)
    end
    # update cholesky factor C
    fill!(store1, 0); fill!(store2, 0)
    store1[j] = store1[i] = store2[j] = sqrt(abs(δ/2))
    store2[i] = -sqrt(abs(δ/2))
    if δ > 0
        cholupdate!(C, store1)
        choldowndate!(C, store2)
    else
        choldowndate!(C, store1)
        cholupdate!(C, store2)
    end
    return nothing
end

"""
    id_partition_groups(X::AbstractMatrix; [rss_target], [force_contiguous])
    id_partition_groups(Σ::Symmetric; [rss_target], [force_contiguous])

Compute group members based on interpolative decompositions. An initial pass 
first selects the most representative features such that regressing each 
non-represented feature on the selected will have residual less than `rss_target`.
The selected features are then defined as group centers and the remaining 
features are assigned to groups

# Inputs
+ `G`: Either individual level data `X` or a correlation matrix `Σ`. If one
    inputs `Σ`, it must be wrapped in the `Symmetric` argument, otherwise
    we will treat it as individual level data
+ `rss_target`: Target residual level (greater than 0) for the first pass, smaller
    means more groups
+ `force_contiguous`: Whether groups are forced to be contiguous. If true,
    variants are assigned its left or right center, whichever
    has the largest correlation with it without breaking contiguity.

# Outputs
+ `groups`: Length `p` vector of group membership for each variable

Note: interpolative decomposition is a stochastic algorithm. Set a seed to
guarantee reproducible results. 
"""
function id_partition_groups(
    G::AbstractMatrix;
    rss_target = 0.25,
    force_contiguous = false
    )
    p = size(G, 2)
    rss_target ≥ 0 || error("Expected rss_target to be > 0.")
    # get empirical correlation matrix
    Σ = typeof(G) <: Symmetric ? Matrix(G) : cor(G)
    all(x -> x ≈ 1, diag(Σ)) || error("G must be scaled to a correlation matrix first.")
    # step 1: compute rep columns by applying ID to X or cholesky of Σ
    A = typeof(G) <: Symmetric ? cholesky(PositiveFactorizations.Positive, G).U : G
    selected, _, _ = id(A)
    rk = search_rank(Σ, A, selected, rss_target)
    centers = sort(selected[1:rk])
    # step 2: bin non-represented members
    groups = zeros(Int, p)
    groups[centers] .= 1:rk
    non_rep = setdiff(1:p, centers)
    force_contiguous ? assign_members_cor_adj!(groups, Σ, non_rep, centers) : 
                       assign_members_cor!(groups, Σ, non_rep, centers)
    return groups
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
    prioritize_idx::Union{Vector{Int}, Nothing}=nothing, Σinv=inv(Σ)
    ) where T
    all(x -> x ≈ 1, diag(Σ)) || error("Σ must be scaled to a correlation matrix first.")
    0 < threshold < 1 || error("threshold should be in (0, 1) but was $threshold")
    unique_groups = unique(groups)
    group_reps = Int[]
    storage1 = zeros(size(Σ, 1), size(Σ, 2))
    storage2 = zeros(size(Σ, 2))
    @inbounds for g in unique_groups
        group_idx = findall(x -> x == g, groups) # all variables in this group
        O = findall(x -> x != g, groups) # all variables outside the group
        group_size = length(group_idx)
        if length(group_idx) == 1
            push!(group_reps, group_idx[1])
            continue
        end
        # for each variable in current group, compute an ordering of importance
        Σg = @view(Σ[group_idx, group_idx])
        index = select_best_rss_subset(Σg, group_size) # indices in current groups
        if !isnothing(prioritize_idx)
            prioritize_g_idx = filter!(!isnothing, 
                indexin(prioritize_idx, group_idx[index]))
            index = prioritize_variants(index, index[prioritize_g_idx])
        end
        indexΣ = group_idx[index] # indices in Σ
        # keep adding reps in current group until stopping criteria
        R = [indexΣ[1]]
        push!(group_reps, indexΣ[1])
        for i in 2:group_size
            RO = union(R, O) # variables in R and O
            Rc = setdiff(indexΣ, R) # variables not yet selected
            ROc = setdiff(1:size(Σ, 1), RO)
            Σ_RR_inv = inv(Σ[R, R])
            # compute Σ_RORO_inv = inv(Σ[RO, RO]) = 
            # Σinv[RO, RO] - Σinv[RO, ROc] * inv(Σinv[ROc, ROc]) * Σinv[ROc, RO]
            # using the fact that the quadratic form is low rank
            L = cholesky(Symmetric(Σinv[ROc, ROc]))
            X = inv(L.L) * Σinv[ROc, RO] # X'X = Σinv[RO, ROc] * inv(Σinv[ROc, ROc]) * Σinv[ROc, RO]
            Σ_RORO_inv = @view(storage1[1:length(RO), 1:length(RO)])
            Σ_RORO_inv .= @view(Σinv[RO, RO])
            BLAS.syrk!('U', 'T', -one(T), X, one(T), Σ_RORO_inv) # upper triangular only
            # LinearAlgebra.copytri!(Σ_RORO_inv, 'U')
            # compute ratio of variation explained by j
            ratio = zero(T)
            for j in Rc
                Σ_Rj = Σ[R, j]
                Σ_ROj = Σ[RO, j]
                R2_R = _dot(Σ_Rj, Σ_RR_inv, Σ_Rj, storage2) # R2_R = Σ_Rj*Σ_RR_inv*Σ_Rj
                R2_RO = _dot(Σ_ROj, Symmetric(Σ_RORO_inv), Σ_ROj, storage2)
                R2_R / R2_RO
                ratio += R2_R / R2_RO
            end
            ratio /= length(Rc)
            if ratio > threshold
                break
            else
                # select ith variable
                push!(R, indexΣ[i])
                push!(group_reps, indexΣ[i])
            end
        end
    end
    return sort!(group_reps)
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
function select_best_rss_subset(C::AbstractMatrix, k::Int)
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
    end
    # return indices, R2
    return indices
end

function assign_members_cor!(groups, Σ, non_rep, centers)
    for j in non_rep
        center, best_dist = 0, typemin(eltype(Σ))
        # find which of the representatives have largest absolute correlation with j
        for c in centers
            if abs(Σ[c, j]) > best_dist
                center = c
                best_dist = abs(Σ[c, j])
            end
        end
        # assign j to the group of its representative
        groups[j] = groups[center]
    end
    return groups
end

function assign_members_cor_adj!(groups, Σ, non_rep, rep_columns)
    issorted(rep_columns) || error("Expected rep_columns to be sorted")
    rep_columns_tmp = copy(rep_columns)
    for j in shuffle(non_rep)
        group_on_right = searchsortedfirst(rep_columns_tmp, j)
        if group_on_right > length(rep_columns_tmp) # no group on the right
            nearest_rep = rep_columns_tmp[end]
            push!(rep_columns_tmp, j)
        elseif group_on_right == 1 # j comes before the first group
            nearest_rep = rep_columns_tmp[1]
            insert!(rep_columns_tmp, 1, j)
        else # test which of the nearest representative is more correlated with j
            left  = rep_columns_tmp[group_on_right - 1]
            right = rep_columns_tmp[group_on_right]
            nearest_rep = abs(Σ[left, j]) > abs(Σ[right, j]) ? left : right
        end
        # assign j to the group of its representative
        groups[j] = groups[nearest_rep]
        insert!(rep_columns_tmp, group_on_right, j)
    end
    issorted(groups) || error("assign_members_cor_adj!: groups not contiguous")
    return groups
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

"""
    interpolative_decomposition(A::AbstractMatrix, rk::Int)

Computes the interpolative decomposition of A with rank `rk`
and returns the top `rk` most representative columns of `A`
"""
function interpolative_decomposition(A::AbstractMatrix, rk::Int)
    p = size(A, 1)
    # quick return
    rk > p && return collect(1:p)
    length(A) == 1 && return [1]
    # Run ID
    col_selected, redun_cols, T = id(A, rank=rk)
    return col_selected
end

function get_PCA_vectors(Σ::AbstractMatrix{T}, groups::AbstractVector{Int}) where T
    p = size(Σ, 1)
    p == size(Σ, 2) == length(groups) || 
        error("Expected size(Σ, 1) == size(Σ, 2) == length(groups)")
    # compute eigenfactorization for Σ blocks
    Σblocks = block_diagonalize(Σ, groups)
    _, evecs = eigen(Σblocks)
    # compute ID for each block
    # V2 = cholesky(Symmetric(Σblocks)).L
    # add columns of Σblocks to result
    # V2 = zeros(T, p, p)
    # for (j, v) in enumerate(eachcol(Σblocks))
    #     V2[:, j] .= v ./ norm(v)
    # end
    # purturb every element in the group equally
    # V2 = zeros(T, p, p)
    # for (j, g) in enumerate(unique(groups))
    #     idx = findall(x -> x == g, groups)
    #     V2[idx, j] .= 1 ./ sqrt(length(idx))
    # end
    # allow purturbion of only diagonal entries
    V2 = zeros(T, p, p)
    for i in 1:p
        V2[i, i] = 1
    end
    return unique([evecs V2], dims=2)
end
