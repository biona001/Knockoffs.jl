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
    groups::Vector{Int}, m::Int, method) where T
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
    return obj
end

"""
    modelX_gaussian_group_knockoffs(X, method, groups, μ, Σ; [m], [nrep], [covariance_approximator])
    modelX_gaussian_group_knockoffs(X, method, groups; [m], [nrep], [covariance_approximator])

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
        knockoffs (contrary to using `:maxent`, we don't do line search for MVR
        group knockoffs because evaluating the objective is expensive)
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
    method::Symbol,
    groups::AbstractVector{Int};
    m::Int = 1,
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
    method::Symbol,
    groups::AbstractVector{Int},
    μ::AbstractVector{T},
    Σ::AbstractMatrix{T};
    m::Int = 1,
    kwargs...
    ) where T
    # first check errors
    length(groups) == size(X, 2) || 
        error("Expected length(groups) == size(X, 2). Each variable in X needs a group membership.")
    # compute S matrix using the specified knockoff method
    S, γs, obj = solve_s_group(Symmetric(Σ), groups, method; m=m, kwargs...)
    # generate knockoffs
    X̃ = condition(X, μ, Σ, S; m=m)
    return GaussianGroupKnockoff(X, X̃, groups, S, γs, m, Symmetric(Σ), method, obj)
end

"""
    modelX_gaussian_rep_group_knockoffs(X, method, groups, group_reps; [nrep], [m], [covariance_approximator], [kwargs...])
    modelX_gaussian_rep_group_knockoffs(X, method, μ, Σ, groups, group_reps; [nrep], [m], [kwargs...])

Constructs group knockoffs by choosing `nrep` representatives from each group and
solving a smaller optimization problem based on the representatives only. Remaining
knockoffs are generated based on a conditional independence assumption similar to
a graphical model (details to be given later). The representatives must be specified,
they can be computed via `hc_partition_groups` or `id_partition_groups`

# Inputs
+ `X`: A `n × p` design matrix. Each row is a sample, each column is a feature.
+ `method`: Method for constructing knockoffs. Options are the same as 
    `modelX_gaussian_group_knockoffs`
+ `groups`: Vector of `Int` denoting group membership. `groups[i]` is the group 
    of `X[:, i]`
+ `group_reps`: Vector of `Int` denoting the columns of `X` that will be used to 
    construct group knockoffs. That is, only `X[:, group_reps]` are used to solve
    the S matrix
+ `covariance_approximator`: A covariance estimator, defaults to 
    `LinearShrinkage(DiagonalUnequalVariance(), :lw)`. See CovarianceEstimation.jl 
    for more options.
+ `μ`: A length `p` vector storing the true column means of `X`
+ `Σ`: A `p × p` covariance matrix for columns of `X`
+ `nrep`: Max number of representatives per group, defaults to 5
+ `m`: Number of knockoffs per variable, defaults to 1. 
+ `kwargs`: Extra keyword arguments for `solve_s_group`
"""
function modelX_gaussian_rep_group_knockoffs(
    X::AbstractMatrix{T}, 
    method::Symbol,
    groups::AbstractVector{Int},
    group_reps::AbstractVector{Int};
    covariance_approximator=LinearShrinkage(DiagonalUnequalVariance(), :lw),
    nrep::Int = 5,
    m::Int = 1,
    kwargs... # extra arguments for solve_s or solve_s_group
    ) where T
    Σapprox = cov(covariance_approximator, X) # approximate covariance matrix
    μ = vec(mean(X, dims=1)) # empirical column means
    return modelX_gaussian_rep_group_knockoffs(X, method, μ, Σapprox, 
        groups, group_reps; nrep=nrep, m=m, kwargs...)
end

"""
Let X_R = (X_1,...,X_r); X_C = (X_{r+1},...,X_p)
"""
function modelX_gaussian_rep_group_knockoffs(
    X::AbstractMatrix{T}, # n × p
    method::Symbol,
    μ::AbstractVector, # p × 1
    Σ::AbstractMatrix, # p × p
    groups::AbstractVector{Int}, # p × 1 Vector{Int} of group membership
    group_reps::AbstractVector{Int}; # Vector{Int} with values in 1,...,p (columns of X that are representatives)
    m::Int = 1,
    nrep::Int = 5,
    verbose::Bool = true,
    kwargs... # extra arguments for solve_s or solve_s_group
    ) where T
    n, p = size(X)
    r = length(group_reps)
    all(x -> 1 ≤ x ≤ p, group_reps) || error("group_reps should be column indices of X")
    group_size = countmap(groups[group_reps]) |> values |> collect
    verbose && println("$r representatives for $p variables, $(sum(abs2, group_size)) optimization variables"); flush(stdout)

    # Compute S matrix on the representatives
    non_reps = setdiff(1:p, group_reps)
    Σ11 = Σ[group_reps, group_reps] # no view because Σ11 needs to be inverted later
    Σ12 = @views Σ[group_reps, non_reps]
    Σ22 = @views Σ[non_reps, non_reps]
    S, _, obj = solve_s_group(Symmetric(Σ11), groups[group_reps], method; m=m, kwargs...)

    # this samples 1 knockoff
    # Xr = @views X[:, group_reps]
    # Xc = @views X[:, non_reps]
    # X̃r_correct = Xr * (I - inv(Σ11) * S) + rand(MvNormal(Symmetric(2S - S * inv(Σ11) * S)), n)'
    # X̃c_correct = X̃r_correct * inv(Σ11) * Σ12 + rand(MvNormal(Symmetric(Σ22 - Σ21 * inv(Σ11) * Σ12)), n)'

    # sample multiple knockoffs (todo: sample each independently)
    Σ11inv = inv(Σ11)
    Σ11inv_Σ12 = Σ11inv * Σ12
    S_Σ11inv_Σ12 = S * Σ11inv_Σ12 # r × (p-r)
    D = Matrix{T}(undef, p, p)
    D[group_reps, group_reps] .= S
    D[group_reps, non_reps] .= S_Σ11inv_Σ12
    D[non_reps, group_reps] .= S_Σ11inv_Σ12'
    D[non_reps, non_reps] .= Σ22 - (Σ12' * Σ11inv * Σ12) + (Σ11inv_Σ12' * S * Σ11inv_Σ12)
    # @show eigmin(Symmetric(D))
    X̃ = condition(X, μ, Σ, Symmetric(D); m=m)

    return GaussianRepGroupKnockoff(X, X̃, groups, group_reps, S, 
        Symmetric(D), m, Symmetric(Σ), method, obj, nrep)
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
        knockoffs (contrary to using `:maxent`, we don't do line search for MVR
        group knockoffs because evaluating the objective is expensive)
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
    less stringent convergence tolerance for MVR knockoffs, specify `tol = 0.001`.

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
    method::Symbol;
    m::Int=1,
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
        # solve ungroup knockoff problem
        s = solve_s(Symmetric(Σcor), 
            method == :sdp_subopt ? :sdp : method;
            m=m, kwargs...
        )
        S = Diagonal(s) |> Matrix
        γs = T[]
        obj = group_block_objective(Σ, S, groups, m, method)
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
        elseif method == :sdp
            S, γs, obj = solve_group_SDP_ccd(Σcor, group_permuted; m=m, kwargs...)
        elseif method == :sdp_full
            S, γs, obj, _, _ = solve_group_SDP_full(Σcor, group_permuted; m=m)
        elseif method == :mvr
            S, γs, obj, _, _ = solve_group_MVR_ccd(Σcor, group_permuted; m=m, kwargs...)
        elseif method == :maxent
            S, γs, obj, _, _ = solve_group_max_entropy_ccd(Σcor, group_permuted; m=m, kwargs...)
        elseif method == :maxent_old
            S, γs, obj = solve_group_max_entropy_ccd_old(Σcor, group_permuted; m=m, kwargs...)
        elseif method == :mvr_old
            S, γs, obj = solve_group_MVR_ccd_old(Σcor, group_permuted; m=m, kwargs...)
        elseif method == :sdp_old
            S, γs, obj = solve_group_SDP_ccd_old(Σcor, group_permuted; m=m, kwargs...)
        elseif method == :maxent_pca
            S, γs, obj, _, _ = solve_group_max_entropy_pca(Σcor, group_permuted; m=m, kwargs...)
        elseif method == :maxent_pca_zihuai
            S, γs, obj = solve_group_max_entropy_pca_zihuai(Σcor, m, groups)
        elseif method == :mvr_pca
            S, γs, obj, _, _ = solve_group_MVR_pca(Σcor, group_permuted; m=m, kwargs...)
        elseif method == :sdp_pca
            S, γs, obj, _, _ = solve_group_SDP_pca(Σcor, group_permuted; m=m, kwargs...)
        elseif method == :maxent_hybrid
            S, γs, obj = solve_group_max_entropy_hybrid(Σcor, group_permuted; m=m, kwargs...)
        elseif method == :mvr_hybrid
            S, γs, obj = solve_group_mvr_hybrid(Σcor, group_permuted; m=m, kwargs...)
        elseif method == :sdp_hybrid
            S, γs, obj = solve_group_sdp_hybrid(Σcor, group_permuted; m=m, kwargs...)
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
function initialize_S(Σ, groups::Vector{Int}, m::Int, method, verbose)
    if occursin("equi", string(method))
        S, _, _ = solve_group_equi(Σ, groups, m=m)
        ϵ = 1e-8
        while !isposdef(S)
            S = (1-ϵ)S + ϵ*I
            ϵ *= 10
        end
        ϵ ≥ 1 && error("Error initialization. S cannot become PSD. Aborting")
        L = cholesky(Symmetric((m+1)/m * Σ - S))
        C = cholesky(Symmetric(S))
    elseif occursin("maxent", string(method))
        verbose && println("Performing 10 PCA-CCD steps to prime main algorithm")
        S, _, _, L, C = solve_group_max_entropy_pca(Σ, groups, m=m, niter=10, 
            verbose=verbose)
    elseif occursin("mvr", string(method))
        verbose && println("Performing 10 PCA-CCD steps to prime main algorithm")
        S, _, _, L, C = solve_group_MVR_pca(Σ, groups, m=m, niter=10, verbose=verbose)
    else
        verbose && println("Performing 10 PCA-CCD steps to prime main algorithm")
        S, _, _, L, C = solve_group_SDP_pca(Σ, groups, m=m, niter=10, verbose=verbose)
    end
    return S, L, C
end

"""
Computes A^{-1/2} via eigen-decomposition
"""
function inverse_mat_sqrt(A::Symmetric; tol=1e-4)
    λ, ϕ = eigen(A)
    for i in eachindex(λ)
        λ[i] < tol && (λ[i] = tol)
    end
    return ϕ * Diagonal(1 ./ sqrt.(λ)) * ϕ'
end

"""
    block_diagonalize(Σ, groups)

Internal function to block-diagonalize the covariance `Σ` according to groups. 
Here `groups` is assumed sorted. 
"""
function block_diagonalize(Σ::AbstractMatrix, groups::Vector{Int})
    issorted(groups) || error("groups is expected to be sorted!")
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
    m::Int = 1 # number of knockoffs per feature to generate
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
    m::Int = 1,
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
    m::Int = 1,
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
    m::Int; # number of knockoffs to generate
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
    m::Int; # number of knockoffs to generate
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
    method::Symbol;
    m::Int = 1,
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
    S, _, _ = initialize_S(Σ, groups, m, method, verbose)
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
            D22inv = inv(D22 + 0.00001I)
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
    m::Int = 1,
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

function solve_group_SDP_ccd(
    Σ::AbstractMatrix{T}, 
    groups::Vector{Int};
    niter::Int = 100,
    tol=0.01, # converges when changes in s are all smaller than tol,
    λmin=1e-6, # minimum eigenvalue of S and (m+1)/m Σ - S
    m::Int = 1, # number of knockoffs per variable
    robust::Bool = false, # whether to use "robust" Cholesky updates (if robust=true, alg will be ~10x slower, only use this if the default causes cholesky updates to fail)
    verbose::Bool = false,
    ) where T
    p = size(Σ, 1)
    group_sizes = [count(x -> x == g, groups) for g in unique(groups)]
    num_var = sum(abs2, group_sizes)
    # whether to use robust cholesky updates or not
    cholupdate! = robust ? lowrankupdate! : lowrankupdate_turbo!
    choldowndate! = robust ? lowrankdowndate! : lowrankdowndate_turbo!
    # initialize S matrix and initial cholesky factors
    S, L, C = initialize_S(Σ, groups, m, :sdp, verbose)
    obj = group_block_objective(Σ, S, groups, m, :sdp)
    verbose && 
        println("Full CCD initial obj = $obj, $(num_var) optimization variables")
    # preallocated vectors for efficiency
    u, v, ei, ej = zeros(p), zeros(p), zeros(p), zeros(p)
    # CCD iterations
    obj, _ = _sdp_ccd_iter!(
        S, L, C, Σ, groups,
        obj, m, group_sizes, niter, tol, λmin, verbose,
        cholupdate!, choldowndate!,
        u, v, ei, ej
    )
    return S, T[], group_block_objective(Σ, S, groups, m, :sdp), L, C
end

function _sdp_ccd_iter!(
    S, L, C, Σ, groups, # main matrix variables
    obj, m, group_sizes, niter, tol, λmin, verbose, # constants
    cholupdate!, choldowndate!, # cholesky update functions
    u, v, ei, ej, # storages
    )
    T = eltype(S)
    t1 = zero(T) # time for updating cholesky factors
    t2 = zero(T) # time for forward/backward solving
    t3 = zero(T) # time for solving offdiag 1D optimization problems
    blocks = length(group_sizes)
    converged = false
    for l in 1:niter
        max_delta = zero(T)
        offset = 0
        for b in 1:blocks
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
                ub = 1 / sum(abs2, u) - λmin
                lb = -1 / sum(abs2, v) + λmin
                lb ≥ ub && continue
                # compute new δ, making sure it is in feasible region
                δj = clamp(Σ[j, j] - S[j, j], lb, ub)
                change_obj = abs(Σ[j, j]-S[j, j]-δj) - abs(Σ[j, j]-S[j, j])
                # if change_obj > 0 || abs(δj) < 1e-15 || isnan(δj) || isinf(δj)
                #     continue
                # end
                # update S
                S[j, j] += δj
                obj += change_obj
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
                lb = max(s1, d1, -2 / (bii + 2bij + bjj)) + λmin
                ub = min(s2, d2, 2 / (aii + 2aij + ajj)) - λmin
                lb ≥ ub && continue
                # find δ ∈ [lb, ub] that maximizes objective
                δ = clamp(Σ[i, j] - S[i, j], lb, ub)
                change_obj = 2*abs(Σ[i, j]-S[i, j]-δ) - 2*abs(Σ[i, j]-S[i, j])
                # if change_obj > 0 || abs(δ) < 1e-15 || isnan(δ) || isinf(δ)
                #     continue
                # end
                # update S
                S[i, j] += δ
                S[j, i] += δ
                obj += change_obj
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
            println("Iter $l: obj = $obj, δ = $max_delta, " * 
                "t1 = $(round(t1, digits=2)), t2 = $(round(t2, digits=2)), " * 
                "t3 = $(round(t3, digits=2))")
        end
        if max_delta < tol
            converged = true
            break 
        end
    end
    return obj, converged
end

function solve_group_MVR_ccd(
    Σ::AbstractMatrix{T}, 
    groups::Vector{Int};
    niter::Int = 100,
    tol=0.01, # converges when changes in s are all smaller than tol,
    λmin=1e-6, # minimum eigenvalue of S and (m+1)/m Σ - S
    m::Int = 1, # number of knockoffs per variable
    robust::Bool = false, # whether to use "robust" Cholesky updates (if robust=true, alg will be ~10x slower, only use this if the default causes cholesky updates to fail)
    verbose::Bool = false
    ) where T
    p = size(Σ, 1)
    group_sizes = [count(x -> x == g, groups) for g in unique(groups)]
    num_var = sum(abs2, group_sizes)
    # whether to use robust cholesky updates or not
    cholupdate! = robust ? lowrankupdate! : lowrankupdate_turbo!
    choldowndate! = robust ? lowrankdowndate! : lowrankdowndate_turbo!
    # initialize S matrix and compute initial cholesky factor
    S, L, C = initialize_S(Σ, groups, m, :mvr, verbose)
    obj = group_block_objective(Σ, S, groups, m, :mvr)
    verbose && 
        println("Full CCD initial obj = $obj, $(num_var) optimization variables")
    # preallocated vectors for efficiency
    u, v, ei, ej, storage = zeros(p), zeros(p), zeros(p), zeros(p), zeros(p)
    # CCD iterations
    obj, _ = _mvr_ccd_iter!(
        S, L, C,
        obj, m, group_sizes, niter, tol, λmin, verbose,
        cholupdate!, choldowndate!,
        u, v, ei, ej, storage
    )
    return S, T[], obj, L, C
end

function _mvr_ccd_iter!(
    S, L, C, # main matrix variables
    obj, m, group_sizes, niter, tol, λmin, verbose, # constants
    cholupdate!, choldowndate!, # cholesky update functions
    u, v, ei, ej, storage # storages
    )
    T = eltype(S)
    t1 = zero(T) # time for updating cholesky factors
    t2 = zero(T) # time for forward/backward solving
    t3 = zero(T) # time for solving offdiag 1D optimization problems
    blocks = length(group_sizes)
    converged = false
    for l in 1:niter
        max_delta = zero(T)
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
                ub = 1 / ajj - λmin
                lb = -1 / bjj + λmin
                lb ≥ ub && continue
                x1, x2 = diag_mvr_obj_root(m, ajj, bjj, cjj, djj)
                δ = lb < x1 < ub ? x1 : x2
                # update S if objective improves
                change_obj = -m^2*δ*cjj/(1+δ*bjj) + δ*djj/(1-δ*ajj)
                if change_obj > 0 || abs(δ) < 1e-15 || isnan(δ) || isinf(δ)
                    continue
                end
                S[j, j] += δ
                obj += change_obj
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
                lb = max(s1, d1, -2 / (bii + 2bij + bjj)) + λmin
                ub = min(s2, d2, 2 / (aii + 2aij + ajj)) - λmin
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
                obj += change_obj
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
            println("Iter $l: obj = $obj, δ = $max_delta, " * 
                "t1 = $(round(t1, digits=2)), t2 = $(round(t2, digits=2))," * 
                "t3 = $(round(t3, digits=2))")
        end
        if max_delta < tol
            converged = true
            break 
        end
    end
    return obj, converged
end

# efficient and numerically stable way to evaluate max entropy objective 
# logdet((m+1)/m*Σ-S) + m*logdet(S) where
# C is cholesky factor of S and L is cholesky factor of (m+1)/m*Σ-S
function group_maxent_obj(L::Cholesky, C::Cholesky, m::Int)
    return logdet(L) + m*logdet(C)
end

function group_mvr_obj(L::Cholesky, C::Cholesky, m::Int, 
    storage::LowerTriangular{T}=LowerTriangular(zeros(size(L)))) where T
    copyto!(storage, I)
    ldiv!(C.L, storage)
    obj = m^2 * sum(abs2, storage)
    copyto!(storage, I)
    ldiv!(L.L, storage)
    obj += sum(abs2, storage)
    return obj
end

function solve_group_max_entropy_ccd(
    Σ::AbstractMatrix{T}, 
    groups::Vector{Int};
    niter::Int = 100,
    tol=0.01, # converges when changes in s are all smaller than tol,
    λmin=1e-6, # minimum eigenvalue of S and (m+1)/m Σ - S
    m::Int = 1, # number of knockoffs per variable
    robust::Bool = false, # whether to use "robust" Cholesky updates (if robust=true, alg will be ~10x slower, only use this if the default causes cholesky updates to fail)
    verbose::Bool = false,
    ) where T
    p = size(Σ, 1)
    group_sizes = [count(x -> x == g, groups) for g in unique(groups)]
    num_var = sum(abs2, group_sizes)
    # whether to use robust cholesky updates or not
    cholupdate! = robust ? lowrankupdate! : lowrankupdate_turbo!
    choldowndate! = robust ? lowrankdowndate! : lowrankdowndate_turbo!
    # initialize S matrix with PCA and compute initial cholesky factor
    S, L, C = initialize_S(Σ, groups, m, :maxent, verbose)
    obj = group_maxent_obj(L, C, m)
    verbose && 
        println("Full CCD initial obj = $obj, $(num_var) optimization variables")
    # preallocated vectors for efficiency
    u, v, ei, ej = zeros(p), zeros(p), zeros(p), zeros(p)
    # coordinate descent iterations
    obj, _ = _maxent_ccd_iter!(
        S, L, C, 
        obj, m, group_sizes, niter, tol, λmin, verbose, 
        cholupdate!, choldowndate!,
        u, v, ei, ej
    )
    obj = group_maxent_obj(L, C, m)
    return S, T[], obj, L, C
end

function _maxent_ccd_iter!(
    S, L, C, # main matrix variables
    obj, m, group_sizes, niter, tol, λmin, verbose, # constants
    cholupdate!, choldowndate!, # cholesky update functions
    u, v, ei, ej, # storages
    )
    T = eltype(S)
    t1 = zero(T) # time for updating cholesky factors
    t2 = zero(T) # time for forward/backward solving
    t3 = zero(T) # time for solving offdiag 1D optimization problems
    blocks = length(group_sizes)
    converged = false
    for l in 1:niter
        max_delta = zero(T)
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
                ub = 1 / ajj - λmin
                lb = -1 / bjj + λmin
                lb ≥ ub && continue
                δ = clamp(sj_new - S[j, j], lb, ub)
                # update S if objective improves
                change_obj = log(1 - δ*ajj) + m*log(1 + δ*bjj)
                if change_obj < 0 || abs(δ) < 1e-15 || isnan(δ) || isinf(δ)
                    continue
                end
                S[j, j] += δ
                obj += change_obj
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
                lb = max(s1, d1, -2 / (bii + 2bij + bjj)) + λmin
                ub = min(s2, d2, 2 / (aii + 2aij + ajj)) - λmin
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
                obj += change_obj
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
            println("Iter $l: obj = $obj, δ = $max_delta, t1 = " * 
                "$(round(t1, digits=2)), t2 = $(round(t2, digits=2)), " * 
                "t3 = $(round(t3, digits=2))")
        end
        if max_delta < tol
            converged = true
            break 
        end
    end
    return obj, converged
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
    size(Σg, 1) == size(Sg, 1) == length(v) || error("Dimension mismatch!")
    obj = zero(eltype(v))
    @inbounds for j in eachindex(v)
        @simd for i in eachindex(v)
            obj += abs(Σg[i, j] - Sg[i, j] - δ*v[i]*v[j])
        end
    end
    return obj
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

function solve_group_max_entropy_pca(
    Σ::AbstractMatrix{T}, 
    groups::Vector{Int};
    niter::Int = 100,
    tol=0.01, # converges when changes in s are all smaller than tol
    λmin=1e-6, # minimum eigenvalue of S and (m+1)/m Σ - S
    m::Int = 1, # number of knockoffs per variable
    verbose::Bool = false
    ) where T
    p = size(Σ, 1)
    # compute eigenfactorization for Σ blocks
    Σblocks = block_diagonalize(Σ, groups)
    _, evecs = eigen(Σblocks)
    # initialize S matrix and initial cholesky factors
    S, L, C = initialize_S(Σ, groups, m, :equi, verbose)
    # intial objective
    obj = group_maxent_obj(L, C, m)
    verbose && println("initial obj = $obj")
    # preallocated vectors for efficiency
    u, w = zeros(p), zeros(p)
    # PCA-CCD iterations
    obj, _ = _maxent_pca_ccd_iter!(
        S, L, C, evecs, 
        obj, m, niter, tol, verbose,
        u, w
    )
    return S, T[], obj, L, C
end

function _maxent_pca_ccd_iter!(
    S, L, C, evecs, # main matrix variables
    obj, m, niter, tol, verbose, # constants
    u, w # storages
    )
    T = eltype(S)
    t1 = zero(T) # time for updating cholesky factors
    t2 = zero(T) # time for forward/backward solving
    converged = false
    for l in 1:niter
        max_delta = zero(T)
        for v in eachcol(evecs)
            # get necessary constants
            t2 += @elapsed begin
                ldiv!(w, UpperTriangular(L.factors)', v)
                ldiv!(u, UpperTriangular(C.factors)', v)
                vt_Sinv_v = dot(u, u)
                vt_Dinv_v = dot(w, w)
            end
            # compute δ ∈ [lb, ub]
            δ = (m*vt_Sinv_v - vt_Dinv_v) / ((m+1)*vt_Sinv_v*vt_Dinv_v)
            lb = -1 / vt_Sinv_v
            ub = 1 / vt_Dinv_v
            δ = clamp(δ, lb, ub)
            # compute new objective
            change_obj = log(1 - δ*vt_Dinv_v) + m*log(1 + δ*vt_Sinv_v)
            if change_obj < 0 || abs(δ) < 1e-15 || isnan(δ) || isinf(δ)
                continue
            end
            # update S_new = S + δ*v*v'
            t1 += @elapsed BLAS.ger!(δ, v, v, S)
            obj += change_obj
            # update cholesky factors (must use robust updates since v is dense)
            u .= sqrt(abs(δ)) .* v
            w .= sqrt(abs(δ)) .* v
            t1 += @elapsed begin
                if δ > 0
                    LinearAlgebra.lowrankdowndate!(L, u)
                    LinearAlgebra.lowrankupdate!(C, w)
                else
                    LinearAlgebra.lowrankupdate!(L, u)
                    LinearAlgebra.lowrankdowndate!(C, w)
                end
            end
            # track convergence
            abs(δ) > max_delta && (max_delta = abs(δ))
        end
        if verbose
            println("Iter $l: obj = $obj, δ = $max_delta, t1 = " * 
                "$(round(t1, digits=2)), t2 = $(round(t2, digits=2))")
        end
        # check convergence
        if max_delta < tol
            converged = true
            break 
        end
    end
    return obj, converged
end

function solve_group_max_entropy_hybrid(
    Σ::AbstractMatrix{T}, 
    groups::Vector{Int};
    outer_iter::Int = 10,
    inner_pca_iter::Int = 10,
    inner_ccd_iter::Int = 5,
    tol=0.01, # converges when changes in s are all smaller than tol
    λmin=1e-6, # minimum eigenvalue of S and (m+1)/m Σ - S
    m::Int = 1, # number of knockoffs per variable
    robust::Bool = false, # whether to use "robust" Cholesky updates (if robust=true, CCD alg will be ~10x slower, only use this if the default causes cholesky updates to fail)
    verbose::Bool = false
    ) where T
    p = size(Σ, 1)
    # whether to use robust cholesky updates or not
    cholupdate! = robust ? lowrankupdate! : lowrankupdate_turbo!
    choldowndate! = robust ? lowrankdowndate! : lowrankdowndate_turbo!
    # compute eigenfactorization for Σ blocks
    Σblocks = block_diagonalize(Σ, groups)
    _, evecs = eigen(Σblocks)
    group_sizes = size.(Σblocks.blocks, 1)
    # initialize S matrix and initial cholesky factors
    S, L, C = initialize_S(Σ, groups, m, :equi, verbose)
    obj = group_maxent_obj(L, C, m)
    verbose && 
        println("Maxent hybrid CCD initial obj = $obj")
    # preallocated vectors for efficiency
    u, w, ei, ej = zeros(p), zeros(p), zeros(p), zeros(p)
    for i in 1:outer_iter
        # PCA iterations
        obj, converged1 = _maxent_pca_ccd_iter!(
            S, L, C, evecs, 
            obj, m, inner_pca_iter, tol, verbose,
            u, w
        )
        # Full CCD iterations
        obj, converged2 = _maxent_ccd_iter!(
            S, L, C, 
            obj, m, group_sizes, inner_ccd_iter, tol, λmin, verbose, 
            cholupdate!, choldowndate!,
            u, w, ei, ej
        )
        # check convergence
        if verbose
            println("\n Hybrid PCA-CCD iter $i: obj = $obj")
        end
        converged1 && converged2 && break
    end
    return S, T[], obj, L, C
end

function solve_group_MVR_pca(
    Σ::AbstractMatrix{T}, 
    groups::Vector{Int};
    niter::Int = 100,
    tol=0.01, # converges when changes in s are all smaller than tol
    λmin=1e-6, # minimum eigenvalue of S and (m+1)/m Σ - S
    m::Int = 1, # number of knockoffs per variable
    verbose::Bool = false
    ) where T
    p = size(Σ, 1)
    # compute eigenfactorization for Σ blocks
    Σblocks = block_diagonalize(Σ, groups)
    _, evecs = eigen(Σblocks)
    # initialize S matrix and initial cholesky factors
    S, L, C = initialize_S(Σ, groups, m, :equi, verbose)
    # intial objective
    obj = group_block_objective(Σ, S, groups, m, :mvr)
    verbose && println("initial obj = $obj")
    # some timers
    t1 = zero(T) # time for updating cholesky factors
    t2 = zero(T) # time for forward/backward solving
    # preallocated vectors for efficiency
    u, w, storage = zeros(p), zeros(p), zeros(p)
    # PCA-CCD iterations
    obj, _ = _mvr_pca_ccd_iter!(
        S, L, C, evecs, Σ,
        obj, m, niter, tol, verbose,
        u, w, storage
    )
    return S, T[], obj, L, C
end

function _mvr_pca_ccd_iter!(
    S, L, C, evecs, Σ, # main matrix variables
    obj, m, niter, tol, verbose, # constants
    u, w, storage # storages
    )
    T = eltype(S)
    t1 = zero(T) # time for updating cholesky factors
    t2 = zero(T) # time for forward/backward solving
    converged = false
    for l in 1:niter
        max_delta = zero(T)
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
            lb = -1 / vt_Sinv_v
            ub = 1 / vt_Dinv_v
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
            obj += change_obj
            t1 += @elapsed BLAS.ger!(δ, v, v, S)
            # update cholesky factors (must use robust updates since v is dense)
            u .= sqrt(abs(δ)) .* v
            w .= sqrt(abs(δ)) .* v
            t1 += @elapsed begin
                if δ > 0
                    LinearAlgebra.lowrankdowndate!(L, u)
                    LinearAlgebra.lowrankupdate!(C, w)
                else
                    LinearAlgebra.lowrankupdate!(L, u)
                    LinearAlgebra.lowrankdowndate!(C, w)
                end
            end
            # track convergence
            abs(δ) > max_delta && (max_delta = abs(δ))
        end
        if verbose
            println("Iter $l: obj = $obj, δ = $max_delta, t1 = " * 
                "$(round(t1, digits=2)), t2 = $(round(t2, digits=2))")
        end
        # check convergence
        if max_delta < tol
            converged = true
            break
        end
    end
    return obj, converged
end

function solve_group_mvr_hybrid(
    Σ::AbstractMatrix{T}, 
    groups::Vector{Int};
    outer_iter::Int = 10,
    inner_pca_iter::Int = 10,
    inner_ccd_iter::Int = 5,
    tol=0.01, # converges when changes in s are all smaller than tol
    λmin=1e-6, # minimum eigenvalue of S and (m+1)/m Σ - S
    m::Int = 1, # number of knockoffs per variable
    robust::Bool = false, # whether to use "robust" Cholesky updates (if robust=true, CCD alg will be ~10x slower, only use this if the default causes cholesky updates to fail)
    verbose::Bool = false
    ) where T
    p = size(Σ, 1)
    # whether to use robust cholesky updates or not
    cholupdate! = robust ? lowrankupdate! : lowrankupdate_turbo!
    choldowndate! = robust ? lowrankdowndate! : lowrankdowndate_turbo!
    # compute eigenfactorization for Σ blocks
    Σblocks = block_diagonalize(Σ, groups)
    _, evecs = eigen(Σblocks)
    group_sizes = size.(Σblocks.blocks, 1)
    # initialize S matrix and initial cholesky factors
    S, L, C = initialize_S(Σ, groups, m, :equi, verbose)
    obj = group_block_objective(Σ, S, groups, m, :mvr)
    verbose && 
        println("MVR hybrid CCD initial obj = $obj")
    # preallocated vectors for efficiency
    u, w, ei, ej, storage = zeros(p), zeros(p), zeros(p), zeros(p), zeros(p)
    for i in 1:outer_iter
        # PCA iterations
        obj, converged1 = _mvr_pca_ccd_iter!(
            S, L, C, evecs, Σ, 
            obj, m, inner_pca_iter, tol, verbose,
            u, w, storage
        )
        # Full CCD iterations
        obj, converged2 = _mvr_ccd_iter!(
            S, L, C,
            obj, m, group_sizes, inner_ccd_iter, tol, λmin, verbose,
            cholupdate!, choldowndate!,
            u, w, ei, ej, storage
        )
        # check convergence
        if verbose
            println("\n Hybrid PCA-CCD iter $i: obj = $obj")
        end
        converged1 && converged2 && break
    end
    return S, T[], obj, L, C
end

function solve_group_SDP_pca(
    Σ::AbstractMatrix{T}, 
    groups::Vector{Int};
    niter::Int = 100,
    tol=0.01, # converges when changes in s are all smaller than tol
    m::Int = 1, # number of knockoffs per variable
    verbose::Bool = false
    ) where T
    p = size(Σ, 1)
    # compute eigenfactorization for Σ blocks
    Σblocks = block_diagonalize(Σ, groups)
    _, evecs = eigen(Σblocks)
    # initialize S matrix and initial cholesky factors
    S, L, C = initialize_S(Σ, groups, m, :equi, verbose)
    # intial objective for each group
    group_objectives, group_idx = T[], Vector{Int}[]
    for g in unique(groups)
        idx = findall(x -> x == g, groups)
        obj_g = _sdp_block_objective(@view(Σ[idx, idx]), @view(S[idx, idx]))
        push!(group_objectives, obj_g)
        push!(group_idx, idx)
    end
    obj = sum(group_objectives)
    verbose && println("initial obj = $obj")
    # for each v, find its non-zero indices, and which group v updates
    nz_indices, v_groups = Vector{Int}[], Int[]
    for v in eachcol(evecs)
        nz_idx = findall(!iszero, v)
        g = findfirst(x -> x == nz_idx, group_idx) |> something
        push!(nz_indices, nz_idx)
        push!(v_groups, g)
    end
    # preallocated vectors for efficiency
    u, w = zeros(p), zeros(p)
    obj, _ = _sdp_pca_ccd_iter!(
        S, L, C, evecs, Σ,
        obj, niter, tol, verbose,
        nz_indices, v_groups, group_objectives,
        u, w, groups, m
    )
    return S, T[], obj, L, C
end

function _sdp_pca_ccd_iter!(
    S, L, C, evecs, Σ, # main matrix variables
    obj, niter, tol, verbose, # constants
    nz_indices, v_groups, group_objectives, # some precomputed variables
    u, w, groups, m # storages
    )
    T = eltype(S)
    t1 = zero(T) # time for updating cholesky factors
    t2 = zero(T) # time for forward/backward solving
    t3 = zero(T) # time for solving 1D optimization problems
    converged = false
    for l in 1:niter
        max_delta = zero(T)
        for (j, v) in enumerate(eachcol(evecs))
            nz_idx = nz_indices[j]
            v_group = v_groups[j]
            # get necessary constants
            t2 += @elapsed begin
                ldiv!(w, UpperTriangular(L.factors)', v)
                ldiv!(u, UpperTriangular(C.factors)', v)
                vt_Sinv_v = dot(u, u)
                vt_Dinv_v = dot(w, w)
            end
            # compute feasible region
            lb = -1 / vt_Sinv_v
            ub = 1 / vt_Dinv_v
            lb ≥ ub && continue
            # compute δ numerically
            Σg, Sg = @view(Σ[nz_idx, nz_idx]), @view(S[nz_idx, nz_idx])
            vg = @view(v[nz_idx])
            t3 += @elapsed opt = optimize(
                δ -> pca_sdp_obj(δ, Σg, Sg, vg),
                lb, ub, Brent(), show_trace=false, abs_tol=0.0001
            )
            δ = clamp(opt.minimizer, lb, ub)
            # find difference in objective
            change_obj = opt.minimum - group_objectives[v_group]
            # if change_obj > 0 || abs(δ) < 1e-15 || isnan(δ) || isinf(δ)
            #     continue
            # end
            # update S_new = S + δ*v*v'
            t1 += @elapsed BLAS.ger!(δ, v, v, S)
            obj += change_obj
            group_objectives[v_group] = opt.minimum
            # update cholesky factors (must use robust updates since v is dense)
            u .= sqrt(abs(δ)) .* v
            w .= sqrt(abs(δ)) .* v
            t1 += @elapsed begin
                if δ > 0
                    LinearAlgebra.lowrankdowndate!(L, u)
                    LinearAlgebra.lowrankupdate!(C, w)
                else
                    LinearAlgebra.lowrankupdate!(L, u)
                    LinearAlgebra.lowrankdowndate!(C, w)
                end
            end
            # track convergence
            abs(δ) > max_delta && (max_delta = abs(δ))
        end
        if verbose
            # obj_true = group_block_objective(Σ, S, groups, m, :sdp)
            # @show obj_true
            println("Iter $l: obj = $obj, δ = $max_delta, t1 = " * 
                "$(round(t1, digits=2)), t2 = $(round(t2, digits=2)), " * 
                "t3 = $(round(t3, digits=2))")
        end
        # check convergence
        if max_delta < tol
            converged = true
            break
        end
    end
    return obj, converged
end

function solve_group_sdp_hybrid(
    Σ::AbstractMatrix{T}, 
    groups::Vector{Int};
    outer_iter::Int = 10,
    inner_pca_iter::Int = 10,
    inner_ccd_iter::Int = 5,
    tol=0.01, # converges when changes in s are all smaller than tol
    λmin=1e-6, # minimum eigenvalue of S and (m+1)/m Σ - S
    m::Int = 1, # number of knockoffs per variable
    robust::Bool = false, # whether to use "robust" Cholesky updates (if robust=true, CCD alg will be ~10x slower, only use this if the default causes cholesky updates to fail)
    verbose::Bool = false
    ) where T
    p = size(Σ, 1)
    # whether to use robust cholesky updates or not
    cholupdate! = robust ? lowrankupdate! : lowrankupdate_turbo!
    choldowndate! = robust ? lowrankdowndate! : lowrankdowndate_turbo!
    # compute eigenfactorization for Σ blocks
    Σblocks = block_diagonalize(Σ, groups)
    _, evecs = eigen(Σblocks)
    group_sizes = size.(Σblocks.blocks, 1)
    # initialize S matrix and initial cholesky factors
    S, L, C = initialize_S(Σ, groups, m, :equi, verbose)
    # intial objective for each group
    group_objectives, group_idx = T[], Vector{Int}[]
    for g in unique(groups)
        idx = findall(x -> x == g, groups)
        obj_g = _sdp_block_objective(@view(Σ[idx, idx]), @view(S[idx, idx]))
        push!(group_objectives, obj_g)
        push!(group_idx, idx)
    end
    obj = sum(group_objectives)
    verbose && println("SDP hybrid CCD initial obj = $obj")
    # for each v, find its non-zero indices, and which group v updates
    nz_indices, v_groups = Vector{Int}[], Int[]
    for v in eachcol(evecs)
        nz_idx = findall(!iszero, v)
        g = findfirst(x -> x == nz_idx, group_idx) |> something
        push!(nz_indices, nz_idx)
        push!(v_groups, g)
    end
    # preallocated vectors for efficiency
    u, w, ei, ej = zeros(p), zeros(p), zeros(p), zeros(p)
    for i in 1:outer_iter
        # PCA iterations
        obj, converged1 = _sdp_pca_ccd_iter!(
            S, L, C, evecs, Σ,
            obj, inner_pca_iter, tol, verbose,
            nz_indices, v_groups, group_objectives,
            u, w, groups, m
        )
        # Full CCD iterations
        obj, converged2 = _sdp_ccd_iter!(
            S, L, C, Σ, groups,
            obj, m, group_sizes, inner_ccd_iter, tol, λmin, verbose,
            cholupdate!, choldowndate!,
            u, w, ei, ej
        )
        # check convergence
        if verbose
            println("\n Hybrid PCA-CCD iter $i: obj = $obj")
        end
        converged1 && converged2 && break
    end
    return S, T[], obj, L, C
end

"""
    id_partition_groups(X::AbstractMatrix; [nrep], [rep_method], [nrep], [rss_target], [force_contiguous])
    id_partition_groups(Σ::Symmetric; [nrep], [rep_method], [nrep], [rss_target], [force_contiguous])

Compute group members based on interpolative decompositions. An initial pass 
first selects the most representative features such that regressing each 
non-represented feature on the selected will have residual less than `rss_target`.
The selected features are then defined as group centers and the remaining 
features are assigned to groups

# Inputs
+ `G`: Either individual level data `X` or a correlation matrix `Σ`. If one
    inputs `Σ`, it must be wrapped in the `Symmetric` argument, otherwise
    we will treat it as individual level data
+ `nrep`: Number of representative per group. Initial group representatives are
    guaranteed to be selected
+ `rep_method`: Method for selecting representatives for each group. Options are
    `:id` (tends to select roughly independent variables) or `:rss` (tends to
    select more correlated variables)
+ `rss_target`: Target residual level (greater than 0) for the first pass, smaller
    means more groups
+ `force_contiguous`: Whether groups are forced to be contiguous. If true,
    variants are assigned its left or right center, whichever
    has the largest correlation with it without breaking contiguity.

# Outputs
+ `groups`: Length `p` vector of group membership for each variable
+ `rep_variables`: Columns of X selected as representatives. Each group have at 
    most `nrep` representatives. These are typically used to construct smaller
    group knockoff for extremely large groups

Note: interpolative decomposition is a stochastic algorithm. Set a seed to
guarantee reproducible results. 
"""
function id_partition_groups(
    G::AbstractMatrix;
    nrep = 1,
    rep_method = :id,
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
    # step 3: pick reprensetatives for each group. Centers are always selected
    group_reps = centers
    nrep > 1 && choose_group_reps!(group_reps, G, groups; method = rep_method, nrep = nrep)
    return groups, group_reps
end

"""
    hc_partition_groups(X::AbstractMatrix; [rep_method], [cutoff], [min_clusters], [nrep], [force_contiguous])
    hc_partition_groups(Σ::Symmetric; [rep_method], [cutoff], [min_clusters], [nrep], [force_contiguous])

Computes a group partition based on individual level data `X` or correlation 
matrix `Σ` using single-linkage hierarchical clustering. By default, a list of
variables most representative of each group will also be computed.

# Inputs
+ `X`: `n × p` data matrix. Each row is a sample
+ `Σ`: `p × p` correlation matrix. Must be wrapped in the `Symmetric` argument,
    otherwise we will treat it as individual level data
+ `cutoff`: Height value for which the clustering result is cut, between 0 and 1
    (default 0.7). This ensures that no variables between 2 groups have correlation
    greater than `cutoff`. 1 recovers ungrouped structure, 0 corresponds to 
    everything in a single group. 
+ `min_clusters`: The desired number of clusters. 
+ `nrep`: Number of representative per group. Defaults 1. If `nrep=1`, the 
    representative will be selected by computing which element has the smallest
    distance to all other elements in the cluster, i.e. the mediod. Otherise, 
    we will run interpolative decomposition to select representatives
+ `rep_method`: Method for selecting representatives for each group. Options are
    `:id` (tends to select roughly independent variables) or `:rss` (tends to
    select more correlated variables)

If both `min_clusters` and `cutoff` are specified, it's guaranteed that the
number of clusters is not less than `min_clusters` and their height is not 
above `cutoff`.

# Outputs
+ `groups`: Length `p` vector of group membership for each variable
+ `rep_variables`: Columns of X selected as representatives. Each group have at 
    most `nrep` representatives. These are typically used to construct smaller
    group knockoff for extremely large groups
+ `force_contiguous`: Whether groups are forced to be contiguous. If true,
    we will run adjacency constrained hierarchical clustering. 
"""
function hc_partition_groups(
    Σ::Symmetric;
    cutoff = 0.7,
    min_clusters = 1,
    nrep = 1,   
    rep_method = :id,
    force_contiguous = false
    )
    all(x -> x ≈ 1, diag(Σ)) || error("Σ must be scaled to a correlation matrix first.")
    # convert correlation matrix to a distance matrix
    distmat = copy(Matrix(Σ))
    @inbounds @simd for i in eachindex(distmat)
        distmat[i] = 1 - abs(distmat[i])
    end
    # hierarchical clustering
    if force_contiguous
        groups = adj_constrained_hclust(distmat, h=1-cutoff)
    else
        cluster_result = hclust(distmat; linkage=:single)
        groups = cutree(cluster_result, h=1-cutoff, k=min_clusters)
    end
    # pick reprensetatives for each group
    group_reps = choose_group_reps(Σ, groups; method = rep_method, nrep = nrep)
    return groups, group_reps
end
hc_partition_groups(X::AbstractMatrix; cutoff = 0.7, min_clusters = 1, nrep = 1, rep_method=:id, force_contiguous=false) = 
    hc_partition_groups(Symmetric(cor(X)), cutoff=cutoff, min_clusters=min_clusters, nrep=nrep, rep_method=rep_method, force_contiguous=force_contiguous)

"""
    adj_constrained_hclust(distmat::AbstractMatrix, h::Number)

Performs (single-linkage) hierarchical clustering, forcing groups to be contiguous.
We implement a bottom-up approach naively because `Clustering.jl` does not 
support adjacency constraints (see https://github.com/JuliaStats/Clustering.jl/issues/230)
"""
function adj_constrained_hclust(distmat::AbstractMatrix{T}; h::Number=0.3) where T
    0 ≤ h ≤ 1 || error("adj_constrained_hclust: expected 0 ≤ h ≤ 1 but got $h")
    p = size(distmat, 2)
    clusters = [[i] for i in 1:p] # initially all variables is its own cluster
    @inbounds for iter in 1:p-1
        remaining_clusters = length(clusters)
        min_d, max_d = typemax(T), typemin(T)
        merge_left, merge_right = 0, 0 # clusters to be merged
        # find min between-cluster distance among adjacent clusters
        for left in 1:remaining_clusters-1
            right = left + 1
            d = single_linkage_distance(distmat, clusters[left], clusters[right])
            if d < min_d
                merge_left, merge_right = left, right
                min_d = d
            end
            d > max_d && (max_d = d)
        end
        # merge 2 clusters with min distance
        for i in clusters[merge_right]
            push!(clusters[merge_left], i)
        end
        deleteat!(clusters, merge_right)
        # check for convergence
        min_d ≥ h && break
    end
    # let each cluster be its own group
    groups = zeros(Int, p)
    for (i, cluster) in enumerate(clusters), g in cluster
        groups[g] = i
    end
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
    choose_group_reps(G::AbstractMatrix, groups::AbstractVector; nrep = 1)
    choose_group_reps!(group_reps::Vector{Int}, C::AbstractMatrix, groups::AbstractVector; nrep = 1)

# Inputs
+ `G`: Either individual level data `X` or the correlation matrix `Σ`. If one
    inputs `Σ`, it must be wrapped in the `Symmetric` argument
"""
function choose_group_reps!(
    group_reps::Vector{Int}, 
    G::AbstractMatrix, 
    groups::AbstractVector;
    method = "id", # id or rss
    nrep = 1
    )
    unique_groups = unique(groups)
    offset = length(group_reps) > 0 ? 1 : 0
    if length(group_reps) > 0
        # if reprensetatives are already present, they are considered "group centers"
        # so check that there's only 1 rep per group
        length(unique(groups[group_reps])) == length(unique_groups) || 
            error("choose_group_reps!: if group_reps are supplied, " * 
                "each group should have only 1 representative")
    end
    if method == :id
        for g in unique_groups
            group_idx = findall(x -> x == g, groups) # all variables in this group
            group_members = setdiff!(group_idx, group_reps) # remove the representative
            length(group_members) == 0 && continue
            # Run ID on X[:, group_members] or cholesky of Σ[group_members, group_members]
            A = typeof(G) <: Symmetric ? 
                cholesky(PositiveFactorizations.Positive, Symmetric(G[group_members, group_members])).U :
                @view(G[:, group_members])
            rep_variables = interpolative_decomposition(A, nrep - offset)
            for rep in rep_variables
                push!(group_reps, group_members[rep])
            end
        end
    elseif method == :rss
        Σ = typeof(G) <: Symmetric ? G : cor(G)
        for g in unique_groups
            group_idx = findall(x -> x == g, groups) # all variables in this group
            group_members = setdiff!(group_idx, group_reps) # remove the representative
            length(group_members) == 0 && continue
            # compute top representatives by minimizing RSS of un-selected variants
            Σg = @view(Σ[group_members, group_members])
            rep_variables = select_best_rss_subset(Σg, nrep - offset)
            for rep in rep_variables
                push!(group_reps, group_members[rep])
            end
        end
    else 
        error("choose_group_reps!: expected method to be :id or :rss")
    end
    return sort!(group_reps)
end
choose_group_reps(G::AbstractMatrix, groups::AbstractVector; method=:id, nrep = 1) = 
    choose_group_reps!(Int[], G, groups, nrep=nrep, method=method)

function choose_group_reps_adapt!(
    group_reps::Vector{Int}, 
    G::AbstractMatrix, 
    groups::AbstractVector;
    method = "id", # id or rss
    nrep = 1
    )
    unique_groups = unique(groups)
    offset = length(group_reps) > 0 ? 1 : 0
    if length(group_reps) > 0
        # if reprensetatives are already present, they are considered "group centers"
        # so check that there's only 1 rep per group
        length(unique(groups[group_reps])) == length(unique_groups) || 
            error("choose_group_reps_adapt!: if group_reps are supplied, " * 
                "each group should have only 1 representative")
    end
    if method == :id
        for g in unique_groups
            group_idx = findall(x -> x == g, groups) # all variables in this group
            group_members = setdiff!(group_idx, group_reps) # remove the representative
            length(group_members) == 0 && continue
            # Run ID on X[:, group_members] or cholesky of Σ[group_members, group_members]
            A = typeof(G) <: Symmetric ? 
                cholesky(PositiveFactorizations.Positive, Symmetric(G[group_members, group_members])).U :
                @view(G[:, group_members])
            rep_variables = interpolative_decomposition(A, nrep - offset)
            for rep in rep_variables
                push!(group_reps, group_members[rep])
            end
        end
    elseif method == :rss
        Σ = typeof(G) <: Symmetric ? G : cor(G)
        for g in unique_groups
            group_idx = findall(x -> x == g, groups) # all variables in this group
            group_members = setdiff!(group_idx, group_reps) # remove the representative
            length(group_members) == 0 && continue
            # compute upper bound of variation that can be explained by other variants

            # compute top representatives by minimizing RSS of un-selected variants
            Σg = @view(Σ[group_members, group_members])
            rep_variables = select_best_rss_subset(Σg, nrep - offset)
            for rep in rep_variables
                push!(group_reps, group_members[rep])
            end
        end
    else 
        error("choose_group_reps_adapt!: expected method to be :id or :rss")
    end
    return sort!(group_reps)
end
choose_group_reps_adapt(G::AbstractMatrix, groups::AbstractVector; method=:id) = 
    choose_group_reps_adapt!(Int[], G, groups, method=method)
    

# faithful re-implementation of Trevor's R code. Probably not the most Julian/efficient Julia code
# select_one and select_best_rss_subset will help us choose k representatives from each group
# such that the RSS of the non-represented variables are minimized
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
    p ≤ k && return collect(1:p) # quick return
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
    for j in non_rep
        group_on_right = searchsortedfirst(rep_columns, j)
        if group_on_right > length(rep_columns) # no group on the right
            nearest_rep = rep_columns[end]
        elseif group_on_right == 1 # j comes before the first group
            nearest_rep = rep_columns[1]
        else # test which of the nearest representative is more correlated with j
            left, right = rep_columns[group_on_right - 1], rep_columns[group_on_right]
            nearest_rep = abs(Σ[left, j]) > abs(Σ[right, j]) ? left : right
        end
        # assign j to the group of its representative
        groups[j] = groups[nearest_rep]
    end
    # adhoc: second pass to ensure all groups are sorted, since routine above doesn't guarantee sorted
    # e.g. [111 22 3 4 55555 6666 5 666] (need to convert 66665666 at the far right to a 66666666)
    prev_group = 1
    for i in eachindex(groups)
        (groups[i] < prev_group) && (groups[i] = prev_group)
        (groups[i] > prev_group) && (prev_group = groups[i])
    end
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

# every `windowsize` SNPs form a group
# function partition_group(snp_idx; windowsize=10)
#     p = length(snp_idx)
#     windows = floor(Int, p / windowsize)
#     remainder = p - windows * windowsize
#     groups = zeros(Int, p)
#     for window in 1:windows
#         groups[(window - 1)*windowsize + 1:window * windowsize] .= window
#     end
#     groups[p-remainder+1:p] .= windows + 1
#     return groups
# end

"""
    modelX_gaussian_group_knockoffs(xdata::SnpData, method)

Generates (model-X Gaussian second-order) group knockoffs for
a single chromosome stored in PLINK formatted data. 

# todo 
+ Handle PLINK files with multiple chromosomes and multiple plink files each storing a chromosome
+ Make this accept multiple knockoffs
+ Output to PGEN which stores dosages
+ Better window definition via hierarchical clustering
"""
# function modelX_gaussian_group_knockoffs(
#     x::SnpArray, # assumes only have 1 chromosome, allows missing data
#     method::Symbol;
#     T::DataType = Float32,
#     covariance_approximator=LinearShrinkage(DiagonalUnequalVariance(), :lw),
#     outfile::Union{String, UndefInitializer} = undef,
#     windowsize::Int = 10000
#     )
#     # estimate rough memory requirement (need Σ which is windowsize*windowsize and X which is n*windowsize)
#     n, p = size(x)
#     windows = ceil(Int, p / windowsize)
#     @info "This routine requires at least $((T.size * windowsize^2 + T.size * n*windowsize) / 10^9) GB of RAM"
#     # preallocated arrays
#     xstore = Matrix{T}(undef, n, windowsize)
#     X̃snparray = SnpArray(outfile, n, p)
#     group_ranges = Vector{Int}[]
#     Sblocks = Matrix{T}[]
#     # loop over each window
#     for window in 1:windows
#         # import genotypes into numeric array
#         cur_range = window == windows ? 
#             ((windows - 1)*windowsize + 1:p) : 
#             ((window - 1)*windowsize + 1:window * windowsize)
#         @time copyto!(xstore, @view(x[:, cur_range]), impute=true)
#         X = @view(xstore[:, 1:length(cur_range)])
#         any(x -> iszero(x), std(X, dims=1)) &&
#             error("Detected monomorphic SNPs. Please make sure QC is done properly.")
#         # approximate covariance matrix and scale it to correlation matrix
#         @time Σapprox = cov(covariance_approximator, X) # ~25 sec for 10k SNPs
#         σs = sqrt.(diag(Σapprox))
#         Σcor = cov2cor!(Σapprox.data, σs)
#         # define group-blocks
#         groups = partition_group(1:length(cur_range); windowsize=10)
#         empty!(group_ranges); empty!(Sblocks)
#         for g in unique(groups)
#             idx = findall(x -> x == g, groups)
#             push!(Sblocks, @view(Σcor[idx, idx]))
#             push!(group_ranges, idx)
#         end
#         Sblock_diag = BlockDiagonal(Sblocks)
#         # compute block diagonal S matrix using the specified knockoff method
#         @time S, γs = solve_s_group(Σcor, Sblock_diag, groups, method) # 44.731886 seconds (13.44 M allocations: 4.467 GiB) (this step requires more memory allocation, need to analyze)
#         # rescale S back to the result for a covariance matrix
#         for (i, idx) in enumerate(group_ranges)
#             cor2cov!(S.blocks[i], @view(σs[idx]))
#         end
#         # generate knockoffs
#         μ = vec(mean(X, dims=1))
#         @time X̃ = Knockoffs.condition(X, μ, Σapprox, S) # ~369 seconds (note: cholesky of 10k matrix takes ~16 seconds so why is this so slow?)
#         # Force X̃_ij ∈ {0, 1, 2} (mainly done for large PLINK files where its impossible to store knockoffs in single/double precision)
#         X̃ .= round.(X̃)
#         clamp!(X̃, 0, 2)
#         # count(vec(X̃) .!= vec(X)) # 160294 / 100000000 for a window
#         # copy result into SnpArray
#         for (j, jj) in enumerate(cur_range), i in 1:n
#             X̃snparray[i, jj] = iszero(X̃[i, j]) ? 0x00 : 
#                 isone(X̃[i, j]) ? 0x02 : 0x03
#         end
#         # xtest = convert(Matrix{Float64}, @view(X̃snparray[:, cur_range]))
#         # @assert all(xtest .== X̃)
#     end
#     return X̃snparray
# end
