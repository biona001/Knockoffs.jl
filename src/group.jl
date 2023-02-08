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

    # sample multiple knockoffs
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
    if max_group_size > 50 && method != :equi
        @warn "Maximum group size is $max_group_size, optimization may be slow. " * 
            "Consider running `modelX_gaussian_rep_group_knockoffs` to speed up convergence."
    end
    # Scale covariance to correlation matrix
    σs = sqrt.(diag(Σ))
    iscor = all(x -> x ≈ 1, σs)
    Σcor = iscor ? Σ : Symmetric(cov2cor!(Σ.data, σs))
    # if groups not contiguous, permute columns/rows of Σ so that they are contiguous
    perm = sortperm(groups)
    permuted = false
    if !issorted(groups)
        permute!(groups, perm)
        Σcor.data .= @view(Σcor.data[perm, perm])
        permuted = true
    end
    unique_groups = unique(groups)
    if length(unique_groups) == length(groups)
        # solve ungroup knockoff problem
        s = solve_s(Symmetric(Σcor), 
            method == :sdp_subopt ? :sdp : method;
            m=m, kwargs...
        )
        S = Diagonal(s) |> Matrix
        γs = T[]
        obj = zero(T) # non-grouped knockoffs do not compute objective
    else
        # solve group knockoff problem
        # grab Σgg blocks, for equicorrelated case we choose Sg = γΣg
        # for other cases we initialize with equi solution
        blocks = Matrix{T}[]
        for g in unique_groups
            idx = findall(x -> x == g, groups)
            push!(blocks, Σcor[idx, idx])
        end
        Sblocks = BlockDiagonal(blocks)
        # solve optimization problem
        if method == :equi
            S, γs, obj = solve_group_equi(Σcor, Sblocks; m=m)
        elseif method == :sdp_subopt
            S, γs, obj = solve_group_SDP_subopt(Σcor, Sblocks; m=m)
        elseif method == :sdp_subopt_correct
            S, γs, obj = solve_group_SDP_subopt_correct(Σcor, Sblocks; m=m)
        elseif method == :sdp_block
            S, γs, obj = solve_group_block_update(Σcor, Sblocks, method; m=m, kwargs...)
        elseif method == :mvr_block
            S, γs, obj = solve_group_block_update(Σcor, Sblocks, method; m=m, kwargs...)
        elseif method == :maxent_block
            S, γs, obj = solve_group_block_update(Σcor, Sblocks, method; m=m, kwargs...)
        elseif method == :sdp
            S, γs, obj = solve_group_SDP_ccd(Σcor, Sblocks; m=m, kwargs...)
        elseif method == :sdp_full
            S, γs, obj = solve_group_SDP_full(Σcor, Sblocks; m=m)
        elseif method == :mvr
            S, γs, obj = solve_group_MVR_ccd(Σcor, Sblocks; m=m, kwargs...)
        elseif method == :maxent
            S, γs, obj = solve_group_max_entropy_ccd(Σcor, Sblocks; m=m, kwargs...)
        elseif method == :maxent_subopt
            S, γs, obj = solve_group_max_entropy_suboptimal(Σcor, Sblocks)
        elseif method == :maxent_pca
            S, γs, obj = solve_group_max_entropy_pca(Σcor, m, groups)
        else
            error("Method must be one of $GROUP_KNOCKOFFS but was $method")
        end
    end
    # permuate S and Σ back to the original noncontiguous group structure
    if permuted
        invpermute!(groups, perm)
        iperm = invperm(perm)
        S .= @view(S[iperm, iperm])
        Σcor.data .= @view(Σcor.data[iperm, iperm])
    end
    # rescale S back to the result for a covariance matrix   
    iscor || cor2cov!(S, σs)
    return S, γs, obj
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
Solves the equi-correlated group knockoff problem. Here
`Σ` is the true covariance matrix (scaled so that it has 1 on its diagonal)
and `Σblocks` is the block-diagonal covariance matrix where each 
block corresponds to groups.

Details can be found in
Dai & Barber 2016, The knockoff filter for FDR control in group-sparse and multitask regression
"""
function solve_group_equi(
    Σ::AbstractMatrix, 
    Σblocks::BlockDiagonal;
    m::Int = 1 # number of knockoffs per feature to generate
    )
    Db = Matrix{eltype(Σ)}[]
    for Σbi in Σblocks.blocks
        push!(Db, inverse_mat_sqrt(Symmetric(Σbi)))
    end
    Db = BlockDiagonal(Db)
    λmin = Symmetric(Db * Σ * Db) |> eigmin
    γ = min(1, (m+1)/m * λmin)
    S = BlockDiagonal(γ .* Σblocks.blocks) |> Matrix
    return S, [γ], zero(eltype(Σ))
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
    Σblocks::BlockDiagonal; 
    m::Int = 1,
    verbose=false
    )
    model = Model(() -> Hypatia.Optimizer(verbose=verbose))
    # model = Model(() -> SCS.Optimizer())
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
    obj = group_block_objective(Σ, S, m, :sdp_subopt)
    return S, γs, obj
end

function solve_group_SDP_subopt_correct(
    Σ::AbstractMatrix, 
    Σblocks::BlockDiagonal; 
    m::Int = 1,
    verbose=false
    )
    model = Model(() -> Hypatia.Optimizer(verbose=verbose))
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
    obj = group_block_objective(Σ, S, m, method)
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
    Sblocks::BlockDiagonal,
    method::Symbol;
    m::Int = 1,
    tol=0.01, # converges when changes in s are all smaller than tol
    niter = 100, # max number of cyclic block updates
    verbose::Bool = false,
    ) where T
    method ∈ [:sdp_block, :maxent_block, :mvr_block] ||
        error("Expected method to be :sdp_block, :maxent_block, or :mvr_block")
    p = size(Σ, 1)
    blocks = nblocks(Sblocks)
    group_sizes = size.(Sblocks.blocks, 1)
    perm = collect(1:p)
    # initialize S/A/D matrices
    S, _ = solve_group_equi(Σ, Sblocks, m=m)
    A = (m+1)/m * Σ
    D = A - S
    Stmp = copy(S)
    # compute initial objective value
    obj = group_block_objective(Σ, S, m, method)
    verbose && println("Init obj = $obj")
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
                S11_new, opt_success = solve_group_MVR_single_block(Σ11, ub, D21, D22inv, m)
            end
            !opt_success && continue
            # check if objective improves
            obj_improves = false
            Stmp .= S
            Stmp[1:g, 1:g] .= S11_new
            new_obj = group_block_objective(Σ, Stmp, m, method)
            if method == :sdp_block || method == :mvr_block
                new_obj < obj && (obj_improves = true; obj = new_obj)
            else
                new_obj > obj && (obj_improves = true; obj = new_obj)
            end
            # only update if optimization was successful and objective improves
            if opt_success && obj_improves
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
        verbose && println("Iter $l δ = $max_delta, obj = $obj")
        max_delta < tol && break 
    end
    return S, T[], obj
end

# this evaluate the objective for SDP/MVR/ME
function group_block_objective(Σ, S, m, method)
    size(Σ) == size(S) || error("expected size(Σ) == size(S)")
    obj = zero(eltype(Σ))
    if occursin("sdp", string(method))
        for j in axes(Σ, 2), i in axes(Σ, 1)
            obj += abs(Σ[i, j] - S[i, j])
        end
    elseif occursin("maxent", string(method))
        obj += logdet((m+1)/m*Σ - S) + m*logdet(S)
    elseif occursin("mvr", string(method))
        obj += m^2*logdet(inv(S)) + tr(inv((m+1)/m*Σ - S))
    else
        error("methods can only be :sdp_block, :maxent_block, or :mvr_block")
    end
    return obj
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
    Σblocks::BlockDiagonal; 
    m::Int = 1,
    optm=Hypatia.Optimizer(verbose=false), # Any solver compatible with JuMP
    )
    model = Model(() -> optm)
    T = eltype(Σ)
    p = size(Σ, 1)
    group_sizes = size.(Σblocks.blocks, 1)
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
    obj = group_block_objective(Σ, S, m, method)
    return JuMP.value.(S), T[], obj
end

function solve_group_SDP_ccd(
    Σ::AbstractMatrix{T}, 
    Sblocks::BlockDiagonal;
    niter::Int = 100,
    tol=0.01, # converges when changes in s are all smaller than tol,
    λmin=1e-6, # minimum eigenvalue of S and (m+1)/m Σ - S
    m::Int = 1, # number of knockoffs per variable
    robust::Bool = false, # whether to use "robust" Cholesky updates (if robust=true, alg will be ~10x slower, only use this if the default causes cholesky updates to fail)
    verbose::Bool = false,
    backtrack::Bool = false # if true, need to evaluate objective which involves matrix inverses
    ) where T
    p = size(Σ, 1)
    blocks = nblocks(Sblocks)
    group_sizes = size.(Sblocks.blocks, 1)
    num_var = sum(abs2, group_sizes)
    verbose && println("solve_group_SDP_ccd: Optimizing $(num_var) variables")
    # whether to use robust cholesky updates or not
    cholupdate! = robust ? lowrankupdate! : lowrankupdate_turbo!
    choldowndate! = robust ? lowrankdowndate! : lowrankdowndate_turbo!
    # initialize S matrix and compute initial cholesky factor
    S, _ = solve_group_equi(Σ, Sblocks, m=m)
    S += λmin*I
    L = cholesky(Symmetric((m+1)/m * Σ - S + 2λmin*I))
    C = cholesky(Symmetric(S))
    verbose && println("initial obj = ", group_block_objective(Σ, S, m, :sdp))
    # some timers
    t1 = zero(T) # time for updating cholesky factors
    t2 = zero(T) # time for forward/backward solving
    t3 = zero(T) # time for solving offdiag 1D optimization problems
    # preallocated vectors for efficiency
    u, v, ei, ej = zeros(p), zeros(p), zeros(p), zeros(p)
    x, ỹ, storage = zeros(p), zeros(p), zeros(p)
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
                t2 += @elapsed ldiv!(u, UpperTriangular(L.factors)', ej) # non-allocating version of ldiv!(u, L.L, ej)
                t2 += @elapsed ldiv!(v, UpperTriangular(C.factors)', ej)
                ub = 1 / sum(abs2, u) - λmin
                lb = -1 / sum(abs2, v) + λmin
                lb ≥ ub && continue
                # compute new δ, making sure it is in feasible region
                δj = clamp(Σ[j, j] - S[j, j], lb, ub)
                abs(δj) < 1e-15 && continue
                # update S
                S[j, j] += δj
                # rank 1 update to cholesky factor
                t1 += @elapsed δj = update_diag_chol_sdp!(
                    S, Σ, L, C, j, ei, ej, δj, choldowndate!, cholupdate!, backtrack
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
                    ldiv!(u, UpperTriangular(L.factors)', ei) # non-allocating version of ldiv!(u, L.L, ei)
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
                lb = max(s1, d1, -1 / (bii + 2bij + bjj)) + λmin
                ub = min(s2, d2, 1 / (aii + 2aij + ajj)) - λmin
                lb ≥ ub && continue
                # find δ ∈ [lb, ub] that maximizes objective
                δ = clamp(Σ[i, j] - S[i, j], lb, ub)
                (abs(δ) < 1e-15 || isnan(δ)) && continue
                # update S
                S[i, j] += δ
                S[j, i] += δ
                # update cholesky factors (if backtrack = true, this also undos the update if objective doesn't improve)
                t1 += @elapsed δ = update_offdiag_chol_sdp!(
                    S, Σ, L, C, storage, i, j, ei, ej, δ, choldowndate!, cholupdate!, backtrack
                )
                # update convergence tol
                abs(δ) > max_delta && (max_delta = abs(δ))
            end
            offset += group_sizes[b]
        end
        if verbose
            obj = group_block_objective(Σ, S, m, :sdp)
            println("Iter $l: obj = $obj, δ = $max_delta, t1 = $(round(t1, digits=2)), t2 = $(round(t2, digits=2)), t3 = $(round(t3, digits=2))")
        end
        max_delta < tol && break 
    end
    return S, T[], group_block_objective(Σ, S, m, :sdp)
end

function solve_group_MVR_ccd(
    Σ::AbstractMatrix{T}, 
    Sblocks::BlockDiagonal;
    niter::Int = 100,
    tol=0.01, # converges when changes in s are all smaller than tol,
    λmin=1e-6, # minimum eigenvalue of S and (m+1)/m Σ - S
    m::Int = 1, # number of knockoffs per variable
    robust::Bool = false, # whether to use "robust" Cholesky updates (if robust=true, alg will be ~10x slower, only use this if the default causes cholesky updates to fail)
    verbose::Bool = false,
    backtrack::Bool = false # if true, need to evaluate objective which involves matrix inverses
    ) where T
    p = size(Σ, 1)
    blocks = nblocks(Sblocks)
    group_sizes = size.(Sblocks.blocks, 1)
    num_var = sum(abs2, group_sizes)
    verbose && println("solve_group_MVR_ccd: Optimizing $(num_var) variables")
    # whether to use robust cholesky updates or not
    cholupdate! = robust ? lowrankupdate! : lowrankupdate_turbo!
    choldowndate! = robust ? lowrankdowndate! : lowrankdowndate_turbo!
    # initialize S matrix and compute initial cholesky factor
    S, _ = solve_group_equi(Σ, Sblocks, m=m)
    S += λmin*I
    L = cholesky(Symmetric((m+1)/m * Σ - S + 2λmin*I))
    C = cholesky(Symmetric(S))
    verbose && println("initial obj = ", group_mvr_obj(L, C, m))
    # some timers
    t1 = zero(T) # time for updating cholesky factors
    t2 = zero(T) # time for forward/backward solving
    t3 = zero(T) # time for solving offdiag 1D optimization problems
    # preallocated vectors for efficiency
    u, v, ei, ej = zeros(p), zeros(p), zeros(p), zeros(p)
    vn, vd, storage = zeros(p), zeros(p), zeros(p)
    for l in 1:niter
        max_delta = zero(T)
        offset = 0
        for b in 1:blocks
            #
            # optimize diagonal entries
            #
            for idx in 1:group_sizes[b]
                j = idx + offset
                fill!(ej, 0)
                ej[j] = 1
                # compute cn and cd as detailed in eq 72
                t2 += @elapsed forward_backward!(vn, L, ej, storage) # solves L*L'*vn = ej for vn via forward-backward substitution
                cn = -sum(abs2, vn)
                # find vd as the solution to L*vd = ej
                t2 += @elapsed ldiv!(vd, UpperTriangular(L.factors)', ej) # non-allocating version of ldiv!(vd, L.L, ej)
                cd = sum(abs2, vd)
                # solve quadratic optimality condition in eq 71
                δj = solve_quadratic(cn, cd, S[j, j], m)
                # ensure new S[j, j] is in feasible region
                fill!(ej, 0)
                ej[j] = 1
                t2 += @elapsed ldiv!(u, UpperTriangular(L.factors)', ej) # non-allocating version of ldiv!(u, L.L, ej)
                t2 += @elapsed ldiv!(v, UpperTriangular(C.factors)', ej)
                ub = 1 / sum(abs2, u) - λmin
                lb = -1 / sum(abs2, v) + λmin
                lb ≥ ub && continue
                δj = clamp(δj, lb, ub)
                abs(δj) < 1e-15 && continue
                # update s
                S[j, j] += δj
                # rank 1 update to cholesky factor
                t1 += @elapsed δj = update_diag_chol_mvr!(
                    S, L, C, j, ei, ej, δj, m, choldowndate!, cholupdate!, backtrack
                )
                # update convergence tol
                abs(δj) > max_delta && (max_delta = abs(δj))
            end
            #
            # optimize off-diagonal entries
            #
            obj = group_mvr_obj(L, C, m)
            for idx1 in 1:group_sizes[b], idx2 in idx1+1:group_sizes[b]
                i, j = idx2 + offset, idx1 + offset
                fill!(ej, 0); fill!(ei, 0)
                ej[j], ei[i] = 1, 1
                # compute aii, ajj, aij, bii, bjj, bij
                t2 += @elapsed begin
                    ldiv!(u, UpperTriangular(L.factors)', ei) # non-allocating version of ldiv!(u, L.L, ei)
                    ldiv!(v, UpperTriangular(L.factors)', ej)
                    aij, aii, ajj = dot(u, v), dot(u, u), dot(v, v)
                    ldiv!(u, UpperTriangular(C.factors)', ei)
                    ldiv!(v, UpperTriangular(C.factors)', ej)
                    bij, bii, bjj = dot(u, v), dot(u, u), dot(v, v)
                    # compute cii, cjj, cij, dii, djj, dij
                    forward_backward!(u, C, ei, storage) # solves C*C'*u = ei for u via forward-backward substitution
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
                lb = max(s1, d1, -1 / (bii + 2bij + bjj)) + λmin
                ub = min(s2, d2, 1 / (aii + 2aij + ajj)) - λmin
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
                if change_obj > 0 || abs(δ) < 1e-15 || isnan(δ)
                    continue
                end
                obj += change_obj
                # update S
                S[i, j] += δ
                S[j, i] += δ
                # update cholesky factors
                t1 += @elapsed δ = update_offdiag_chol_mvr!(
                    L, C, storage, i, j, ei, ej, δ, choldowndate!, cholupdate!
                )
                # update convergence tol
                abs(δ) > max_delta && (max_delta = abs(δ))
            end
            offset += group_sizes[b]
        end
        if verbose
            obj = group_mvr_obj(L, C, m)
            println("Iter $l: obj = $obj, δ = $max_delta, t1 = $(round(t1, digits=2)), t2 = $(round(t2, digits=2)), t3 = $(round(t3, digits=2))")
        end
        max_delta < tol && break 
    end
    return S, Float64[], group_mvr_obj(L, C, m)
end

function solve_group_max_entropy_ccd(
    Σ::AbstractMatrix{T}, 
    Sblocks::BlockDiagonal;
    niter::Int = 100,
    tol=0.01, # converges when changes in s are all smaller than tol,
    λmin=1e-6, # minimum eigenvalue of S and (m+1)/m Σ - S
    m::Int = 1, # number of knockoffs per variable
    robust::Bool = false, # whether to use "robust" Cholesky updates (if robust=true, alg will be ~10x slower, only use this if the default causes cholesky updates to fail)
    verbose::Bool = false,
    backtrack::Bool = true
    ) where T
    p = size(Σ, 1)
    blocks = nblocks(Sblocks)
    group_sizes = size.(Sblocks.blocks, 1)
    num_var = sum(abs2, group_sizes)
    verbose && println("solve_group_max_entropy_ccd: Optimizing $(num_var) variables")
    # whether to use robust cholesky updates or not
    cholupdate! = robust ? lowrankupdate! : lowrankupdate_turbo!
    choldowndate! = robust ? lowrankdowndate! : lowrankdowndate_turbo!
    # initialize S matrix and compute initial cholesky factor
    S, _ = solve_group_equi(Σ, Sblocks, m=m)
    S += λmin*I
    L = cholesky(Symmetric((m+1)/m * Σ - S + 2λmin*I))
    C = cholesky(Symmetric(S))
    verbose && println("initial obj = ", group_maxent_obj(L, C, m))
    # some timers
    t1 = zero(T) # time for updating cholesky factors
    t2 = zero(T) # time for forward/backward solving
    t3 = zero(T) # time for solving offdiag 1D optimization problems
    # preallocated vectors for efficiency
    x, ỹ = zeros(p), zeros(p)
    u, v, ei, ej = zeros(p), zeros(p), zeros(p), zeros(p)
    for l in 1:niter
        max_delta = zero(T)
        offset = 0
        for b in 1:blocks
            #
            # optimize diagonal entries
            #
            for idx in 1:group_sizes[b]
                j = idx + offset
                @simd for i in 1:p
                    ỹ[i] = (m+1)/m * Σ[i, j]
                end
                ỹ[j] = 0
                # compute x as the solution to L*x = ỹ
                t2 += @elapsed ldiv!(x, UpperTriangular(L.factors)', ỹ) # non-allocating version of ldiv!(x, L.L, ỹ)
                x_l2sum = sum(abs2, x)
                # compute zeta and c as in alg 2.2 of askari et al
                ζ = (m+1)/m * Σ[j, j] - S[j, j]
                c = (ζ * x_l2sum) / (ζ + x_l2sum)
                # solve optimality condition in eq 75 of spector et al 2020
                sj_new = ((m+1)/m * Σ[j, j] - c) / 2
                # ensure new S[j, j] is in feasible region
                fill!(ej, 0)
                ej[j] = 1
                t2 += @elapsed ldiv!(u, UpperTriangular(L.factors)', ej) # non-allocating version of ldiv!(u, L.L, ej)
                t2 += @elapsed ldiv!(v, UpperTriangular(C.factors)', ej)
                ub = 1 / sum(abs2, u) - λmin
                lb = -1 / sum(abs2, v) + λmin
                lb ≥ ub && continue
                δ = clamp(sj_new - S[j, j], lb, ub)
                abs(δ) < 1e-15 && continue
                # update S
                S[j, j] += δ
                # rank 1 update to cholesky factors
                fill!(x, 0); fill!(ỹ, 0)
                x[j] = ỹ[j] = sqrt(abs(δ))
                t1 += @elapsed δ = update_diag_chol_maxent!(
                    S, L, C, x, j, ỹ, δ, m, choldowndate!, cholupdate!, backtrack
                )
                # update convergence tol
                abs(δ) > max_delta && (max_delta = abs(δ))
            end
            #
            # optimize off-diagonal entries
            #
            obj = group_maxent_obj(L, C, m)
            for idx1 in 1:group_sizes[b], idx2 in idx1+1:group_sizes[b]
                i, j = idx2 + offset, idx1 + offset
                fill!(ej, 0); fill!(ei, 0)
                ej[j], ei[i] = 1, 1
                # compute aii, ajj, aij, bii, bjj, bij
                t2 += @elapsed begin
                    ldiv!(u, UpperTriangular(L.factors)', ei) # non-allocating version of ldiv!(u, L.L, ei)
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
                lb = max(s1, d1, -1 / (bii + 2bij + bjj)) + λmin
                ub = min(s2, d2, 1 / (aii + 2aij + ajj)) - λmin
                lb ≥ ub && continue
                # find δ ∈ [lb, ub] that maximizes objective
                t3 += @elapsed opt = optimize(
                    δ -> offdiag_maxent_obj(δ, m, aij, aii, ajj, bij, bii, bjj),
                    lb, ub, Brent(), show_trace=false, abs_tol=0.0001
                )
                δ = clamp(opt.minimizer, lb, ub)
                change_obj = -opt.minimum
                if change_obj < 0 || abs(δ) < 1e-15 || isnan(δ)
                    continue
                end
                obj += change_obj
                # update S
                S[i, j] += δ
                S[j, i] += δ
                # update cholesky factors
                t1 += @elapsed δ = update_offdiag_chol_maxent!(
                    L, C, x, i, j, ei, ej, δ, choldowndate!, cholupdate!
                )
                # update convergence tol
                abs(δ) > max_delta && (max_delta = abs(δ))
            end
            offset += group_sizes[b]
        end
        if verbose
            obj = logdet(L) + m*logdet(C)
            println("Iter $l: obj = $obj, δ = $max_delta, t1 = $(round(t1, digits=2)), t2 = $(round(t2, digits=2)), t3 = $(round(t3, digits=2))")
        end
        max_delta < tol && break 
    end
    return S, T[], logdet(L) + m*logdet(C)
end

# objective function to minimize when optimizing off-diagonal entries in max entropy group knockoffs
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

# todo: can this be more efficient?
function group_mvr_obj(L::Cholesky, C::Cholesky, m::Int)
    return m^2*tr(inv(C.L * C.U)) + tr(inv(L.L * L.U))
end

function group_maxent_obj(L::Cholesky, C::Cholesky, m::Int)
    return logdet(L) + m*logdet(C)
end

function update_diag_chol_sdp!(S, Σ, L, C, j, ei, ej, δj, choldowndate!, cholupdate!, backtrack = true)
    obj_old = abs(Σ[j, j] - S[j, j])
    new_obj = abs(Σ[j, j] - S[j, j] - δj)
    if backtrack && new_obj > obj_old
        # undo the update if objective got worse
        S[j, j] -= δj
        return zero(typeof(δj))
    end
    # update cholesky factors
    fill!(ei, 0)
    fill!(ej, 0)
    ej[j] = ei[j] = sqrt(abs(δj))
    if δj > 0
        choldowndate!(L, ej)
        cholupdate!(C, ei)
    else
        cholupdate!(L, ej)
        choldowndate!(C, ei)
    end
    return δj
end

function update_offdiag_chol_sdp!(S, Σ, L, C, storage, i, j, ei, ej, δ, choldowndate!, cholupdate!, backtrack = true)
    obj_old = abs(Σ[i, j] - S[i, j])
    new_obj = abs(Σ[i, j] - S[i, j] - δ)
    if backtrack && new_obj > obj_old
        # undo the update if objective got worse
        S[i, j] -= δ
        S[j, i] -= δ
        return zero(typeof(δj))
    end
    # update cholesky factor L
    fill!(storage, 0); fill!(ei, 0); fill!(ej, 0)
    storage[j] = storage[i] = ei[i] = ej[j] = sqrt(abs(δ))
    if δ > 0
        choldowndate!(L, storage)
        cholupdate!(L, ei)
        cholupdate!(L, ej)
    else 
        cholupdate!(L, storage)
        choldowndate!(L, ei)
        choldowndate!(L, ej)
    end
    # update cholesky factor C
    fill!(storage, 0); fill!(ei, 0); fill!(ej, 0)
    storage[j] = storage[i] = ei[i] = ej[j] = sqrt(abs(δ))
    if δ > 0
        cholupdate!(C, storage)
        choldowndate!(C, ei)
        choldowndate!(C, ej)
    else
        choldowndate!(C, storage)
        cholupdate!(C, ei)
        cholupdate!(C, ej)
    end
    return δ
end

function update_diag_chol_mvr!(S, L, C, j, ei, ej, δj, m, choldowndate!, cholupdate!, backtrack = true)
    backtrack && (obj_old = group_mvr_obj(L, C, m))
    fill!(ei, 0)
    fill!(ej, 0)
    ej[j] = ei[j] = sqrt(abs(δj))
    if δj > 0
        choldowndate!(L, ej)
        cholupdate!(C, ei)
    else
        cholupdate!(L, ej)
        choldowndate!(C, ei)
    end
    !backtrack && return δj
    # if objective didn't decrease, undo the update
    new_obj = group_mvr_obj(L, C, m)
    failed = new_obj > obj_old
    if backtrack && failed
        S[j, j] -= δj
        fill!(ei, 0)
        fill!(ej, 0)
        ej[j] = ei[j] = sqrt(abs(δj))
        if δj > 0
            cholupdate!(L, ej)
            choldowndate!(C, ei)
        else
            choldowndate!(L, ej)
            cholupdate!(C, ei)
        end
    end
    return failed ? 0 : δj
end

function update_offdiag_chol_mvr!(L, C, storage, i, j, ei, ej, δ, choldowndate!, cholupdate!)
    # update cholesky factor L
    fill!(storage, 0); fill!(ei, 0); fill!(ej, 0)
    storage[j] = storage[i] = ei[i] = ej[j] = sqrt(abs(δ))
    if δ > 0
        choldowndate!(L, storage)
        cholupdate!(L, ei)
        cholupdate!(L, ej)
    else 
        cholupdate!(L, storage)
        choldowndate!(L, ei)
        choldowndate!(L, ej)
    end
    # update cholesky factor C
    fill!(storage, 0); fill!(ei, 0); fill!(ej, 0)
    storage[j] = storage[i] = ei[i] = ej[j] = sqrt(abs(δ))
    if δ > 0
        cholupdate!(C, storage)
        choldowndate!(C, ei)
        choldowndate!(C, ej)
    else
        choldowndate!(C, storage)
        cholupdate!(C, ei)
        cholupdate!(C, ej)
    end
    return δ
end

function update_diag_chol_maxent!(S, L, C, x, j, ỹ, δ, m, choldowndate!, cholupdate!, backtrack = true)
    backtrack && (obj_old = group_maxent_obj(L, C, m))
    fill!(x, 0); fill!(ỹ, 0)
    x[j] = ỹ[j] = sqrt(abs(δ))
    if δ > 0
        choldowndate!(L, x)
        cholupdate!(C, ỹ)
    else
        cholupdate!(L, x)
        choldowndate!(C, ỹ)
    end
    !backtrack && return δ
    # if objective didn't increase, undo the update
    new_obj = group_maxent_obj(L, C, m)
    failed = new_obj < obj_old
    if backtrack && failed
        S[j, j] -= δ
        fill!(x, 0); fill!(ỹ, 0)
        x[j] = ỹ[j] = sqrt(abs(δ))
        if δ > 0
            cholupdate!(L, x)
            choldowndate!(C, ỹ)
        else
            choldowndate!(L, x)
            cholupdate!(C, ỹ)
        end
    end
    return failed ? 0 : δ
end

function update_offdiag_chol_maxent!(L, C, x, i, j, ei, ej, δ, choldowndate!, cholupdate!)
    # update cholesky factor L
    fill!(x, 0); fill!(ei, 0); fill!(ej, 0)
    x[j] = x[i] = ei[i] = ej[j] = sqrt(abs(δ))
    if δ > 0
        choldowndate!(L, x)
        cholupdate!(L, ei)
        cholupdate!(L, ej)
    else 
        cholupdate!(L, x)
        choldowndate!(L, ei)
        choldowndate!(L, ej)
    end
    # update cholesky factor C
    fill!(x, 0); fill!(ei, 0); fill!(ej, 0)
    x[j] = x[i] = ei[i] = ej[j] = sqrt(abs(δ))
    if δ > 0
        cholupdate!(C, x)
        choldowndate!(C, ei)
        choldowndate!(C, ej)
    else
        choldowndate!(C, x)
        cholupdate!(C, ei)
        cholupdate!(C, ej)
    end
    return δ
end

# SDP construction for ME by choosing S_g = γ_g * Σ_{g,g}
# todo: multiple knockoffs, check objective value is correct
function solve_group_max_entropy_suboptimal(Σ::AbstractMatrix, Σblocks::BlockDiagonal)
    p = size(Σ, 1)
    G = length(Σblocks.blocks)
    group_sizes = size.(Σblocks.blocks, 1)
    # calculate Db = bdiag(Σ_{11}^{-1/2}, ..., Σ_{GG}^{-1/2})
    Db = Matrix{eltype(Σ)}[]
    for Σbi in Σblocks.blocks
        push!(Db, inverse_mat_sqrt(Symmetric(Σbi)))
    end
    Db = BlockDiagonal(Db)
    λ = Symmetric(2Db * Σ * Db) |> eigvals
    reverse!(λ)
    # solve non-linear objective using Ipopt
    model = Model(() -> Ipopt.Optimizer())
    set_optimizer_attribute(model, "print_level", 0)
    variant_to_group = zeros(Int, p)
    offset = 1
    for (idx, g) in enumerate(group_sizes)
        variant_to_group[offset:offset+g-1] .= idx
        offset += g
    end
    @variable(model, 0 <= γ[1:G] <= minimum(λ)) # todo: should this be γ[g] ≤ the corresponding λ?
    @NLobjective(model, Max, 
        sum(group_sizes[g] * log(γ[g]) for g in 1:G) + 
        sum(log(λ[i] - γ[variant_to_group[i]]) for i in 1:p)
    )
    JuMP.optimize!(model)
    success = check_model_solution(model)
    if !success
        @warn "Optimization unsuccessful, solution may be inaccurate"
    end
    # convert solution to vector and return resulting block diagonal matrix
    γs = convert(Vector{Float64}, clamp!(JuMP.value.(γ), 0, 1))
    S = BlockDiagonal(γs[1] .* Σblocks.blocks)
    return S, γs, objective_value(model)
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
