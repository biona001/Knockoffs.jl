"""
    solve_s_group(Σ, groups, [method=:equi]; kwargs...)

Solves the group knockoff problem, returns block diagonal matrix S
satisfying `2Σ - S ⪰ 0` and the constant(s) γ.

# Inputs 
+ `Σ`: A covariance matrix that has been scaled to a correlation matrix.
+ `groups`: Vector of group membership, does not need to be contiguous
+ `method`: Method for constructing knockoffs. Options are `:equi` or `:sdp`
+ `m`: Number of knockoffs per variable, defaults to 1. 
+ `kwargs`: Extra arguments available for specific methods. For example, to use 
    less stringent convergence tolerance for MVR knockoffs, specify `tol = 0.001`.

# Output
+ `S`: A matrix solved so that `(m+1)/m*Σ - S ⪰ 0` and `S ⪰ 0`
+ `γ`: A vector that is only non-empty for equi and SDP knockoffs. They correspond to 
    values of γ where `S_{gg} = γΣ_{gg}`. So for equi, the vector is length 1. For 
    SDP, the vector has length equal to number of groups
"""
function solve_s_group(
    Σ::AbstractMatrix{T}, 
    groups::Vector{Int},
    method::Symbol=:maxent;
    m::Int=1,
    kwargs...
    ) where T
    # check for errors
    length(groups) == size(Σ, 1) == size(Σ, 2) || 
        error("Length of groups should be equal to dimension of Σ")
    all(x -> x ≈ 1, diag(Σ)) || error("Σ must be scaled to a correlation matrix first.")
    # if groups not contiguous, permute columns/rows of Σ so that they are contiguous
    perm = sortperm(groups)
    permuted = false
    if !issorted(groups)
        permute!(groups, perm)
        Σ .= @view(Σ[perm, perm])
        permuted = true
    end
    unique_groups = unique(groups)
    if length(unique_groups) == length(groups)
        # solve ungroup knockoff problem
        s = solve_s(Symmetric(Σ), 
            method == :sdp_subopt ? :sdp : method;
            m=m, kwargs...
        )
        S = Diagonal(s) |> Matrix
        γs = T[]
    else
        # solve group knockoff problem
        # grab Σgg blocks, for equicorrelated case we choose Sg = γΣg
        # for other cases we initialize with equi solution
        blocks = Matrix{T}[]
        for g in unique_groups
            idx = findall(x -> x == g, groups)
            push!(blocks, Σ[idx, idx])
        end
        Sblocks = BlockDiagonal(blocks)
        # solve optimization problem
        if method == :equi
            S, γs = solve_group_equi(Σ, Sblocks; m=m)
        elseif method == :sdp_subopt
            S, γs = solve_group_SDP_subopt(Σ, Sblocks; m=m)
        elseif method == :sdp_subopt_correct
            S, γs = solve_group_SDP_subopt_correct(Σ, Sblocks; m=m)
        elseif method == :sdp
            S, γs = solve_group_SDP_block_update(Σ, Sblocks; m=m, kwargs...)
        elseif method == :sdp_ccd
            S, γs = solve_group_SDP_ccd(Σ, Sblocks; m=m, kwargs...)
        elseif method == :sdp_full
            S, γs = solve_group_SDP_full(Σ, Sblocks; m=m)
        elseif method == :mvr
            S, γs = solve_group_MVR_ccd(Σ, Sblocks; m=m, kwargs...)
        elseif method == :maxent
            S, γs = solve_group_max_entropy_ccd(Σ, Sblocks; m=m, kwargs...)
        elseif method == :maxent_subopt
            S, γs = solve_group_max_entropy_suboptimal(Σ, Sblocks)
        else
            error("Method must be one of $GROUP_KNOCKOFFS but was $method")
        end
    end
    # permuate S back to the original noncontiguous group structure
    if permuted
        invpermute!(groups, perm)
        iperm = invperm(perm)
        S .= @view(S[iperm, iperm])
    end
    return S, γs
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
    return S, [γ]
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
    γs = clamp!(JuMP.value.(γ), 0, 1)
    S = BlockDiagonal(γs .* Σblocks.blocks) |> Matrix
    return S, γs
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
    γs = JuMP.value.(γ)
    S = BlockDiagonal(γs .* Σblocks.blocks) |> Matrix
    return S, γs
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

"""
# Todo
+ somehow avoid reallocating ub every iteration
+ When solving each individual block,
    - warmstart
    - avoid reallocating S1_new
    - allocate vector of models
    - use loose convergence criteria
+ Avoid allocating iperm in each iter by modifying https://github.com/JuliaLang/julia/blob/36034abf26062acad4af9dcec7c4fc53b260dbb4/base/combinatorics.jl#L278
+ For singleton groups, don't use JuMP and directly update
"""
function solve_group_SDP_block_update(
    Σ::AbstractMatrix, 
    Sblocks::BlockDiagonal;
    m::Int = 1,
    tol=0.01, # converges when changes in s are all smaller than tol
    niter = 10, # max number of cyclic block updates
    optm=Hypatia.Optimizer(verbose=false), # Any solver compatible with JuMP
    verbose::Bool = false,
    )
    p = size(Σ, 1)
    blocks = nblocks(Sblocks)
    group_sizes = size.(Sblocks.blocks, 1)
    perm = collect(1:p)
    # initialize S/A/D matrices
    S, _ = solve_group_equi(Σ, Sblocks, m=m)
    A = (m+1)/m * Σ
    D = A - S
    # compute initial objective value
    objective_values = group_sdp_objective(Σ, S, group_sizes)
    verbose && println("Init obj = $(sum(objective_values))")
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
            S .= @view(S[perm, perm])
            A .= @view(A[perm, perm])
            D .= @view(D[perm, perm])
            Σ .= @view(Σ[perm, perm])
            # update constraints
            S11 = @view(S[1:g, 1:g])
            Σ11 = @view(Σ[1:g, 1:g])
            A11 = @view(A[1:g, 1:g])
            D12 = @view(D[1:g, g + 1:end])
            D21 = @view(D[g + 1:end, 1:g])
            D22 = @view(D[g + 1:end, g + 1:end])
            ub = Symmetric(A11 - D12 * inv(D22 + 0.00001I) * D21)
            # solve SDP problem for current block
            S11_new, success = solve_group_SDP_single_block(Σ11, ub)
            block_obj = group_sdp_objective_single_block(Σ11, S11_new)
            # only update if objective decreased and optimization was successful
            if success && block_obj < objective_values[b]
                # find max difference between previous block S
                for i in eachindex(S11_new)
                    if abs(S11_new[i] - S11[i]) > max_delta
                        max_delta = abs(S11_new[i] - S11[i])
                    end
                end
                # update relevant blocks
                S11 .= S11_new
                D[1:g, 1:g] .= A11 .- S11_new
                objective_values[b] = block_obj
            end
            # repermute columns/rows of S back
            iperm = invperm(perm)
            S .= @view(S[iperm, iperm])
            A .= @view(A[iperm, iperm])
            D .= @view(D[iperm, iperm])
            Σ .= @view(Σ[iperm, iperm])
            sort!(perm)
            offset += g
        end
        verbose && println("Iter $l δ = $max_delta, obj = $(sum(objective_values))")
        max_delta < tol && break 
    end
    return S, Float64[]
end

# this assumes groups are contiguous, and each group's size is stored in group_sizes
function group_sdp_objective(Σ, S, group_sizes)
    blocks = length(group_sizes)
    objective_values, offset = zeros(blocks), 0
    for b in 1:blocks
        cur_idx = offset + 1:offset + group_sizes[b]
        objective_values[b] = group_sdp_objective_single_block(
            @view(Σ[cur_idx, cur_idx]), @view(S[cur_idx, cur_idx])
        )
        offset += group_sizes[b]
    end
    return objective_values # returns sum | Σij - Sij | for each group
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
    return JuMP.value.(S), T[]
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
    # whether to use robust cholesky updates or not
    cholupdate! = robust ? lowrankupdate! : lowrankupdate_turbo!
    choldowndate! = robust ? lowrankdowndate! : lowrankdowndate_turbo!
    # initialize S matrix and compute initial cholesky factor
    S, _ = solve_group_equi(Σ, Sblocks, m=m)
    S += λmin*I
    L = cholesky(Symmetric((m+1)/m * Σ - S + 2λmin*I))
    C = cholesky(Symmetric(S))
    verbose && println("initial obj = ", sum(group_sdp_objective(Σ, S, group_sizes)))
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
            obj = sum(group_sdp_objective(Σ, S, group_sizes))
            println("Iter $l: obj = $obj, δ = $max_delta, t1 = $(round(t1, digits=2)), t2 = $(round(t2, digits=2)), t3 = $(round(t3, digits=2))")
        end
        max_delta < tol && break 
    end
    return S, Float64[]
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
                (abs(δ) < 1e-15 || isnan(δ)) && continue
                # update S
                S[i, j] += δ
                S[j, i] += δ
                # update cholesky factors (if backtrack = true, this also undos the update if objective doesn't improve)
                t1 += @elapsed δ = update_offdiag_chol_mvr!(
                    S, L, C, storage, i, j, ei, ej, δ, m, choldowndate!, cholupdate!, backtrack
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
    return S, Float64[]
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
                (abs(δ) < 1e-15 || isnan(δ)) && continue
                # update S
                S[i, j] += δ
                S[j, i] += δ
                # update cholesky factors (if backtrack = true, this also undos the update if objective doesn't improve)
                t1 += @elapsed δ = update_offdiag_chol_maxent!(
                    S, L, C, x, i, j, ei, ej, δ, m, choldowndate!, cholupdate!, backtrack
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
    return S, Float64[]
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
        println("reached here1")
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

function update_offdiag_chol_mvr!(S, L, C, storage, i, j, ei, ej, δ, m, choldowndate!, cholupdate!, backtrack = true)
    backtrack && (obj_old = group_mvr_obj(L, C, m))
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
    !backtrack && return δ
    # if objective didn't decrease, undo the update
    new_obj = group_mvr_obj(L, C, m)
    failed = new_obj > obj_old
    if backtrack && failed
        S[i, j] -= δ
        S[j, i] -= δ
        # undo cholesky update to L
        fill!(storage, 0); fill!(ei, 0); fill!(ej, 0)
        storage[j] = storage[i] = ei[i] = ej[j] = sqrt(abs(δ))
        if δ > 0
            choldowndate!(L, ej)
            choldowndate!(L, ei)
            cholupdate!(L, storage)
        else 
            cholupdate!(L, ej)
            cholupdate!(L, ei)
            choldowndate!(L, storage)
        end
        # update cholesky factor C
        fill!(storage, 0); fill!(ei, 0); fill!(ej, 0)
        storage[j] = storage[i] = ei[i] = ej[j] = sqrt(abs(δ))
        if δ > 0
            cholupdate!(C, ej)
            cholupdate!(C, ei)
            choldowndate!(C, storage)
        else
            choldowndate!(C, ej)
            choldowndate!(C, ei)
            cholupdate!(C, storage)
        end
    end
    return failed ? 0 : δ
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

function update_offdiag_chol_maxent!(S, L, C, x, i, j, ei, ej, δ, m, choldowndate!, cholupdate!, backtrack = true)
    backtrack && (obj_old = group_maxent_obj(L, C, m))
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
    !backtrack && return δ
    # if objective didn't increase, undo the update
    new_obj = group_maxent_obj(L, C, m)
    failed = new_obj < obj_old
    if backtrack && failed
        S[i, j] -= δ
        S[j, i] -= δ
        # undo update to cholesky factor L
        fill!(x, 0); fill!(ei, 0); fill!(ej, 0)
        x[j] = x[i] = ei[i] = ej[j] = sqrt(abs(δ))
        if δ > 0
            choldowndate!(L, ej)
            choldowndate!(L, ei)
            cholupdate!(L, x)
        else 
            cholupdate!(L, ej)
            cholupdate!(L, ei)
            choldowndate!(L, x)
        end
        # undo update to cholesky factor C
        fill!(x, 0); fill!(ei, 0); fill!(ej, 0)
        x[j] = x[i] = ei[i] = ej[j] = sqrt(abs(δ))
        if δ > 0
            cholupdate!(C, ej)
            cholupdate!(C, ei)
            choldowndate!(C, x)
        else
            choldowndate!(C, ej)
            choldowndate!(C, ei)
            cholupdate!(C, x)
        end
    end
    return failed ? 0 : δ
end

# SDP construction by choosing S_g = γ_g * Σ_{g,g}
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
    return S, γs
end

"""
    modelX_gaussian_group_knockoffs(X, method, groups, μ, Σ)
    modelX_gaussian_group_knockoffs(X, method, groups; [covariance_approximator])

Constructs Gaussian model-X group knockoffs. If the covariance `Σ` and mean `μ` 
are not specified, they will be estimated from data, i.e. we will make second-order
group knockoffs. To incorporate group structure, the (true or estimated) covariance 
matrix is block-diagonalized according to `groups` membership to solve a relaxed 
optimization problem. See reference paper and Knockoffs.jl docs for more details. 

# Inputs
+ `X`: A `n × p` design matrix. Each row is a sample, each column is a feature.
+ `method`: Method for constructing knockoffs. Options are `:equi` or `:sdp`
+ `groups`: Vector of group membership
+ `μ`: A length `p` vector storing the true column means of `X`
+ `Σ`: A `p × p` covariance matrix for columns of `X`
+ `covariance_approximator`: A covariance estimator, defaults to 
    `LinearShrinkage(DiagonalUnequalVariance(), :lw)`. See CovarianceEstimation.jl 
    for more options.
+ `kwargs`: Extra keyword arguments for `solve_s_group`

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
    # Scale covariance to correlation matrix
    σs = sqrt.(diag(Σ))
    iscor = all(x -> x ≈ 1, σs)
    Σcor = iscor ? Σ : cov2cor!(Matrix(Σ), σs)
    # compute S matrix using the specified knockoff method
    S, γs = solve_s_group(Σcor, groups, method; m=m, kwargs...)
    # rescale S back to the result for a covariance matrix   
    iscor || cor2cov!(S, σs)
    # generate knockoffs
    X̃ = condition(X, μ, Σ, S; m=m)
    return GaussianGroupKnockoff(X, X̃, groups, S, γs, m, Symmetric(Σ), method)
end

"""
    modelX_gaussian_rep_group_knockoffs(X, method; [nrep], [cutoff], [m], [covariance_approximator], [kwargs...])
    modelX_gaussian_rep_group_knockoffs(X, method, μ, Σ; [nrep], [cutoff], [m], [kwargs...])
    modelX_gaussian_rep_group_knockoffs(X, method, μ, Σ, groups, group_reps; [nrep], [m], [kwargs...])

Selects `nrep` variables from each group and generate group knockoffs based on the 
smaller set of variants. If `nrep=1`, we generate (non-grouped) knockoffs.
"""
function modelX_gaussian_rep_group_knockoffs(
    X::AbstractMatrix{T}, 
    method::Symbol;
    nrep::Int = 1,
    cutoff::Number = 0.7,
    m::Int = 1,
    covariance_approximator=LinearShrinkage(DiagonalUnequalVariance(), :lw),
    kwargs... # extra arguments for solve_s or solve_s_group
    ) where T
    Σapprox = cov(covariance_approximator, X) # approximate covariance matrix
    μ = vec(mean(X, dims=1))                  # mean component is just column means
    return modelX_gaussian_rep_group_knockoffs(X, method, μ, Σapprox; m=m, 
        nrep=nrep, curoff=cutoff, kwargs...)
end

function modelX_gaussian_rep_group_knockoffs(
    X::AbstractMatrix{T}, 
    method::Symbol,
    μ::AbstractVector, 
    Σ::AbstractMatrix;
    nrep::Int = 1,
    m::Int = 1,
    cutoff::Number = 0.7,
    kwargs... # extra arguments for solve_s or solve_s_group
    ) where T
    groups, group_reps = hc_partition_groups(cov(X), cutoff=cutoff, nrep=nrep)
    return modelX_gaussian_rep_group_knockoffs(X, method, μ, Σ, groups, group_reps;
        m=m, nrep=nrep, kwargs...)
end

function modelX_gaussian_rep_group_knockoffs(
    X::AbstractMatrix{T}, 
    method::Symbol,
    μ::AbstractVector, 
    Σ::AbstractMatrix,
    groups::AbstractVector{Int},
    group_reps::AbstractVector{Int};
    nrep::Int = 1,
    m::Int = 1,
    kwargs... # extra arguments for solve_s or solve_s_group
    ) where T
    # note: these cannot be views because in the resulting struct requires concrete types not subarrays
    μrep = μ[group_reps]
    Σrep = Σ[group_reps, group_reps]
    Xrep = X[:, group_reps]

    if nrep == 1
        # generate (non-grouped) knockoff of X restricted to representative columns
        ko = modelX_gaussian_knockoffs(Xrep, method, μrep, Σrep; m=m, kwargs...)
    else
        # generate (smaller) group knockoffs of X
        Xrep_groups = groups[group_reps]
        ko = modelX_gaussian_group_knockoffs(Xrep, method, Xrep_groups, μrep, Σrep;
            m=m, kwargs...)
    end

    return GaussianRepGroupKnockoff(X, ko, groups, group_reps, nrep)
end

"""
    hc_partition_groups(Σ; [rep_method], [cutoff], [min_clusters], [nrep])

Computes a group partition based on correlation matrix `Σ` using single-linkage
hierarchical clustering. By default, a list of variables most representative
of each group will also be computed.

# Inputs
+ `Σ`: `p × p` correlation matrix
+ `cutoff`: Height value for which the clustering result is cut, between 0 and 1
    (default 0.7). This ensures that no variables between 2 groups have correlation
    greater than `cutoff`. 1 recovers ungrouped structure, 0 corresponds to 
    everything in a single group. 
+ `min_clusters`: The desired number of clusters. 
+ `nrep`: Number of representative per group. Defaults 1. If `nrep=1`, the 
    representative will be selected by computing which element has the smallest
    distance to all other elements in the cluster, i.e. the mediod. Otherise, 
    we will run interpolative decomposition to select representatives

If both `min_clusters` and `cutoff` are specified, it's guaranteed that the
number of clusters is not less than `min_clusters` and their height is not 
above `cutoff`.

# Outputs
+ `groups`: Length `p` vector of group membership for each variable
+ `rep_variables`: Variables most representative of X. These are typically used
    to compare group knockoff vs representative group knockoff

# todo:
add option to enforce adjacency constraint
"""
function hc_partition_groups(
    Σ::AbstractMatrix;
    cutoff = 0.7,
    min_clusters = 1,
    nrep = 1,
    )
    # check error
    p = size(Σ, 1)
    p == size(Σ, 2) || error("Expected size(Σ, 1) == size(Σ, 2)")
    all(x -> x ≈ 1, diag(Σ)) || error("Σ must be scaled to a correlation matrix first.")
    # convert correlation matrix to a distance matrix
    distmat = copy(Σ)
    @inbounds @simd for i in eachindex(distmat)
        distmat[i] = 1 - abs(distmat[i])
    end
    # hierarchical clustering
    cluster_result = hclust(distmat; linkage=:single)
    groups = cutree(cluster_result, h=1-cutoff, k=min_clusters)
    # select representatives
    if nrep == 1
        rep_variables = top_rep(distmat, groups)
    else
        rep_variables = id_reps(Σ, groups, nrep)
    end
    return groups, rep_variables
end

# computes a single representative from each group
function top_rep(distmat::AbstractMatrix, groups::AbstractVector)
    group_reps = Int[]
    for g in unique(groups)
        group_idx = findall(x -> x == g, groups)
        @views colsum = sum(distmat[group_idx, group_idx], dims=1) |> vec
        _, r1 = findmin(colsum)
        push!(group_reps, group_idx[r1])
    end
    return sort!(group_reps)
end

function id_reps(Σ::AbstractMatrix, groups::AbstractVector, nrep::Int)
    group_reps = Int[]
    for g in unique(groups)
        group_idx = findall(x -> x == g, groups)
        Σg = @views(Σ[group_idx, group_idx])
        rk = min(nrep, length(group_idx))
        col_selected = interpolative_decomposition(Σg, rk)
        for c in col_selected
            push!(group_reps, group_idx[c])
        end
    end
    return sort!(group_reps)
end

function interpolative_decomposition(Σ::AbstractMatrix, rk::Int)
    # check error
    p = size(Σ, 1)
    p == size(Σ, 2) || error("Expected size(Σ, 1) == size(Σ, 2)")
    rk ≤ p || error("maximum rank $rk exceeded dimension of Σ")
    all(x -> x ≈ 1, diag(Σ)) || error("Σ must be scaled to a correlation matrix first.")
    # Run ID
    A = cholesky(PositiveFactorizations.Positive, Σ).U
    col_selected, redun_cols, T = id(A, rank=rk)
    # col_selected = nothing
    # for rk in 1:max_rank
    #     col_selected, redun_cols, T = id(A, rank=rk)
    #     if norm(A[:, redun_cols] - A[:, col_selected]*T) / Anorm > 0.25
    #         break
    #     end
    # end
    return col_selected
end

# every `windowsize` SNPs form a group
function partition_group(snp_idx; windowsize=10)
    p = length(snp_idx)
    windows = floor(Int, p / windowsize)
    remainder = p - windows * windowsize
    groups = zeros(Int, p)
    for window in 1:windows
        groups[(window - 1)*windowsize + 1:window * windowsize] .= window
    end
    groups[p-remainder+1:p] .= windows + 1
    return groups
end

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
function modelX_gaussian_group_knockoffs(
    x::SnpArray, # assumes only have 1 chromosome, allows missing data
    method::Symbol;
    T::DataType = Float32,
    covariance_approximator=LinearShrinkage(DiagonalUnequalVariance(), :lw),
    outfile::Union{String, UndefInitializer} = undef,
    windowsize::Int = 10000
    )
    # estimate rough memory requirement (need Σ which is windowsize*windowsize and X which is n*windowsize)
    n, p = size(x)
    windows = ceil(Int, p / windowsize)
    @info "This routine requires at least $((T.size * windowsize^2 + T.size * n*windowsize) / 10^9) GB of RAM"
    # preallocated arrays
    xstore = Matrix{T}(undef, n, windowsize)
    X̃snparray = SnpArray(outfile, n, p)
    group_ranges = Vector{Int}[]
    Sblocks = Matrix{T}[]
    # loop over each window
    for window in 1:windows
        # import genotypes into numeric array
        cur_range = window == windows ? 
            ((windows - 1)*windowsize + 1:p) : 
            ((window - 1)*windowsize + 1:window * windowsize)
        @time copyto!(xstore, @view(x[:, cur_range]), impute=true)
        X = @view(xstore[:, 1:length(cur_range)])
        any(x -> iszero(x), std(X, dims=1)) &&
            error("Detected monomorphic SNPs. Please make sure QC is done properly.")
        # approximate covariance matrix and scale it to correlation matrix
        @time Σapprox = cov(covariance_approximator, X) # ~25 sec for 10k SNPs
        σs = sqrt.(diag(Σapprox))
        Σcor = cov2cor!(Σapprox.data, σs)
        # define group-blocks
        groups = partition_group(1:length(cur_range); windowsize=10)
        empty!(group_ranges); empty!(Sblocks)
        for g in unique(groups)
            idx = findall(x -> x == g, groups)
            push!(Sblocks, @view(Σcor[idx, idx]))
            push!(group_ranges, idx)
        end
        Sblock_diag = BlockDiagonal(Sblocks)
        # compute block diagonal S matrix using the specified knockoff method
        @time S, γs = solve_s_group(Σcor, Sblock_diag, groups, method) # 44.731886 seconds (13.44 M allocations: 4.467 GiB) (this step requires more memory allocation, need to analyze)
        # rescale S back to the result for a covariance matrix
        for (i, idx) in enumerate(group_ranges)
            cor2cov!(S.blocks[i], @view(σs[idx]))
        end
        # generate knockoffs
        μ = vec(mean(X, dims=1))
        @time X̃ = Knockoffs.condition(X, μ, Σapprox, S) # ~369 seconds (note: cholesky of 10k matrix takes ~16 seconds so why is this so slow?)
        # Force X̃_ij ∈ {0, 1, 2} (mainly done for large PLINK files where its impossible to store knockoffs in single/double precision)
        X̃ .= round.(X̃)
        clamp!(X̃, 0, 2)
        # count(vec(X̃) .!= vec(X)) # 160294 / 100000000 for a window
        # copy result into SnpArray
        for (j, jj) in enumerate(cur_range), i in 1:n
            X̃snparray[i, jj] = iszero(X̃[i, j]) ? 0x00 : 
                isone(X̃[i, j]) ? 0x02 : 0x03
        end
        # xtest = convert(Matrix{Float64}, @view(X̃snparray[:, cur_range]))
        # @assert all(xtest .== X̃)
    end
    return X̃snparray
end
