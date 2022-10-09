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
    # grab Σgg blocks, for equicorrelated case we choose Sg = γΣg, for other cases we initialize with equi solution
    blocks = Matrix{T}[]
    for g in unique(groups)
        idx = findall(x -> x == g, groups)
        push!(blocks, Σ[idx, idx])
    end
    Sblocks = BlockDiagonal(blocks)
    # solve optimization problem
    if method == :equi
        S, γs = solve_group_equi(Σ, Sblocks; m=m)
    elseif method == :sdp
        S, γs = solve_group_SDP(Σ, Sblocks; m=m)
    elseif method == :sdp_full
        S, γs = solve_group_SDP_full(Σ, Sblocks; m=m)
    elseif method == :mvr
        S, γs = solve_group_MVR_full(Σ, Sblocks; m=m, kwargs...)
    elseif method == :maxent
        S, γs = solve_group_max_entropy_full(Σ, Sblocks; m=m, kwargs...)
    else
        error("Method can only be :equi, :sdp, or :maxent, but was $method")
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
function solve_group_SDP(
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
    check_model_solution(model)
    γs = clamp!(JuMP.value.(γ), 0, 1)
    S = BlockDiagonal(γs .* Σblocks.blocks) |> Matrix
    return S, γs
end

function solve_group_SDP_full(Σ::AbstractMatrix, Σblocks::BlockDiagonal; m::Int = 1)
    model = Model(() -> Hypatia.Optimizer(verbose=false))
    # model = Model(() -> SCS.Optimizer())
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
    # @objective(model, Max, tr(S))
    @objective(model, Max, sum(S))
    JuMP.optimize!(model)
    check_model_solution(model)
    # construct block diagonal S
    Sm = convert(Matrix{T}, clamp!(JuMP.value.(S), 0, 1))
    Sdata = Matrix{T}[]
    idx = 0
    for g in group_sizes
        push!(Sdata, Symmetric(Sm[idx+1 : idx+g, idx+1 : idx+g]))
        idx += g
    end
    return BlockDiagonal(Sdata), T[]
end

function solve_group_MVR_full(
    Σ::AbstractMatrix{T}, 
    Sblocks::BlockDiagonal;
    niter::Int = 100,
    tol=1e-6, # converges when changes in s are all smaller than tol,
    λmin=1e-6, # minimum eigenvalue of S and (m+1)/m Σ - S
    m::Int = 1, # number of knockoffs per variable
    robust::Bool = false, # whether to use "robust" Cholesky updates (if robust=true, alg will be ~10x slower, only use this if the default causes cholesky updates to fail)
    verbose::Bool = false
    ) where T
    p = size(Σ, 1)
    blocks = nblocks(Sblocks)
    group_sizes = size.(Sblocks.blocks, 1)
    # whether to use robust cholesky updates or not
    cholupdate! = robust ? lowrankupdate! : lowrankupdate_turbo!
    choldowndate! = robust ? lowrankdowndate : lowrankdowndate_turbo!
    # initialize S matrix and compute initial cholesky factor
    S, _ = solve_group_equi(Σ, Sblocks)
    S = convert(Matrix{T}, S + λmin*I)
    L = cholesky(Symmetric((m+1)/m * Σ - S + 2λmin*I))
    C = cholesky(Symmetric(S))
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
                fill!(ei, 0)
                ej[j] = ei[j] = sqrt(abs(δj))
                t1 += @elapsed begin
                    if δj > 0
                        choldowndate!(L, ej)
                        cholupdate!(C, ei)
                    else
                        cholupdate!(L, ej)
                        choldowndate!(C, ei)
                    end
                end
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
                abs(δ) < 1e-15 || isnan(δ) && continue
                # update S
                S[i, j] += δ
                S[j, i] += δ
                # update cholesky factor L
                fill!(storage, 0); fill!(ei, 0); fill!(ej, 0)
                storage[j] = storage[i] = ei[i] = ej[j] = sqrt(abs(δ))
                t1 += @elapsed begin
                    if δ > 0
                        choldowndate!(L, storage)
                        cholupdate!(L, ei)
                        cholupdate!(L, ej)
                    else 
                        cholupdate!(L, storage)
                        choldowndate!(L, ei)
                        choldowndate!(L, ej)
                    end
                end
                # update cholesky factor C
                fill!(storage, 0); fill!(ei, 0); fill!(ej, 0)
                storage[j] = storage[i] = ei[i] = ej[j] = sqrt(abs(δ))
                t1 += @elapsed begin
                    if δ > 0
                        cholupdate!(C, storage)
                        choldowndate!(C, ei)
                        choldowndate!(C, ej)
                    else
                        choldowndate!(C, storage)
                        cholupdate!(C, ei)
                        cholupdate!(C, ej)
                    end
                end
                # update convergence tol
                abs(δ) > max_delta && (max_delta = abs(δ))
            end
            offset += group_sizes[b]
        end
        verbose && println("Iter $l: δ = $max_delta, t1 = $(round(t1, digits=2)), t2 = $(round(t2, digits=2)), t3 = $(round(t3, digits=2))")
        max_delta < tol && break 
    end
    return S, Float64[]
end

function solve_group_max_entropy_full(
    Σ::AbstractMatrix{T}, 
    Sblocks::BlockDiagonal;
    niter::Int = 100,
    tol=1e-6, # converges when changes in s are all smaller than tol,
    λmin=1e-6, # minimum eigenvalue of S and (m+1)/m Σ - S
    m::Int = 1, # number of knockoffs per variable
    robust::Bool = false, # whether to use "robust" Cholesky updates (if robust=true, alg will be ~10x slower, only use this if the default causes cholesky updates to fail)
    verbose::Bool = false
    ) where T
    p = size(Σ, 1)
    blocks = nblocks(Sblocks)
    group_sizes = size.(Sblocks.blocks, 1)
    # whether to use robust cholesky updates or not
    cholupdate! = robust ? lowrankupdate! : lowrankupdate_turbo!
    choldowndate! = robust ? lowrankdowndate : lowrankdowndate_turbo!
    # initialize S matrix and compute initial cholesky factor
    S, _ = solve_group_equi(Σ, Sblocks)
    S = convert(Matrix{T}, S + λmin*I)
    L = cholesky(Symmetric((m+1)/m * Σ - S + 2λmin*I))
    C = cholesky(Symmetric(S))
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
                t1 += @elapsed begin
                    if δ > 0
                        choldowndate!(L, x)
                        cholupdate!(C, ỹ)
                    else
                        cholupdate!(L, x)
                        choldowndate!(C, ỹ)
                    end
                end
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
                abs(δ) < 1e-15 || isnan(δ) && continue
                # update S
                S[i, j] += δ
                S[j, i] += δ
                # update cholesky factor L
                fill!(x, 0); fill!(ei, 0); fill!(ej, 0)
                x[j] = x[i] = ei[i] = ej[j] = sqrt(abs(δ))
                t1 += @elapsed begin
                    if δ > 0
                        choldowndate!(L, x)
                        cholupdate!(L, ei)
                        cholupdate!(L, ej)
                    else 
                        cholupdate!(L, x)
                        choldowndate!(L, ei)
                        choldowndate!(L, ej)
                    end
                end
                # update cholesky factor C
                fill!(x, 0); fill!(ei, 0); fill!(ej, 0)
                x[j] = x[i] = ei[i] = ej[j] = sqrt(abs(δ))
                t1 += @elapsed begin
                    if δ > 0
                        cholupdate!(C, x)
                        choldowndate!(C, ei)
                        choldowndate!(C, ej)
                    else
                        choldowndate!(C, x)
                        cholupdate!(C, ei)
                        cholupdate!(C, ej)
                    end
                end
                # update convergence tol
                abs(δ) > max_delta && (max_delta = abs(δ))
            end
            offset += group_sizes[b]
        end
        verbose && println("Iter $l: δ = $max_delta, t1 = $(round(t1, digits=2)), t2 = $(round(t2, digits=2)), t3 = $(round(t3, digits=2))")
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

# Reference
Dai & Barber 2016, The knockoff filter for FDR control in group-sparse and multitask regression
"""
function modelX_gaussian_group_knockoffs(
    X::AbstractMatrix{T}, 
    method::Symbol,
    groups::AbstractVector{Int};
    covariance_approximator=LinearShrinkage(DiagonalUnequalVariance(), :lw),
    kwargs...
    ) where T
    # approximate covariance matrix
    Σapprox = cov(covariance_approximator, X)
    # mean component is just column means
    μ = vec(mean(X, dims=1))
    return modelX_gaussian_group_knockoffs(X, method, groups, μ, Σapprox)
end

function modelX_gaussian_group_knockoffs(
    X::AbstractMatrix{T}, 
    method::Symbol,
    groups::AbstractVector{Int},
    μ::AbstractVector{T},
    Σ::AbstractMatrix{T};
    kwargs...
    ) where T
    # first check errors
    length(groups) == size(X, 2) || 
        error("Expected length(groups) == size(X, 2). Each variable in X needs a group membership.")
    # Scale covariance to correlation matrix
    σs = sqrt.(diag(Σ))
    iscor = all(x -> x ≈ 1, σs)
    Σcor = iscor ? Σ : StatsBase.cov2cor!(Matrix(Σ), σs)
    # compute S matrix using the specified knockoff method
    S, γs = solve_s_group(Σcor, groups, method; kwargs...)
    # rescale S back to the result for a covariance matrix   
    iscor || StatsBase.cor2cov!(S, σs)
    # generate knockoffs
    X̃ = condition(X, μ, Σ, S)
    return GaussianGroupKnockoff(X, X̃, groups, S, γs, Symmetric(Σ), method)
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
        Σcor = StatsBase.cov2cor!(Σapprox.data, σs)
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
            StatsBase.cor2cov!(S.blocks[i], @view(σs[idx]))
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
