## This file contains depreciated code and will be removed soon 
## The main problem was (1) Diagonal updates for MVR/ME re-used code from 
## non-grouped knockoff solvers, which assumed S is a diagonal matrix. This 
## does not violate any PSD and the solution seems generally good, but 
## technically the updates are not correct. (2) The MVR solver required repeated
## inversion of S and ((m+1)/mΣ-S) to backtrack, which is not necessary. 

function solve_group_max_entropy_ccd_old(
    Σ::AbstractMatrix{T}, 
    groups::Vector{Int};
    niter::Int = 100,
    tol=0.01, # converges when changes in s are all smaller than tol,
    λmin=1e-6, # minimum eigenvalue of S and (m+1)/m Σ - S
    m::Int = 1, # number of knockoffs per variable
    robust::Bool = false, # whether to use "robust" Cholesky updates (if robust=true, alg will be ~10x slower, only use this if the default causes cholesky updates to fail)
    verbose::Bool = false,
    backtrack::Bool = true
    ) where T
    Sblocks = block_diagonalize(Σ, groups)
    # constants
    p = size(Σ, 1)
    blocks = nblocks(Sblocks)
    group_sizes = size.(Sblocks.blocks, 1)
    num_var = sum(abs2, group_sizes)
    verbose && println("solve_group_max_entropy_ccd: Optimizing $(num_var) variables")
    # whether to use robust cholesky updates or not
    cholupdate! = robust ? lowrankupdate! : lowrankupdate_turbo!
    choldowndate! = robust ? lowrankdowndate! : lowrankdowndate_turbo!
    # initialize S matrix and compute initial cholesky factor
    S, _ = solve_group_equi(Σ, groups, m=m)
    S += λmin*I
    S ./= 2
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
                t1 += @elapsed δ = update_diag_chol_maxent_old!(
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
                t1 += @elapsed δ = update_offdiag_chol_maxent_old!(
                    L, C, x, i, j, ei, ej, δ, choldowndate!, cholupdate!
                )
                # update convergence tol
                abs(δ) > max_delta && (max_delta = abs(δ))
            end
            offset += group_sizes[b]
        end
        if verbose
            obj = group_maxent_obj(L, C, m)
            println("Iter $l: obj = $obj, δ = $max_delta, t1 = $(round(t1, digits=2)), t2 = $(round(t2, digits=2)), t3 = $(round(t3, digits=2))")
        end
        max_delta < tol && break 
    end
    return S, T[], group_maxent_obj(L, C, m)
end

function solve_group_MVR_ccd_old(
    Σ::AbstractMatrix{T}, 
    groups::Vector{Int};
    niter::Int = 100,
    tol=0.01, # converges when changes in s are all smaller than tol,
    λmin=1e-6, # minimum eigenvalue of S and (m+1)/m Σ - S
    m::Int = 1, # number of knockoffs per variable
    robust::Bool = false, # whether to use "robust" Cholesky updates (if robust=true, alg will be ~10x slower, only use this if the default causes cholesky updates to fail)
    verbose::Bool = false,
    backtrack::Bool = false # if true, need to evaluate objective which involves matrix inverses
    ) where T
    Sblocks = block_diagonalize(Σ, groups)
    # constants
    p = size(Σ, 1)
    blocks = nblocks(Sblocks)
    group_sizes = size.(Sblocks.blocks, 1)
    num_var = sum(abs2, group_sizes)
    verbose && println("solve_group_MVR_ccd_old: Optimizing $(num_var) variables")
    # whether to use robust cholesky updates or not
    cholupdate! = robust ? lowrankupdate! : lowrankupdate_turbo!
    choldowndate! = robust ? lowrankdowndate! : lowrankdowndate_turbo!
    # initialize S matrix and compute initial cholesky factor
    S, _ = solve_group_equi(Σ, groups, m=m)
    S += λmin*I
    S ./= 2
    L = cholesky(Symmetric((m+1)/m * Σ - S + 2λmin*I))
    C = cholesky(Symmetric(S))
    M = LowerTriangular(copy(S)) # used as storage for evaluating objective
    verbose && println("initial obj = ", group_mvr_obj!(M, L, C, m))
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
                    S, M, L, C, j, ei, ej, δj, m, choldowndate!, cholupdate!, backtrack
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
                # update S
                S[i, j] += δ
                S[j, i] += δ
                # update cholesky factors (if backtrack = true, this also undos the update if objective doesn't improve)
                t1 += @elapsed δ = update_offdiag_chol_mvr!(
                    S, M, L, C, storage, i, j, ei, ej, δ, m, choldowndate!, cholupdate!, backtrack
                )
                # update convergence tol
                abs(δ) > max_delta && (max_delta = abs(δ))
            end
            offset += group_sizes[b]
        end
        if verbose
            obj = group_mvr_obj!(M, L, C, m)
            println("Iter $l: obj = $obj, δ = $max_delta, t1 = $(round(t1, digits=2)), t2 = $(round(t2, digits=2)), t3 = $(round(t3, digits=2))")
        end
        max_delta < tol && break 
    end
    return S, T[], group_mvr_obj!(M, L, C, m)
end

function solve_group_SDP_ccd_old(
    Σ::AbstractMatrix{T}, 
    groups::Vector{Int};
    niter::Int = 100,
    tol=0.01, # converges when changes in s are all smaller than tol,
    λmin=1e-6, # minimum eigenvalue of S and (m+1)/m Σ - S
    m::Int = 1, # number of knockoffs per variable
    robust::Bool = false, # whether to use "robust" Cholesky updates (if robust=true, alg will be ~10x slower, only use this if the default causes cholesky updates to fail)
    verbose::Bool = false,
    backtrack::Bool = true
    ) where T
    Sblocks = block_diagonalize(Σ, groups)
    # constants
    p = size(Σ, 1)
    blocks = nblocks(Sblocks)
    group_sizes = size.(Sblocks.blocks, 1)
    num_var = sum(abs2, group_sizes)
    verbose && println("solve_group_SDP_ccd: Optimizing $(num_var) variables")
    # whether to use robust cholesky updates or not
    cholupdate! = robust ? lowrankupdate! : lowrankupdate_turbo!
    choldowndate! = robust ? lowrankdowndate! : lowrankdowndate_turbo!
    # initialize S matrix and compute initial cholesky factor
    S, _ = solve_group_equi(Σ, groups, m=m)
    # S += λmin*I
    # S ./= 2
    L = cholesky(Symmetric((m+1)/m * Σ - S + 2λmin*I))
    C = cholesky(Symmetric(S))
    verbose && println("initial obj = ", group_block_objective(Σ, S, groups, m, :sdp))
    # some timers
    t1 = zero(T) # time for updating cholesky factors
    t2 = zero(T) # time for forward/backward solving
    t3 = zero(T) # time for solving offdiag 1D optimization problems
    # preallocated vectors for efficiency
    u, v, ei, ej, storage = zeros(p), zeros(p), zeros(p), zeros(p), zeros(p)
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
            obj = group_block_objective(Σ, S, groups, m, :sdp)
            println("Iter $l: obj = $obj, δ = $max_delta, t1 = $(round(t1, digits=2)), t2 = $(round(t2, digits=2)), t3 = $(round(t3, digits=2))")
        end
        max_delta < tol && break 
    end
    return S, T[], group_block_objective(Σ, S, groups, m, :sdp)
end

function group_mvr_obj!(M::LowerTriangular{T}, L::Cholesky, C::Cholesky, m::Int) where T
    copyto!(M, I)
    ldiv!(C.L, M)
    obj = m^2 * sum(abs2, M)
    copyto!(M, I)
    ldiv!(L.L, M)
    obj += sum(abs2, M)
    return obj
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
        return zero(typeof(δ))
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

function update_diag_chol_mvr!(S, M, L, C, j, ei, ej, δj, m, choldowndate!, cholupdate!, backtrack = true)
    backtrack && (obj_old = group_mvr_obj!(M, L, C, m))
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
    new_obj = group_mvr_obj!(M, L, C, m)
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

function update_offdiag_chol_mvr!(S, M, L, C, storage, i, j, ei, ej, δ, m, choldowndate!, cholupdate!, backtrack = true)
    backtrack && (obj_old = group_mvr_obj!(M, L, C, m))
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
    new_obj = group_mvr_obj!(M, L, C, m)
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
    return failed ? zero(typeof(δ)) : δ
end

function update_diag_chol_maxent_old!(S, L, C, x, j, ỹ, δ, m, choldowndate!, cholupdate!, backtrack = true)
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
        # println("diag ($j, $j) objective failed to increase")
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

function update_offdiag_chol_maxent_old!(L, C, x, i, j, ei, ej, δ, choldowndate!, cholupdate!)
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
