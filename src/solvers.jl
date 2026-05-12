"""
    solve_s(Σ::Symmetric, method::Symbol; m=1, kwargs...)

Solves the vector `s` for generating knockoffs. `Σ` can be a general 
covariance matrix but it must be wrapped in the `Symmetric` keyword. 

# Inputs
+ `Σ`: A covariance matrix (one must wrap `Symmetric(Σ)` explicitly)
+ `method`: Can be one of the following
    * `:mvr` for minimum variance-based reconstructability knockoffs (alg 1 in ref 2)
    * `:mvr_fast` for adaptive window-parallel MVR knockoffs
    * `:maxent` for maximum entropy knockoffs (alg 2 in ref 2)
    * `:maxent_fast` for adaptive window-parallel maximum entropy knockoffs
    * `:equi` for equi-distant knockoffs (eq 2.3 in ref 1), 
    * `:sdp` for SDP knockoffs via coordinate descent (alg 2.2 in ref 3)
    * `:sdp_parallel` for adaptive window-parallel SDP coordinate descent knockoffs
+ `m`: Number of knockoffs per variable, defaults to 1. 
+ `kwargs`: Extra arguments available for specific methods. For example, to use 
    less stringent convergence tolerance for MVR knockoffs, specify `tol = 0.01`.
    For a list of available options, see [`solve_MVR`](@ref),
    [`solve_max_entropy`](@ref), [`solve_SDP`](@ref), [`solve_sdp_parallel`](@ref), or
    [`solve_equi`](@ref)

# Reference
1. "Controlling the false discovery rate via Knockoffs" by Barber and Candes (2015).
2. "Powerful knockoffs via minimizing reconstructability" by Spector, Asher, and Lucas Janson (2020)
3. "FANOK: Knockoffs in Linear Time" by Askari et al. (2020).
"""
function solve_s(Σ::Symmetric, method::Union{Symbol, String}; m::Number=1, kwargs...)
    m < 1 && error("m should be 1 or larger but was $m.")
    method = Symbol(method)
    # create correlation matrix
    σs = sqrt.(diag(Σ))
    iscor = all(x -> x ≈ 1, σs)
    Σcor = iscor ? Σ : cov2cor(Σ.data, σs)
    # solve optimization problem
    if method == :equi
        s = solve_equi(Σcor; m=m)
    elseif method == :mvr
        s = solve_MVR(Σcor; m=m, kwargs...)
    elseif method == :mvr_fast
        s = solve_MVR_parallel(Σcor; m=m, kwargs...)
    elseif method == :maxent
        s = solve_max_entropy(Σcor; m=m, kwargs...)
    elseif method == :maxent_fast
        s = solve_max_entropy_parallel(Σcor; m=m, kwargs...)
    elseif method == :sdp
        s = solve_SDP(Σcor; m=m, kwargs...)
    elseif method == :sdp_parallel # change function name to solve_sdp_parallel but option name should be sdp_parallel
        s = solve_sdp_parallel(Σcor; m=m, kwargs...)
    else
        error("Method must be one of $SINGLE_KNOCKOFFS but was $method")
    end
    # rescale s back to the result for a covariance matrix   
    iscor || (s .*= σs.^2)
    return s
end

# this uses Convex.jl
"""
    solve_equi(Σ::AbstractMatrix)

Solves the equicorrelated problem for fixed-X and model-X knockoffs given 
correlation matrix Σ. Users should call `solve_s` instead of this function. 
"""
function solve_equi(
    Σ::AbstractMatrix{T}; # correlation matrix
    m::Number = 1 # number of multiple knockoffs to generate
    ) where T
    λmin = eigvals(Σ) |> minimum
    sj = min(1, (m+1)/m * λmin)
    return fill(sj, size(Σ, 1))
end

function _sdp_objective(Σ::AbstractMatrix, s::AbstractVector)
    return sum(abs.(diag(Σ) .- s))
end

function _maxent_objective(L::Cholesky, s::AbstractVector, m::Number)
    all(>(0), s) || return -Inf
    return logdet(L) + m * sum(log, s)
end

function _mvr_objective(L::Cholesky{T}, s::AbstractVector, m::Number) where T
    all(>(0), s) || return Inf
    storage = Matrix{T}(I, size(L, 1), size(L, 2))
    ldiv!(UpperTriangular(L.factors), storage)
    return sum(abs2, storage) + m^2 * sum(inv, s)
end

"""
    solve_MVR(Σ::AbstractMatrix)

Solves the minimum variance-based reconstructability problem for fixed-X
and model-X knockoffs given correlation matrix Σ. Users should call `solve_s` 
instead of this function. 

See algorithm 1 of "Powerful knockoffs via minimizing 
reconstructability" by Spector, Asher, and Lucas Janson (2020)
https://arxiv.org/pdf/2011.14625.pdf
"""
function solve_MVR(
    Σ::AbstractMatrix{T}; # correlation matrix
    niter::Int = 100,
    tol=1e-3, # converges when changes in s are all smaller than tol
    λmin=1e-6, # minimum eigenvalue of S and (m+1)/m Σ - S
    m::Number = 1, # number of knockoffs per variable
    s_init = solve_equi(Σ, m=m) ./ 2, # initialize away from the equicorrelated boundary
    robust::Bool = false, # whether to use "robust" Cholesky updates (if robust=true, alg will be ~10x slower, only use this if the default causes cholesky updates to fail)
    verbose::Bool = false
    ) where T
    p = size(Σ, 1)
    downdate_margin = sqrt(eps(T))
    # whether to use robust cholesky updates or not
    cholupdate! = robust ? lowrankupdate! : lowrankupdate_turbo!
    choldowndate! = robust ? lowrankdowndate! : lowrankdowndate_turbo!
    # initialize s vector and compute initial cholesky factor
    s = copy(s_init)
    L = cholesky(Symmetric(Matrix((m+1)/m*Σ - Diagonal(s) + λmin*I), :U))
    obj = verbose ? _mvr_objective(L, s, m) : zero(T)
    verbose && println("MVR initial obj = $obj")
    # preallocated vectors for efficiency
    vn, ej, vd, storage = zeros(p), zeros(p), zeros(p), zeros(p)
    @inbounds for l in 1:niter
        max_delta = zero(T)
        obj_new = obj
        for j in 1:p
            fill!(ej, 0)
            ej[j] = 1
            # compute cn and cd as detailed in eq 72
            forward_backward!(vn, L, ej, storage) # solves L*L'*vn = ej for vn via forward-backward substitution
            cn = -sum(abs2, vn)
            # find vd as the solution to L*vd = ej
            ldiv!(vd, UpperTriangular(L.factors)', ej) # non-allocating version of ldiv!(vd, L.L, ej)
            cd = sum(abs2, vd)
            # solve quadratic optimality condition in eq 71
            δj = solve_quadratic(cn, cd, s[j], m)
            # ensure s[j] + δj is in feasible region
            ub = max(zero(T), (1 - downdate_margin) / cd - λmin)
            δj > ub && (δj = ub)
            δj < -s[j] && (δj = -s[j])
            abs(δj) < 1e-15 && continue
            if verbose
                obj_new += δj * (-cn) / (1 - δj * cd) + m^2 * (inv(s[j] + δj) - inv(s[j]))
            end
            s[j] += δj
            # rank 1 update to cholesky factor
            ej[j] = sqrt(abs(δj))
            δj > 0 ? choldowndate!(L, ej) : cholupdate!(L, ej)
            # update convergence tol
            abs(δj) > max_delta && (max_delta = abs(δj))
        end
        if verbose
            println("Iter $l: obj = $obj_new, δ = $max_delta")
            flush(stdout)
        end
        obj = obj_new
        # declare convergence if changes in s are all smaller than tol
        max_delta < tol && break
    end
    return s
end

function _mvr_delta!(
    vn::AbstractVector{T},
    ej::AbstractVector{T},
    vd::AbstractVector{T},
    storage::AbstractVector{T},
    L,
    s::AbstractVector{T},
    j::Int,
    m,
    λmin
    ) where T
    fill!(ej, zero(T))
    ej[j] = one(T)
    forward_backward!(vn, L, ej, storage)
    cn = -sum(abs2, vn)
    ldiv!(vd, UpperTriangular(L.factors)', ej)
    cd = sum(abs2, vd)
    δ = solve_quadratic(cn, cd, s[j], m)
    ub = max(zero(T), (1 - sqrt(eps(T))) / cd - λmin)
    δ > ub && (δ = ub)
    δ < -s[j] && (δ = -s[j])
    return δ
end

function _print_parallel_serial_fallback(solver_name::AbstractString, nworkers::Int)
    println("$solver_name: using the serial solver because only $nworkers worker thread is available.")
    println("  Julia threads available: $(Threads.nthreads())")
    flush(stdout)
end

function _print_parallel_single_window_fallback(solver_name::AbstractString)
    println("$solver_name: only one approximately independent window was found; using the serial solver.")
    flush(stdout)
end

function _print_parallel_factor_check(err, factor_check_tol)
    println("Post-optimization Cholesky check: passed.")
    println("  Verified L'L - lambda_min*I ≈ ((m + 1) / m)Σ - S.")
    println("  Relative residual: $err")
    println("  Tolerance: $factor_check_tol")
    flush(stdout)
end

function _assert_parallel_cholesky_factor(
    L,
    Σ::AbstractMatrix,
    s::AbstractVector,
    γ,
    λmin;
    factor_check_tol=1e-4
    )
    U = UpperTriangular(L.factors)
    target = Matrix(γ * Σ - Diagonal(s))
    residual = Matrix(U' * U - target)
    if !iszero(λmin)
        @inbounds for j in 1:length(s)
            residual[j, j] -= λmin
        end
    end
    abs_error = norm(residual, Inf)
    rel_error = abs_error / max(one(abs_error), norm(target, Inf))
    if !(isfinite(rel_error) && rel_error ≤ factor_check_tol)
        error("Post-optimization Cholesky check failed: L'L - lambda_min*I was not close to ((m + 1) / m)Σ - S. Relative error $rel_error exceeds tolerance $factor_check_tol.")
    end
    return rel_error
end

"""
    solve_MVR_parallel(Σ::AbstractMatrix; kwargs...)

Adaptive window-parallel version of [`solve_MVR`](@ref). Users should call
[`solve_s`](@ref) with `method=:mvr_fast`.

The solver first reorders features using a nearest-correlation graph, partitions
the reordered variables at weak cross-correlation boundaries, and runs one
serial coordinate-descent sweep per window in parallel. The returned vector is
always sorted back to the input feature order. After optimization, the solver
checks that the maintained factor satisfies
`L' * L - λmin*I ≈ (m+1)/m*Σ - Diagonal(s)` and errors if the approximation is
not accurate enough.
"""
function solve_MVR_parallel(
    Σ::AbstractMatrix{T};
    niter::Int = 100,
    tol=1e-3,
    λmin=1e-6,
    m::Number = 1,
    s_init = solve_equi(Σ, m=m) ./ 2,
    verbose::Bool = false,
    nworkers::Int = Threads.nthreads(),
    feature_order::Union{Nothing, AbstractVector{Int}} = nothing,
    order_neighbors::Int = 8,
    boundary_band::Union{Nothing, Int} = nothing,
    min_window_size::Union{Nothing, Int} = nothing,
    window_corr_tol = 1e-3,
    factor_check::Bool = true,
    factor_check_tol = 1e-4
    ) where T
    p = size(Σ, 1)
    nworkers = max(1, min(nworkers, Threads.nthreads(), p))
    if nworkers == 1
        verbose && _print_parallel_serial_fallback("solve_MVR_parallel", nworkers)
        return solve_MVR(
            Σ;
            niter=niter,
            tol=tol,
            λmin=λmin,
            m=m,
            s_init=s_init,
            verbose=verbose
        )
    end

    Σinput = Σ
    Σ, s, order, inverse_order, windows, nworkers, boundary_band, min_window_size = _parallel_setup(
        Σ,
        s_init,
        nworkers,
        feature_order,
        order_neighbors,
        boundary_band=boundary_band,
        min_window_size=min_window_size,
        window_corr_tol=window_corr_tol
    )
    verbose && _print_parallel_setup_report(
        order,
        windows,
        nworkers,
        feature_order,
        boundary_band,
        min_window_size,
        window_corr_tol
    )
    if length(windows) == 1
        verbose && _print_parallel_single_window_fallback("solve_MVR_parallel")
        return solve_MVR(
            Σinput;
            niter=niter,
            tol=tol,
            λmin=λmin,
            m=m,
            s_init=s_init,
            verbose=verbose
        )
    end

    p = size(Σ, 1)
    L = cholesky(Symmetric(Matrix((m+1)/m*Σ - Diagonal(s) + λmin*I), :U))
    obj = verbose ? _mvr_objective(L, s, m) : zero(T)
    verbose && println("MVR initial obj = $obj")

    nthread_buffers = Threads.maxthreadid()
    vnwork = [zeros(T, p) for _ in 1:nthread_buffers]
    ejwork = [zeros(T, p) for _ in 1:nthread_buffers]
    vdwork = [zeros(T, p) for _ in 1:nthread_buffers]
    storagework = [zeros(T, p) for _ in 1:nthread_buffers]
    update_work = [zeros(T, p) for _ in 1:nthread_buffers]
    max_deltas = zeros(T, nthread_buffers)

    @inbounds for l in 1:niter
        fill!(max_deltas, zero(T))
        Threads.@threads for widx in eachindex(windows)
            tid = Threads.threadid()
            local_max_delta = zero(T)
            for j in windows[widx]
                δ = _mvr_delta!(
                    vnwork[tid],
                    ejwork[tid],
                    vdwork[tid],
                    storagework[tid],
                    L,
                    s,
                    j,
                    m,
                    λmin
                )
                abs(δ) < 1e-15 && continue
                s[j] += δ
                v = update_work[tid]
                fill!(v, zero(T))
                v[j] = sqrt(abs(δ))
                δ > 0 ? lowrankdowndate_turbo!(L, v) : lowrankupdate_turbo!(L, v)
                abs(δ) > local_max_delta && (local_max_delta = abs(δ))
            end
            max_deltas[tid] = max(max_deltas[tid], local_max_delta)
        end
        max_delta = maximum(max_deltas)
        if verbose
            obj = _mvr_objective(L, s, m)
            println("Iter $l: obj = $obj, δ = $max_delta, windows = $(length(windows))")
            flush(stdout)
        end
        max_delta < tol && break
    end
    if factor_check
        err = _assert_parallel_cholesky_factor(L, Σ, s, (m+1)/m, λmin, factor_check_tol=factor_check_tol)
        verbose && _print_parallel_factor_check(err, factor_check_tol)
    end
    return s[inverse_order]
end

"""
    forward_backward!(x, L, y, storage=zeros(length(x)))

Non-allocating solver for finding `x` to the solution of LL'x = y where L is a cholesky factor. 
"""
function forward_backward!(x, L, y, storage=zeros(length(x)))
    ldiv!(storage, UpperTriangular(L.factors)', y) # non-allocating version of ldiv!(storage, L.L, y)
    ldiv!(x, UpperTriangular(L.factors), storage) # non-allocating version of ldiv!(x, L.U, storage)
end

function solve_quadratic(cn, cd, Sjj, m, verbose=false)
    isfinite(cn) || return 0
    isfinite(cd) || return 0
    cd > 0 || return 0
    a = -cn - cd^2*m^2
    b = 2*(-cn*Sjj + cd*m^2)
    c = -cn*Sjj^2 - m^2
    a == c == 0 && return 0 # quick return; when a = c = 0, only solution is δ = 0
    lb, ub = -Sjj, inv(cd)
    if abs(a) ≤ eps(typeof(float(a))) * max(abs(b), abs(c), one(float(a)))
        abs(b) ≤ eps(typeof(float(b))) * max(abs(c), one(float(b))) && return 0
        x = -c / b
        return isfinite(x) && lb < x < ub ? x : 0
    end
    discriminant = b^2 - 4*a*c
    if discriminant < 0
        discriminant ≥ -sqrt(eps(typeof(float(discriminant)))) * max(abs(b)^2, abs(4*a*c), one(float(discriminant))) || return 0
        discriminant = zero(discriminant)
    end
    root = sqrt(discriminant)
    x1 = (-b + root) / (2a)
    x2 = (-b - root) / (2a)
    δj = isfinite(x1) && lb < x1 < ub ? x1 : x2
    isfinite(δj) || return 0
    lb < δj < ub || return 0
    verbose && println("-Sjj = $(-Sjj), inv(cd) = $(inv(cd)), x1 = $x1, x2 = $x2")
    return δj
end

"""
    solve_max_entropy(Σ::AbstractMatrix)

Solves the maximum entropy knockoff problem for fixed-X and model-X knockoffs
given correlation matrix Σ. Users should call `solve_s` instead of this function. 

# Reference
Algorithm 2.2 from Powerful Knockoffs via Minimizing Reconstructability: https://arxiv.org/pdf/2011.14625.pdf

# Note
There is a typo in algorithm for computing ME knockoffs in "Powerful knockoffs
via minimizing reconstructability" by Spector, Asher, and Lucas Janson (2020).
In the supplemental section, equation 59, they needed to evaluate 
`c_m = D^t_{-j,j}D^{-1}_{-j,-j}D_{-j,j}`. They claimed the FANOK paper 
("FANOK: KNOCKOFFS IN LINEAR TIME" by Askari et al. (2020)) implies that
`c_m = ||v_m||^2` where `Lv_m = u`. However, according to section A.1.2
of the FANOK paper, it seems like the actual update should be
`D^t_{-j,j}D^{-1}_{-j,-j}D_{-j,j} = ζ*||c_m||^2 / (ζ + ||c_m||^2)` 
where `ζ = 2Σ_{jj} - s_j`.
"""
function solve_max_entropy(
    Σ::AbstractMatrix{T}; # correlation matrix
    niter::Int = 100,
    tol=1e-3, # converges when changes in s are all smaller than tol
    λmin=1e-6, # minimum eigenvalue of S and (m+1)/m Σ - S
    m::Number = 1, # number of knockoffs per variable
    s_init = solve_equi(Σ, m=m) ./ 2, # initialize away from the equicorrelated boundary
    robust::Bool = false, # whether to use "robust" Cholesky updates (if robust=true, alg will be ~10x slower, only use this if the default causes cholesky updates to fail)
    verbose::Bool = false
    ) where T
    p = size(Σ, 1)
    downdate_margin = sqrt(eps(T))
    # whether to use robust cholesky updates or not
    cholupdate! = robust ? lowrankupdate! : lowrankupdate_turbo!
    choldowndate! = robust ? lowrankdowndate! : lowrankdowndate_turbo!
    # initialize s vector and compute initial cholesky factor
    s = copy(s_init)
    L = cholesky(Symmetric(Matrix((m+1)/m*Σ - Diagonal(s) + λmin*I), :U))
    obj = verbose ? _maxent_objective(L, s, m) : zero(T)
    verbose && println("Maxent initial obj = $obj")
    # preallocated vectors for efficiency
    x, ỹ = zeros(p), zeros(p)
    @inbounds for l in 1:niter
        max_delta = zero(T)
        obj_new = obj
        for j in 1:p
            @simd for i in 1:p
                ỹ[i] = (m+1)/m * Σ[i, j]
            end
            ỹ[j] = 0
            # compute x as the solution to L*x = ỹ
            ldiv!(x, UpperTriangular(L.factors)', ỹ) # non-allocating version of ldiv!(x, L.L, ỹ)
            x_l2sum = sum(abs2, x)
            # compute zeta and c as in alg 2.2 of askari et al
            ζ = (m+1)/m * Σ[j, j] - s[j]
            c = (ζ * x_l2sum) / (ζ + x_l2sum)
            # solve optimality condition in eq 75 of spector et al 2020
            sj_new = ((m+1)/m * Σ[j, j] - c) / 2
            # ensure new s[j] is in feasible region
            fill!(x, 0)
            x[j] = 1
            ldiv!(ỹ, UpperTriangular(L.factors)', x) # non-allocating version of ldiv!(ỹ, L.L, x)
            ub = max(zero(T), (1 - downdate_margin) / sum(abs2, ỹ) - λmin)
            δ = sj_new - s[j]
            δ > ub && (δ = ub)
            δ < -s[j] && (δ = -s[j])
            abs(δ) < 1e-15 && continue
            verbose && (obj_new += log(1 - δ * sum(abs2, ỹ)) + m * log1p(δ / s[j]))
            # update s
            s[j] += δ
            # rank 1 update to cholesky factor
            fill!(x, 0)
            x[j] = sqrt(abs(δ))
            δ > 0 ? choldowndate!(L, x) : cholupdate!(L, x)
            # update convergence tol
            abs(δ) > max_delta && (max_delta = abs(δ))
        end
        # declare convergence if changes in s are all smaller than tol
        if verbose
            println("Iter $l: obj = $obj_new, δ = $max_delta")
            flush(stdout)
        end
        obj = obj_new
        max_delta < tol && break 
    end
    return s
end

function _max_entropy_delta!(
    x::AbstractVector{T},
    ỹ::AbstractVector{T},
    Σ::AbstractMatrix{T},
    L,
    s::AbstractVector{T},
    j::Int,
    γ,
    λmin
    ) where T
    p = length(s)
    @inbounds @simd for i in 1:p
        ỹ[i] = γ * Σ[i, j]
    end
    ỹ[j] = zero(T)
    ldiv!(x, UpperTriangular(L.factors)', ỹ)
    x_l2sum = sum(abs2, x)
    ζ = γ * Σ[j, j] - s[j]
    c = (ζ * x_l2sum) / (ζ + x_l2sum)
    sj_new = (γ * Σ[j, j] - c) / 2

    fill!(x, zero(T))
    x[j] = one(T)
    ldiv!(ỹ, UpperTriangular(L.factors)', x)
    ub = max(zero(T), (1 - sqrt(eps(T))) / sum(abs2, ỹ) - λmin)
    δ = sj_new - s[j]
    δ > ub && (δ = ub)
    δ < -s[j] && (δ = -s[j])
    return δ
end

"""
    solve_max_entropy_parallel(Σ::AbstractMatrix; kwargs...)

Adaptive window-parallel version of [`solve_max_entropy`](@ref).
Users should call [`solve_s`](@ref) with `method=:maxent_fast` instead of
this function.

The solver reorders features using a nearest-correlation graph, partitions the
reordered variables at weak cross-correlation boundaries, and runs one serial
coordinate-descent sweep per window in parallel. The returned vector is sorted
back to the input feature order. A post-optimization consistency check verifies
that the maintained factor still represents the target knockoff matrix.

# Keyword arguments
+ `nworkers`: maximum number of coordinates proposed concurrently. Defaults to
  `Threads.nthreads()`. If this resolves to 1, the serial solver is used.
+ `feature_order`: optional permutation to use instead of automatic ordering.
+ `order_neighbors`: number of nearest-correlation neighbors used to build the
  automatic feature ordering graph.
+ `factor_check_tol`: relative tolerance for the final Cholesky consistency
  check.
"""
function solve_max_entropy_parallel(
    Σ::AbstractMatrix{T};
    niter::Int = 100,
    tol=1e-3,
    λmin=1e-6,
    m::Number = 1,
    s_init = solve_equi(Σ, m=m) ./ 2,
    verbose::Bool = false,
    nworkers::Int = Threads.nthreads(),
    feature_order::Union{Nothing, AbstractVector{Int}} = nothing,
    order_neighbors::Int = 8,
    boundary_band::Union{Nothing, Int} = nothing,
    min_window_size::Union{Nothing, Int} = nothing,
    window_corr_tol = 1e-3,
    factor_check::Bool = true,
    factor_check_tol = 1e-4
    ) where T
    p = size(Σ, 1)
    nworkers = max(1, min(nworkers, Threads.nthreads(), p))
    if nworkers == 1
        verbose && _print_parallel_serial_fallback("solve_max_entropy_parallel", nworkers)
        return solve_max_entropy(
            Σ;
            niter=niter,
            tol=tol,
            λmin=λmin,
            m=m,
            s_init=s_init,
            verbose=verbose
        )
    end

    Σinput = Σ
    Σ, s, order, inverse_order, windows, nworkers, boundary_band, min_window_size = _parallel_setup(
        Σ,
        s_init,
        nworkers,
        feature_order,
        order_neighbors,
        boundary_band=boundary_band,
        min_window_size=min_window_size,
        window_corr_tol=window_corr_tol
    )
    verbose && _print_parallel_setup_report(
        order,
        windows,
        nworkers,
        feature_order,
        boundary_band,
        min_window_size,
        window_corr_tol
    )
    if length(windows) == 1
        verbose && _print_parallel_single_window_fallback("solve_max_entropy_parallel")
        return solve_max_entropy(
            Σinput;
            niter=niter,
            tol=tol,
            λmin=λmin,
            m=m,
            s_init=s_init,
            verbose=verbose
        )
    end

    p = size(Σ, 1)
    L = cholesky(Symmetric(Matrix((m+1)/m*Σ - Diagonal(s) + λmin*I), :U))
    γ = (m+1) / m
    obj = verbose ? _maxent_objective(L, s, m) : zero(T)
    verbose && println("Maxent initial obj = $obj")

    nthread_buffers = Threads.maxthreadid()
    xwork = [zeros(T, p) for _ in 1:nthread_buffers]
    ywork = [zeros(T, p) for _ in 1:nthread_buffers]
    update_work = [zeros(T, p) for _ in 1:nthread_buffers]
    max_deltas = zeros(T, nthread_buffers)

    @inbounds for l in 1:niter
        fill!(max_deltas, zero(T))
        Threads.@threads for widx in eachindex(windows)
            tid = Threads.threadid()
            local_max_delta = zero(T)
            for j in windows[widx]
                δ = _max_entropy_delta!(
                    xwork[tid],
                    ywork[tid],
                    Σ,
                    L,
                    s,
                    j,
                    γ,
                    λmin
                )
                abs(δ) < 1e-15 && continue
                s[j] += δ
                v = update_work[tid]
                fill!(v, zero(T))
                v[j] = sqrt(abs(δ))
                δ > 0 ? lowrankdowndate_turbo!(L, v) : lowrankupdate_turbo!(L, v)
                abs(δ) > local_max_delta && (local_max_delta = abs(δ))
            end
            max_deltas[tid] = max(max_deltas[tid], local_max_delta)
        end
        max_delta = maximum(max_deltas)
        if verbose
            obj = _maxent_objective(L, s, m)
            println("Iter $l: obj = $obj, δ = $max_delta, windows = $(length(windows))")
            flush(stdout)
        end
        max_delta < tol && break
    end
    if factor_check
        err = _assert_parallel_cholesky_factor(L, Σ, s, γ, λmin, factor_check_tol=factor_check_tol)
        verbose && _print_parallel_factor_check(err, factor_check_tol)
    end
    return s[inverse_order]
end

"""
    solve_SDP(Σ::AbstractMatrix)

Solves the SDP problem for fixed-X and model-X knockoffs using coordinate descent, 
given correlation matrix Σ. Users should call `solve_s` instead of this function. 

# Reference
Algorithm 2.2 from "FANOK: Knockoffs in Linear Time" by Askari et al. (2020).
"""
function solve_SDP(
    Σ::AbstractMatrix{T};
    λ::T = 0.5, # barrier coefficient
    μ::T = 0.8, # decay parameter
    niter::Int = 100,
    m::Number = 1, # number of knockoffs per variable
    tol=1e-3, # converges when lambda < tol?
    λmin=1e-6, # minimum eigenvalue margin for (m+1)/m Σ - Diagonal(s)
    robust::Bool = false, # whether to use "robust" Cholesky updates (if robust=true, alg will be ~10x slower, only use this if the default causes cholesky updates to fail)
    verbose::Bool = false
    ) where T
    0 ≤ μ ≤ 1 || error("Decay parameter μ must be in [0, 1] but was $μ")
    0 < λ || error("Barrier coefficient λ must be > 0 but was $λ")
    # whether to use robust cholesky updates or not
    cholupdate! = robust ? lowrankupdate! : lowrankupdate_turbo!
    choldowndate! = robust ? lowrankdowndate! : lowrankdowndate_turbo!
    # initialize s vector and compute initial cholesky factor
    p = size(Σ, 1)
    downdate_margin = sqrt(eps(T))
    s = zeros(T, p)
    L = cholesky(Symmetric(Matrix((m+1)/m*Σ), :U))
    obj = verbose ? _sdp_objective(Σ, s) : zero(T)
    verbose && println("SDP initial obj = $obj")
    # preallocated vectors for efficiency
    x, ỹ = zeros(p), zeros(p)
    @inbounds for l in 1:niter
        obj_new = obj
        for j in 1:p
            @simd for i in 1:p
                ỹ[i] = (m+1)/m * Σ[i, j]
            end
            ỹ[j] = 0
            # compute c as the solution to L*x = ỹ
            ldiv!(x, UpperTriangular(L.factors)', ỹ) # non-allocating version of ldiv!(x, L.L, ỹ)
            x_l2sum = sum(abs2, x)
            # compute zeta and c as in alg 2.2 of askari et al
            ζ = (m+1)/m*Σ[j, j] - s[j]
            c = (ζ * x_l2sum) / (ζ + x_l2sum)
            # 1st order optimality condition
            sj_new = clamp((m+1)/m*Σ[j, j] - c - λ, 0, 1)
            δ = sj_new - s[j]
            δ > 0 && begin
                fill!(x, 0)
                x[j] = 1
                ldiv!(ỹ, UpperTriangular(L.factors)', x)
                ub = max(zero(T), (1 - downdate_margin) / sum(abs2, ỹ) - λmin)
                δ > ub && (δ = ub)
            end
            δ < -s[j] && (δ = -s[j])
            abs(δ) < 1e-15 && continue
            verbose && (obj_new += abs(Σ[j, j] - s[j] - δ) - abs(Σ[j, j] - s[j]))
            s[j] += δ
            # rank 1 update to cholesky factor
            fill!(x, 0)
            x[j] = sqrt(abs(δ))
            δ > 0 ? choldowndate!(L, x) : cholupdate!(L, x)
        end
        # check convergence 
        obj = obj_new
        if verbose
            println("Iter $l: λ = $λ, obj = $obj, sum(s) = $(sum(s))")
            flush(stdout)
        end
        λ *= μ
        λ < tol && break
    end
    return s
end

function _sdp_ccd_delta!(
    x::AbstractVector{T},
    ỹ::AbstractVector{T},
    Σ::AbstractMatrix{T},
    L,
    s::AbstractVector{T},
    j::Int,
    γ,
    λ,
    λmin
    ) where T
    p = length(s)
    @inbounds @simd for i in 1:p
        ỹ[i] = γ * Σ[i, j]
    end
    ỹ[j] = zero(T)
    ldiv!(x, UpperTriangular(L.factors)', ỹ)
    x_l2sum = sum(abs2, x)
    ζ = γ * Σ[j, j] - s[j]
    c = (ζ * x_l2sum) / (ζ + x_l2sum)
    sj_new = clamp(γ * Σ[j, j] - c - λ, zero(T), one(T))
    δ = sj_new - s[j]
    if δ > 0
        fill!(x, zero(T))
        x[j] = one(T)
        ldiv!(ỹ, UpperTriangular(L.factors)', x)
        ub = max(zero(T), (1 - sqrt(eps(T))) / sum(abs2, ỹ) - λmin)
        δ > ub && (δ = ub)
    end
    δ < -s[j] && (δ = -s[j])
    return δ
end

"""
    solve_sdp_parallel(Σ::AbstractMatrix; kwargs...)

Adaptive window-parallel version of [`solve_SDP`](@ref). Users should call
[`solve_s`](@ref) with `method=:sdp_parallel`.

The solver reorders features using a nearest-correlation graph, partitions the
reordered variables at weak cross-correlation boundaries, runs windows in
parallel, and sorts the optimized vector back to the input order. It errors if
the final factorization fails the Cholesky consistency check.
"""
function solve_sdp_parallel(
    Σ::AbstractMatrix{T};
    λ::T = 0.5,
    μ::T = 0.8,
    niter::Int = 100,
    m::Number = 1,
    tol=1e-3,
    λmin=1e-6,
    verbose::Bool = false,
    nworkers::Int = Threads.nthreads(),
    feature_order::Union{Nothing, AbstractVector{Int}} = nothing,
    order_neighbors::Int = 8,
    boundary_band::Union{Nothing, Int} = nothing,
    min_window_size::Union{Nothing, Int} = nothing,
    window_corr_tol = 1e-3,
    factor_check::Bool = true,
    factor_check_tol = 1e-4
    ) where T
    0 ≤ μ ≤ 1 || error("Decay parameter μ must be in [0, 1] but was $μ")
    0 < λ || error("Barrier coefficient λ must be > 0 but was $λ")
    p = size(Σ, 1)
    nworkers = max(1, min(nworkers, Threads.nthreads(), p))
    if nworkers == 1
        verbose && _print_parallel_serial_fallback("solve_sdp_parallel", nworkers)
        return solve_SDP(
            Σ;
            λ=λ,
            μ=μ,
            niter=niter,
            m=m,
            tol=tol,
            λmin=λmin,
            verbose=verbose
        )
    end

    Σinput = Σ
    s_init = zeros(T, p)
    Σ, s, order, inverse_order, windows, nworkers, boundary_band, min_window_size = _parallel_setup(
        Σ,
        s_init,
        nworkers,
        feature_order,
        order_neighbors,
        boundary_band=boundary_band,
        min_window_size=min_window_size,
        window_corr_tol=window_corr_tol
    )
    verbose && _print_parallel_setup_report(
        order,
        windows,
        nworkers,
        feature_order,
        boundary_band,
        min_window_size,
        window_corr_tol
    )
    if length(windows) == 1
        verbose && _print_parallel_single_window_fallback("solve_sdp_parallel")
        return solve_SDP(
            Σinput;
            λ=λ,
            μ=μ,
            niter=niter,
            m=m,
            tol=tol,
            λmin=λmin,
            verbose=verbose
        )
    end

    p = size(Σ, 1)

    L = cholesky(Symmetric(Matrix((m+1)/m*Σ), :U))
    γ = (m+1) / m
    obj = verbose ? _sdp_objective(Σ, s) : zero(T)
    verbose && println("SDP initial obj = $obj")

    nthread_buffers = Threads.maxthreadid()
    xwork = [zeros(T, p) for _ in 1:nthread_buffers]
    ywork = [zeros(T, p) for _ in 1:nthread_buffers]
    update_work = [zeros(T, p) for _ in 1:nthread_buffers]

    @inbounds for l in 1:niter
        Threads.@threads for widx in eachindex(windows)
            tid = Threads.threadid()
            for j in windows[widx]
                δ = _sdp_ccd_delta!(
                    xwork[tid],
                    ywork[tid],
                    Σ,
                    L,
                    s,
                    j,
                    γ,
                    λ,
                    λmin
                )
                abs(δ) < 1e-15 && continue
                s[j] += δ
                v = update_work[tid]
                fill!(v, zero(T))
                v[j] = sqrt(abs(δ))
                δ > 0 ? lowrankdowndate_turbo!(L, v) : lowrankupdate_turbo!(L, v)
            end
        end
        verbose && (obj = _sdp_objective(Σ, s))
        if verbose
            println("Iter $l: λ = $λ, obj = $obj, sum(s) = $(sum(s)), windows = $(length(windows))")
            flush(stdout)
        end
        λ *= μ
        λ < tol && break
    end
    if factor_check
        err = _assert_parallel_cholesky_factor(L, Σ, s, γ, zero(T), factor_check_tol=factor_check_tol)
        verbose && _print_parallel_factor_check(err, factor_check_tol)
    end
    return s[inverse_order]
end
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
+ `m`: Number of knockoffs per variable, defaults to 1. 
+ `kwargs`: Extra arguments available for specific methods. For example, to use 
    less stringent convergence tolerance, specify `tol = 0.001`.
    For a list of available options, see [`solve_group_mvr_hybrid`](@ref),
    [`solve_group_max_entropy_hybrid`](@ref), [`solve_group_sdp_hybrid`](@ref), or
    [`solve_group_equi`](@ref)

# Output
+ `S`: A matrix solved so that `(m+1)/m*Σ - S ⪰ 0` and `S ⪰ 0`
+ `γ`: A vector that is only non-empty for equi-correlated knockoff constructions.
    It stores the value of γ where `S_{gg} = γΣ_{gg}`.
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
        s = solve_s(Symmetric(Σcor), method; m=m, kwargs...)
        S = Diagonal(s) |> Matrix
        γs = T[]
        obj = zero(T)
    else
        # solve group knockoff optimization problem
        if method == :equi
            S, γs, obj = solve_group_equi(Σcor, group_permuted; m=m)
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

function group_sdp_objective_single_block(Σg::AbstractMatrix{T}, Sg::AbstractMatrix{T}) where T
    p = size(Σg, 1)
    size(Σg) == size(Sg) || error("group_sdp_objective_single_block: Expected size of Σg and Sg to be equal")
    obj = zero(T)
    for j in 1:p, i in 1:p
        obj += abs(Σg[i, j] - Sg[i, j])
    end
    return obj
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
                δ = (m*bjj-ajj) / ((m+1)*ajj*bjj)
                # ensure feasibility
                ub = 1 / ajj - ϵ
                lb = -1 / bjj + ϵ
                lb ≥ ub && continue
                δ = clamp(δ, lb, ub)
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
