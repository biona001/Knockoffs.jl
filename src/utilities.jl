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

function _validated_feature_order(feature_order, p::Int)
    length(feature_order) == p || error("feature_order must have length $p but had length $(length(feature_order)).")
    order = collect(Int, feature_order)
    seen = falses(p)
    for j in order
        1 ≤ j ≤ p || error("feature_order contains index $j, which is outside 1:$p.")
        seen[j] && error("feature_order must be a permutation, but index $j appears more than once.")
        seen[j] = true
    end
    return order
end

function _top_correlation_neighbors(Σ::AbstractMatrix, k::Int)
    p = size(Σ, 1)
    k = min(k, p - 1)
    neighbors = [Int[] for _ in 1:p]
    k == 0 && return neighbors

    @inbounds for i in 1:p
        best_scores = fill(-Inf, k)
        best_idx = fill(0, k)
        for j in 1:p
            i == j && continue
            score = abs(Σ[i, j])
            slot = argmin(best_scores)
            if score > best_scores[slot]
                best_scores[slot] = score
                best_idx[slot] = j
            end
        end
        for j in best_idx
            j == 0 && continue
            push!(neighbors[i], j)
            push!(neighbors[j], i)
        end
    end
    foreach(unique!, neighbors)
    return neighbors
end

function _locality_feature_order(Σ::AbstractMatrix, order_neighbors::Int)
    p = size(Σ, 1)
    p ≤ 2 && return collect(1:p)
    order_neighbors ≥ 1 || error("order_neighbors must be positive but was $order_neighbors.")

    neighbors = _top_correlation_neighbors(Σ, order_neighbors)
    degree = length.(neighbors)
    visited = falses(p)
    order = Int[]
    sizehint!(order, p)

    while length(order) < p
        start = 0
        best_degree = typemax(Int)
        for j in 1:p
            if !visited[j] && degree[j] < best_degree
                start = j
                best_degree = degree[j]
            end
        end

        queue = [start]
        visited[start] = true
        head = 1
        while head ≤ length(queue)
            v = queue[head]
            head += 1
            push!(order, v)
            nbrs = sort(neighbors[v], by = j -> degree[j])
            for u in nbrs
                if !visited[u]
                    visited[u] = true
                    push!(queue, u)
                end
            end
        end
    end

    return reverse(order)
end

function _parallel_feature_order(Σ::AbstractMatrix, feature_order, order_neighbors::Int)
    p = size(Σ, 1)
    return isnothing(feature_order) ?
        _locality_feature_order(Σ, order_neighbors) :
        _validated_feature_order(feature_order, p)
end

function _window_boundary_scores(Σ::AbstractMatrix, boundary_band::Int)
    p = size(Σ, 1)
    scores = zeros(eltype(Σ), max(p - 1, 0))
    @inbounds for b in 1:(p - 1)
        score = zero(eltype(Σ))
        ilo = max(1, b - boundary_band + 1)
        ihi = b
        jlo = b + 1
        jhi = min(p, b + boundary_band)
        for j in jlo:jhi, i in ilo:ihi
            score = max(score, abs(Σ[i, j]))
        end
        scores[b] = score
    end
    return scores
end

function _cut_is_spaced(cut::Int, cuts::Vector{Int}, p::Int, min_window_size::Int)
    cut < min_window_size && return false
    p - cut < min_window_size && return false
    for c in cuts
        abs(cut - c) < min_window_size && return false
    end
    return true
end

function _parallel_window_parameters(
    p::Int,
    nworkers::Int,
    boundary_band::Union{Nothing, Int},
    min_window_size::Union{Nothing, Int}
    )
    nwindows = max(1, min(nworkers, p))
    isnothing(boundary_band) && (boundary_band = max(1, min(p - 1, cld(p, 4nwindows))))
    boundary_band = max(1, min(boundary_band, p - 1))
    isnothing(min_window_size) && (min_window_size = max(1, cld(p, 4nwindows)))
    min_window_size = max(1, min(min_window_size, p))
    return boundary_band, min_window_size
end

function _parallel_windows(
    Σ::AbstractMatrix,
    nworkers::Int;
    boundary_band::Union{Nothing, Int}=nothing,
    min_window_size::Union{Nothing, Int}=nothing,
    window_corr_tol=1e-3
    )
    p = size(Σ, 1)
    p == 0 && return UnitRange{Int}[]
    p == 1 && return [1:1]

    nwindows = max(1, min(nworkers, p))
    boundary_band, min_window_size =
        _parallel_window_parameters(p, nwindows, boundary_band, min_window_size)

    scores = _window_boundary_scores(Σ, boundary_band)
    boundary_order = sortperm(scores)
    cuts = Int[]
    for b in boundary_order
        length(cuts) == nwindows - 1 && break
        scores[b] ≤ window_corr_tol || continue
        _cut_is_spaced(b, cuts, p, min_window_size) && push!(cuts, b)
    end

    sort!(cuts)
    windows = UnitRange{Int}[]
    lo = 1
    for cut in cuts
        push!(windows, lo:cut)
        lo = cut + 1
    end
    push!(windows, lo:p)
    return windows
end

function _parallel_setup(
    Σ::AbstractMatrix,
    s_init,
    nworkers::Int,
    feature_order,
    order_neighbors::Int;
    boundary_band::Union{Nothing, Int}=nothing,
    min_window_size::Union{Nothing, Int}=nothing,
    window_corr_tol=1e-3
    )
    p = size(Σ, 1)
    nworkers = max(1, min(nworkers, Threads.nthreads(), p))
    order = _parallel_feature_order(Σ, feature_order, order_neighbors)
    inverse_order = invperm(order)
    Σordered = Matrix(Σ[order, order])
    sordered = collect(s_init[order])
    boundary_band, min_window_size =
        _parallel_window_parameters(p, nworkers, boundary_band, min_window_size)
    windows = _parallel_windows(
        Σordered,
        nworkers,
        boundary_band=boundary_band,
        min_window_size=min_window_size,
        window_corr_tol=window_corr_tol
    )
    return Σordered, sordered, order, inverse_order, windows, nworkers, boundary_band, min_window_size
end

function _print_parallel_setup_report(
    order::AbstractVector{Int},
    windows::AbstractVector{<:UnitRange{Int}},
    nworkers::Int,
    feature_order,
    boundary_band::Int,
    min_window_size::Int,
    window_corr_tol
    )
    p = length(order)
    reordered = any(order[j] != j for j in 1:p)
    ordering_source = isnothing(feature_order) ? "automatic nearest-correlation ordering" : "user-supplied feature_order"
    ordering_result = reordered ? "features were reordered" : "original feature order was kept"

    println("Adaptive window-parallel setup:")
    println("  Feature ordering: $ordering_source; $ordering_result.")
    println("  Approximately independent windows: $(length(windows))")
    println("  Julia threads available: $(Threads.nthreads())")
    println("  Worker threads used: $nworkers")
    println("  Boundary band: $boundary_band")
    println("  Minimum window size: $min_window_size")
    println("  Window correlation tolerance: $window_corr_tol")
    if isempty(windows)
        println("  Window ranges: none")
    else
        ranges = join(("$(first(w)):$(last(w))" for w in windows), ", ")
        sizes = join((string(length(w)) for w in windows), ", ")
        println("  Window ranges in ordered coordinates: $ranges")
        println("  Window sizes: $sizes")
    end
    flush(stdout)
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
    simulate_AR1(p::Int, a=1, b=1, tol=1e-3, max_corr=1, rho=nothing)

Generates `p`-dimensional correlation matrix for
AR(1) Gaussian process, where successive correlations
are drawn from Beta(`a`,`b`) independently. If `rho` is
specified, then the process is stationary with correlation
`rho`.

# Source
https://github.com/amspector100/knockpy/blob/20eddb3eb60e0e82b206ec989cb936e3c3ee7939/knockpy/dgp.py#L61
"""
function simulate_AR1(p::Int; a=1, b=1, tol=1e-3, max_corr=1, rho=nothing)
    # Generate rhos, take log to make multiplication easier
    d = Beta(a, b)
    if isnothing(rho)
        rhos = log.(clamp!(rand(d, p), 0, max_corr))
    else
        abs(rho) > 1 && error("rho must be a correlation between -1 and 1")
        rhos = log.([rho for _ in 1:p])
    end
    rhos[1] = 0

    # Log correlations between x_1 and x_i for each i
    cumrhos = cumsum(rhos)

    # Use cumsum tricks to calculate all correlations
    log_corrs = -1 * abs.(cumrhos .- cumrhos')
    corr_matrix = exp.(log_corrs)

    # Ensure PSD-ness
    shifted = shift_until_PSD!(corr_matrix, tol)
    corr_matrix = cov2cor(shifted, sqrt.(diag(shifted)))

    return corr_matrix
end

"""
    simulate_ER(p::Int; [invert])

Simulates a covariance matrix from a clustered Erdos-Renyi graph, which is
a block diagonal matrix where each block is an Erdo-Renyi graph. The result is
scaled back to a correlation matrix. 

For details, see the 4th simulation routine in section 5.1 of Li and Maathius 
https://academic.oup.com/jrsssb/article/83/3/534/7056103?login=false

# Inputs
+ `p`: Dimension of covariance matrix
+ `ϕ`: Probability of forming an edge between any 2 nodes
+ `lb`: lower bound for the value of an edge (drawn from uniform distribution)
+ `ub`: upper bound for the value of an edge (drawn from uniform distribution)
+ `invert`: Whther to invert the covariance matrix (to obtain the precision)
+ `λmin`: minimum eigenvalue of the resulting covariance matrix
+ `blocksize`: Number of variables within each ER graph. 
"""
function simulate_ER(p::Int; ϕ=0.1, lb=0.3, ub=0.9, λmin=0.1, blocksize = 10, invert::Bool=false)
    V = zeros(p, p)
    b = Bernoulli(ϕ)
    u = Uniform(lb, ub)
    for j in 1:p, i in j:p
        if i == j
            V[i, j] = 1
        elseif i - j > blocksize
            continue
        else
            V[i, j] = rand(-1:2:1) * rand(b) * rand(u)
        end
    end
    LinearAlgebra.copytri!(V, 'L')
    λ = Symmetric(V) |> eigmin |> abs
    Σ = V + (λmin + λ)*I
    invert && (Σ = inv(Σ))
    cov2cor!(Σ, sqrt.(diag(Σ)))
    return Σ
end

"""
    shift_until_PSD!(Σ::AbstractMatrix)

Keeps adding λI to Σ until the minimum eigenvalue > tol
"""
function shift_until_PSD!(Σ::AbstractMatrix, tol=1e-4)
    while eigmin(Symmetric(Σ)) ≤ tol
        Σ += tol*I
    end
    return Σ
end

"""
    normalize_col!(X::AbstractVecOrMat, [center=false])

Normalize each column of `X` to have unit Euclidean norm, optionally after centering.
"""
function normalize_col!(X::AbstractVecOrMat; center::Bool=false)
    @inbounds for x in eachcol(X)
        μi = center ? mean(x) : zero(eltype(X))
        @simd for i in eachindex(x)
            x[i] -= μi
        end
        xnorm = norm(x)
        iszero(xnorm) && error("normalize_col!: cannot normalize a zero-norm column.")
        @simd for i in eachindex(x)
            x[i] /= xnorm
        end
    end
    return X
end
normalize_col(X; center::Bool=false) = normalize_col!(copy(X), center=center)

function sample_DMC(q, Q; n=1)
    p = size(Q, 3)
    d = Categorical(q)
    X = zeros(Int, n, p)
    for i in 1:n
        X[i, 1] = rand(d)
        for j in 2:p
            d.p .= @view(Q[X[i, j-1], :, j])
            X[i, j] = rand(d)
        end
    end
    return X
end

"""
    simulate_block_covariance(groups, ρ, γ, num_v, w)

Simulates a block covariance matrix similar to the one in `Dai & Barber 2016, 
The knockoff filter for FDR control in group-sparse and multitask regression`. 
That is, all diagonal elements will be 1, correlation within groups will be `ρ`,
and correlation between groups will be `ρ*γ`. 

# Inputs
+ `groups`: Vector of group membership
+ `ρ`: within group correlation 
+ `γ`: between group correlation

# Optional arguments
+ `num_v`: Number of added rank 1 update `Σ + v1*v1' + ... + vn*vn'` where `v` 
    is iid `N(0, w)` (default 0)
+ `w`: variance of the rank 1 update used in `num_v` (default 1)
"""
function simulate_block_covariance(
    groups::Vector{Int},
    ρ::T, # within group correlation 
    γ::T; # between group correlation
    num_v::Int=0, # adds a rank 1 update Σ + v1*v1' + ... + vn*vn' where v is iid N(0, w)
    w::T = one(T)
    ) where T <: AbstractFloat
    issorted(groups) || error("groups needs to be a sorted vector (i.e. continuous)")
    # form block diagonals to handle within group correlation
    Σ = Matrix{Float64}[]
    for g in unique(groups)
        cnt = count(x -> x == g, groups)
        Σg = (1-ρ) * Matrix(I, cnt, cnt) + ρ * ones(cnt, cnt)
        push!(Σ, Σg)
    end
    Σ = BlockDiagonal(Σ) |> Matrix
    for i in 1:num_v
        v = rand(Normal(0, w), length(groups))
        BLAS.ger!(one(T), v, v, Σ) # Σ = Σ + vv'
    end
    # now add between group correlation
    Σ[findall(iszero, Σ)] .= γ*ρ
    # rescale to correlation matrix
    cov2cor!(Σ, diag(Σ))
    return Σ
end

const _TURBO_ROTATION_TAIL_LENGTH = 10

_turbo_rotation_tol(::Type{T}) where T = 10eps(T)

function _negligible_rotation_tail(s, i::Int, idx_end::Int, tail_length::Int, ::Type{T}) where T
    if i ≥ idx_end && abs(s) ≤ _turbo_rotation_tol(T)
        tail_length += 1
    else
        tail_length = 0
    end
    return tail_length, tail_length ≥ _TURBO_ROTATION_TAIL_LENGTH
end

"""
    lowrankupdate_turbo!(C::Cholesky, v::AbstractVector)

Vectorized version of lowrankupdate!, source https://github.com/JuliaLang/julia/blob/742b9abb4dd4621b667ec5bb3434b8b3602f96fd/stdlib/LinearAlgebra/src/cholesky.jl#L707
Takes advantage of the fact that `v` is 0 everywhere except at 1 position
"""
function lowrankupdate_turbo!(C::Cholesky{T}, v::AbstractVector) where T <: AbstractFloat
    A = C.factors
    n = length(v)
    if size(C, 1) != n
        throw(DimensionMismatch("updating vector must fit size of factorization"))
    end
    # if C.uplo == 'U'
    #     conj!(v)
    # end

    tail_length = 0
    idx_start = findfirst(!iszero, v)
    isnothing(idx_start) && return C
    idx_end = something(findlast(!iszero, v))
    @inbounds for i = idx_start:n

        # Compute Givens rotation
        c, s, r = LinearAlgebra.givensAlgorithm(A[i,i], v[i])

        # The sparse-update tail is negligible once several consecutive
        # rotations are indistinguishable from the identity at machine scale.
        tail_length, should_stop = _negligible_rotation_tail(s, i, idx_end, tail_length, T)
        should_stop && break

        # Store new diagonal element
        A[i,i] = r

        # Update remaining elements in row/column
        if C.uplo == 'U'
            @turbo for j = i + 1:n
                Aij = A[i,j]
                vj  = v[j]
                A[i,j]  =   c*Aij + s*vj
                v[j]    = -s*Aij + c*vj
            end
        else
            @turbo for j = i + 1:n
                Aji = A[j,i]
                vj  = v[j]
                A[j,i]  =   c*Aji + s*vj
                v[j]    = -s*Aji + c*vj
            end
        end
    end
    return C
end

"""
    lowrankdowndate_turbo!(C::Cholesky, v::AbstractVector)

Vectorized version of lowrankdowndate!, source https://github.com/JuliaLang/julia/blob/742b9abb4dd4621b667ec5bb3434b8b3602f96fd/stdlib/LinearAlgebra/src/cholesky.jl#L753
Takes advantage of the fact that `v` is 0 everywhere except at 1 position
"""
function lowrankdowndate_turbo!(C::Cholesky{T}, v::AbstractVector) where T <: AbstractFloat
    A = C.factors
    n = length(v)
    if size(C, 1) != n
        throw(DimensionMismatch("updating vector must fit size of factorization"))
    end
    # if C.uplo == 'U'
    #     conj!(v)
    # end

    tail_length = 0
    idx_start = findfirst(!iszero, v)
    isnothing(idx_start) && return C
    idx_end = something(findlast(!iszero, v))
    @inbounds for i = idx_start:n

        Aii = A[i,i]

        # Compute Givens rotation
        s = v[i] / Aii
        s2 = abs2(s)
        if s2 > 1
            throw(LinearAlgebra.PosDefException(i))
        end
        c = sqrt(1 - abs2(s))

        # The sparse-update tail is negligible once several consecutive
        # rotations are indistinguishable from the identity at machine scale.
        tail_length, should_stop = _negligible_rotation_tail(s, i, idx_end, tail_length, T)
        should_stop && break

        # Store new diagonal element
        A[i,i] = c*Aii

        # Update remaining elements in row/column
        if C.uplo == 'U'
            @turbo for j = i + 1:n
                vj = v[j]
                Aij = (A[i,j] - s*vj)/c
                A[i,j] = Aij
                v[j] = -s*Aij + c*vj
            end
        else
            @turbo for j = i + 1:n
                vj = v[j]
                Aji = (A[j,i] - s*vj)/c
                A[j,i] = Aji
                v[j] = -s*Aji + c*vj
            end
        end
    end
    return C
end
