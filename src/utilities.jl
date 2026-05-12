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

