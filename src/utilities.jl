"""
    solve_s(Σ::Symmetric, method::Symbol; m=1, kwargs...)

Solves the vector `s` for generating knockoffs. `Σ` can be a general 
covariance matrix but it must be wrapped in the `Symmetric` keyword. 

# Inputs
+ `Σ`: A covariance matrix (one must wrap `Symmetric(Σ)` explicitly)
+ `method`: Can be one of the following
    * `:mvr` for minimum variance-based reconstructability knockoffs (alg 1 in ref 2)
    * `:maxent` for maximum entropy knockoffs (alg 2 in ref 2)
    * `:equi` for equi-distant knockoffs (eq 2.3 in ref 1), 
    * `:sdp` for SDP knockoffs (eq 2.4 in ref 1)
    * `:sdp_ccd` fast SDP knockoffs via coordiate descent (alg 2.2 in ref 3)
+ `m`: Number of knockoffs per variable, defaults to 1. 
+ `kwargs`: Extra arguments available for specific methods. For example, to use 
    less stringent convergence tolerance for MVR knockoffs, specify `tol = 0.001`.
    For a list of available options, see [`solve_MVR`](@ref),
    [`solve_max_entropy`](@ref), [`solve_sdp_ccd`](@ref), [`solve_sdp`](@ref), or
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
    elseif method == :sdp
        s = solve_SDP(Σcor; m=m)
    elseif method == :mvr
        s = solve_MVR(Σcor; m=m, kwargs...)
    elseif method == :maxent
        s = solve_max_entropy(Σcor; m=m, kwargs...)
    elseif method == :sdp_ccd
        s = solve_sdp_ccd(Σcor; m=m, kwargs...)
    else
        error("Method must be one of $SINGLE_KNOCKOFFS but was $method")
    end
    # rescale s back to the result for a covariance matrix   
    iscor || (s .*= σs.^2)
    return s
end

"""
    solve_SDP(Σ::AbstractMatrix)

Solves the SDP problem for fixed-X and model-X knockoffs given correlation matrix Σ. 
Users should call `solve_s` instead of this function. 

The optimization problem is stated in equation 3.13 of
https://arxiv.org/pdf/1610.02351.pdf

# Arguments
+ `Σ`: A correlation matrix (diagonals all equal to 1)
+ `m`: Number of knockoffs to generate, defaults to 1
+ `optm`: SDP solver. Defaults to `Hypatia.Optimizer(verbose=false)`. This can
    be any solver that supports the JuMP interface. For example, use 
    `SDPT3.Optimizer` in SDPT3.jl package (which is a MATLAB dependency)
    for the best performance. 
"""
function solve_SDP(
    Σ::AbstractMatrix; # correlation matrix
    m::Number = 1, # number of multiple knockoffs to generate
    optm=Hypatia.Optimizer(verbose=false) # Any solver compatible with JuMP
    )
    # Build model via JuMP
    p = size(Σ, 1)
    model = Model(() -> optm)
    @variable(model, 0 ≤ s[i = 1:p] ≤ 1)
    @objective(model, Max, sum(s))
    @constraint(model, Symmetric((m+1)/m*Σ - diagm(s[1:p])) in PSDCone())
    # Solve optimization problem
    JuMP.optimize!(model)
    # Retrieve solution
    return clamp!(JuMP.value.(s), 0, 1)
end

# this uses Convex.jl
# function solve_SDP(Σ::AbstractMatrix)
#     svar = Variable(size(Σ, 1), Convex.Positive())
#     add_constraint!(svar, svar ≤ 1)
#     constraint = 2*Symmetric(Σ) - diagm(svar) in :SDP
#     problem = maximize(sum(svar), constraint)
#     solve!(problem, Hypatia.Optimizer; silent_solver=true)
#     s = clamp.(evaluate(svar), 0, 1) # make sure s_j ∈ (0, 1)
#     return s
# end

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
    tol=1e-6, # converges when changes in s are all smaller than tol
    λmin=1e-6, # minimum eigenvalue of S and (m+1)/m Σ - S
    m::Number = 1, # number of knockoffs per variable
    s_init = solve_equi(Σ, m=m), # initialize s vector with equicorrelated solution
    robust::Bool = false, # whether to use "robust" Cholesky updates (if robust=true, alg will be ~10x slower, only use this if the default causes cholesky updates to fail)
    verbose::Bool = false
    ) where T
    p = size(Σ, 1)
    # whether to use robust cholesky updates or not
    cholupdate! = robust ? lowrankupdate! : lowrankupdate_turbo!
    choldowndate! = robust ? lowrankdowndate! : lowrankdowndate_turbo!
    # initialize s vector and compute initial cholesky factor
    s = copy(s_init)
    L = cholesky(Symmetric((m+1)/m*Σ - Diagonal(s)) + λmin*I)
    # preallocated vectors for efficiency
    vn, ej, vd, storage = zeros(p), zeros(p), zeros(p), zeros(p)
    @inbounds for l in 1:niter
        max_delta = zero(T)
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
            ub = 1 / cd - λmin
            δj > ub && (δj = ub)
            abs(δj) < 1e-15 && continue
            s[j] += δj
            # rank 1 update to cholesky factor
            ej[j] = sqrt(abs(δj))
            δj > 0 ? choldowndate!(L, ej) : cholupdate!(L, ej)
            # update convergence tol
            abs(δj) > max_delta && (max_delta = abs(δj))
        end
        verbose && println("Iter $l: δ = $max_delta")
        # declare convergence if changes in s are all smaller than tol
        max_delta < tol && break
    end
    return s
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
    a = -cn - cd^2*m^2
    b = 2*(-cn*Sjj + cd*m^2)
    c = -cn*Sjj^2 - m^2
    a == c == 0 && return 0 # quick return; when a = c = 0, only solution is δ = 0
    x1 = (-b + sqrt(b^2 - 4*a*c)) / (2a)
    x2 = (-b - sqrt(b^2 - 4*a*c)) / (2a)
    δj = -Sjj < x1 < inv(cd) ? x1 : x2
    isinf(δj) && error("δj is Inf, aborting. Sjj = $Sjj, cn = $cn, cd = $cd, x1 = $x1, x2 = $x2")
    isnan(δj) && error("δj is NaN, aborting. Sjj = $Sjj, cn = $cn, cd = $cd, x1 = $x1, x2 = $x2")
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
    tol=1e-6, # converges when changes in s are all smaller than tol
    λmin=1e-6, # minimum eigenvalue of S and (m+1)/m Σ - S
    m::Number = 1, # number of knockoffs per variable
    s_init = solve_equi(Σ, m=m), # initialize s vector with equicorrelated solution
    robust::Bool = false, # whether to use "robust" Cholesky updates (if robust=true, alg will be ~10x slower, only use this if the default causes cholesky updates to fail)
    verbose::Bool = false
    ) where T
    p = size(Σ, 1)
    # whether to use robust cholesky updates or not
    cholupdate! = robust ? lowrankupdate! : lowrankupdate_turbo!
    choldowndate! = robust ? lowrankdowndate! : lowrankdowndate_turbo!
    # initialize s vector and compute initial cholesky factor
    s = copy(s_init)
    L = cholesky(Symmetric((m+1)/m*Σ - Diagonal(s)) + λmin*I)
    # preallocated vectors for efficiency
    x, ỹ = zeros(p), zeros(p)
    @inbounds for l in 1:niter
        max_delta = zero(T)
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
            ub = 1 / sum(abs2, ỹ) - λmin
            δ = sj_new - s[j]
            δ > ub && (δ = ub)
            abs(δ) < 1e-15 && continue
            # update s
            s[j] = sj_new
            # rank 1 update to cholesky factor
            fill!(x, 0)
            x[j] = sqrt(abs(δ))
            δ > 0 ? choldowndate!(L, x) : cholupdate!(L, x)
            # update convergence tol
            abs(δ) > max_delta && (max_delta = abs(δ))
        end
        # declare convergence if changes in s are all smaller than tol
        verbose && println("Iter $l: δ = $max_delta")
        max_delta < tol && break 
    end
    return s
end

"""
    solve_sdp_ccd(Σ::AbstractMatrix)

Solves the SDP problem for fixed-X and model-X knockoffs using coordinate descent, 
given correlation matrix Σ. Users should call `solve_s` instead of this function. 

# Reference
Algorithm 2.2 from "FANOK: Knockoffs in Linear Time" by Askari et al. (2020).
"""
function solve_sdp_ccd(
    Σ::AbstractMatrix{T};
    λ::T = 0.5, # barrier coefficient
    μ::T = 0.8, # decay parameter
    niter::Int = 100,
    m::Number = 1, # number of knockoffs per variable
    tol=1e-6, # converges when lambda < tol?
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
    s = zeros(T, p)
    L = cholesky(Symmetric((m+1)/m*Σ))
    # preallocated vectors for efficiency
    x, ỹ = zeros(p), zeros(p)
    @inbounds for l in 1:niter
        verbose && println("Iter $l: λ = $λ, sum(s) = $(sum(s))")
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
            δ = s[j] - sj_new
            abs(δ) < 1e-15 && continue
            s[j] = sj_new
            # rank 1 update to cholesky factor
            fill!(x, 0)
            x[j] = sqrt(abs(δ))
            δ > 0 ? cholupdate!(L, x) : choldowndate!(L, x)
        end
        # check convergence 
        λ *= μ
        λ < tol && break
    end
    return s
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

Normalize each column of `X` so they sum to 1. 
"""
function normalize_col!(X::AbstractVecOrMat; center::Bool=false)
    @inbounds for x in eachcol(X)
        μi = center ? mean(x) : zero(eltype(X))
        xnorm = norm(x)
        @simd for i in eachindex(x)
            x[i] = (x[i] - μi) / xnorm
        end
    end
    return X
end
normalize_col(X) = normalize_col!(copy(X))

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
    download_1000genomes(; chr="all", outdir=Knockoffs.datadir())

Downloads the 1000 genomes phase 3 reference panel in VCF format. Each chromosome
is separated into different VCF files accompanied by a tabix index file. By default, 
data will be saved in a folder called "1000genomes" in the Knockoffs package 
directory, i.e. in `Knockoffs.datadir()`.
"""
function download_1000genomes(; chr="all", outdir=Knockoffs.datadir())
    link = "https://bochet.gcc.biostat.washington.edu/beagle/1000_Genomes_phase3_v5a/b37.vcf"
    outpath = joinpath(outdir, "1000genomes")
    isdir(outpath) || mkdir(outpath)
    if chr == "all"
        for chr in vcat(string.(1:22), "X")
            vcffile = "chr$chr.1kg.phase3.v5a.vcf.gz"
            tabixfile = "chr$chr.1kg.phase3.v5a.vcf.gz.tbi"
            Downloads.download(joinpath(link, vcffile), joinpath(outpath, vcffile))
            Downloads.download(joinpath(link, tabixfile), joinpath(outpath, tabixfile))
        end
    else
        vcffile = "chr$chr.1kg.phase3.v5a.vcf.gz"
        tabixfile = "chr$chr.1kg.phase3.v5a.vcf.gz.tbi"
        Downloads.download(joinpath(link, vcffile), joinpath(outpath, vcffile))
        Downloads.download(joinpath(link, tabixfile), joinpath(outpath, tabixfile))
    end
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

    early_term = 0
    idx_start = something(findfirst(!iszero, v))
    idx_end = something(findlast(!iszero, v))
    @inbounds for i = idx_start:n

        # Compute Givens rotation
        c, s, r = LinearAlgebra.givensAlgorithm(A[i,i], v[i])

        # check for early termination
        if abs(s) < 1e-15 && i ≥ idx_end
            early_term += 1
            early_term > 10 && break
        else
            early_term = 0
        end

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

    early_term = 0
    idx_start = something(findfirst(!iszero, v))
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

        # check for early termination
        if abs(s) < 1e-15 && i ≥ idx_end
            early_term += 1
            early_term > 10 && break
        else
            early_term = 0
        end

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

"""
    check_model_solution(model; verbose=false)

After solving a JuMP model, checks if the solution is accurate. 
"""
function check_model_solution(model; verbose=false)
    success = true
    if termination_status(model) == OPTIMAL
        verbose && println("Solution is optimal")
    elseif termination_status(model) == LOCALLY_SOLVED
        verbose && println("Solution is locally optimal")
    elseif termination_status(model) == ALMOST_OPTIMAL
        verbose && println("Solution is almost optimal")
    elseif termination_status(model) == TIME_LIMIT && has_values(model)
        verbose && println("Solution is suboptimal due to a time limit, but a primal solution is available")
    else
        success = false
        verbose && println("The model was not solved correctly.")
    end
    return success
end
