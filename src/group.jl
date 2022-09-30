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
function solve_group_equi(Σ::AbstractMatrix, Σblocks::BlockDiagonal)
    Db = Matrix{eltype(Σ)}[]
    for Σbi in Σblocks.blocks
        push!(Db, inverse_mat_sqrt(Symmetric(Σbi)))
    end
    Db = BlockDiagonal(Db)
    λmin = Symmetric(Db * Σ * Db) |> eigmin
    γ = min(1, 2λmin)
    S = BlockDiagonal(γ .* Σblocks.blocks)
    return S, [γ]
end

"""
Solves the SDP group knockoff problem using analogy to the equi-correlated
group knockoffs. Basically, the idea is to optimize a vector `γ` where `γ[j]` 
multiplies Σ_jj. In the equi-correlated setting, all `γ[j]` is forced to be equal.

Details can be found in
Dai & Barber 2016, The knockoff filter for FDR control in group-sparse and multitask regression
"""
function solve_group_SDP(Σ::AbstractMatrix, Σblocks::BlockDiagonal)
    model = Model(() -> Hypatia.Optimizer(verbose=false))
    # model = Model(() -> SCS.Optimizer())
    n = nblocks(Σblocks)
    block_sizes = size.(Σblocks.blocks, 1)
    @variable(model, 0 <= γ[1:n] <= 1)
    blocks = BlockDiagonal([γ[i] * Σblocks.blocks[i] for i in 1:n]) |> Matrix
    @objective(model, Max, block_sizes' * γ)
    @constraint(model, Symmetric(2Σ - blocks) in PSDCone())
    JuMP.optimize!(model)
    check_model_solution(model)
    γs = clamp!(JuMP.value.(γ), 0, 1)
    S = BlockDiagonal(γs .* Σblocks.blocks)
    return S, γs
end

# equicorrelated construction by choosing S_g = γΣ_{g,g}
function solve_group_max_entropy_equi(Σ::AbstractMatrix, Σblocks::BlockDiagonal)
    p = size(Σ, 1)
    # calculate Db = bdiag(Σ_{11}^{-1/2}, ..., Σ_{GG}^{-1/2})
    Db = Matrix{eltype(Σ)}[]
    for Σbi in Σblocks.blocks
        push!(Db, inverse_mat_sqrt(Symmetric(Σbi)))
    end
    Db = BlockDiagonal(Db)
    λ = Symmetric(2Db * Σ * Db) |> eigvals
    # solve non-linear objective using Ipopt
    model = Model(() -> Ipopt.Optimizer())
    set_optimizer_attribute(model, "print_level", 0)
    @variable(model, 0 <= γ <= minimum(λ))
    @NLobjective(model, Max, p * log(γ) + sum(log(λ[i] - γ) for i in 1:p))
    JuMP.optimize!(model)
    check_model_solution(model, verbose=false)
    # convert solution to vector and return resulting block diagonal matrix
    γs = [JuMP.value.(γ)]
    S = BlockDiagonal(γs[1] .* Σblocks.blocks)
    return S, γs
end

# SDP construction by choosing S_g = γ_g * Σ_{g,g}
function solve_group_max_entropy_sdp(Σ::AbstractMatrix, Σblocks::BlockDiagonal)
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
    check_model_solution(model)
    # convert solution to vector and return resulting block diagonal matrix
    γs = convert(Vector{Float64}, clamp!(JuMP.value.(γ), 0, 1))
    S = BlockDiagonal(γs[1] .* Σblocks.blocks)
    return S, γs
end

function solve_group_SDP_full(Σ::AbstractMatrix, Σblocks::BlockDiagonal)
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
    @constraint(model, Symmetric(2Σ - S) in PSDCone())
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
    Sblocks::BlockDiagonal, 
    niter::Int = 100,
    tol=1e-6, # converges when changes in s are all smaller than tol
    m::Int = 1, # number of knockoffs per variable
    verbose::Bool = false
    ) where T
    p = size(Σ, 1)
    blocks = nblocks(Sblocks)
    group_sizes = size.(Sblocks.blocks, 1)
    # initialize S matrix and compute initial cholesky factor
    S, _ = solve_group_equi(Σ, Sblocks)
    S = convert(Matrix{T}, S)
    L = cholesky(Symmetric(2Σ - S))
    C = cholesky(Symmetric(S))

                    # # check equality is maintained
                    # sigma = 2Σ - C.L * C.L' # 2Σ - S
                    # LLt = L.L * L.L' # 2Σ - S
                    # @show sigma[1:5, 1:5]
                    # @show LLt[1:5, 1:5]

    # preallocated vectors for efficiency
    vn, ej, vd, storage = zeros(p), zeros(p), zeros(p), zeros(p)
    u, v, ei = zeros(p), zeros(p), zeros(p)
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
                forward_backward!(vn, L, ej, storage) # solves L*L'*vn = ej for vn
                cn = -sum(abs2, vn)
                # find vd as the solution to L*vd = ej
                ldiv!(vd, UpperTriangular(L.factors)', ej) # non-allocating version of ldiv!(vd, L.L, ej)
                cd = sum(abs2, vd)
                # solve quadratic optimality condition in eq 71
                δj = solve_quadratic(cn, cd, S[j, j])
                abs(δj) < 1e-15 && continue
                S[j, j] += δj

                @show 
                @show eigmin(S)
                # rank 1 update to cholesky factor
                ej[j] = sqrt(abs(δj))
                if δj > 0
                    # lowrankdowndate_turbo!(L, ej)
                    # lowrankupdate_turbo!(C, ej)
                    lowrankdowndate!(L, ej)
                    fill!(ej, 0)
                    ej[j] = sqrt(abs(δj))
                    lowrankupdate!(C, ej)


                    # check equality is maintained
                    # sigma = 2Σ - C.L * C.L' # 2Σ - S
                    # LLt = L.L * L.L' # 2Σ - S
                    # @show sigma[1:5, 1:5]
                    # @show LLt[1:5, 1:5]
                else
                    lowrankupdate!(L, ej)
                    fill!(ej, 0)
                    ej[j] = sqrt(abs(δj))
                    lowrankdowndate!(C, ej)
                end
                # update convergence tol
                abs(δj) > max_delta && (max_delta = abs(δj))
            end
            # @show eigvals(Symmetric(S)) |> minimum
            # fdsa
            #
            # optimize off-diagonal entries
            #
            # for idx1 in 1:group_sizes[b], idx2 in idx1+1:group_sizes[b]
            #     j, i = idx1 + offset, idx2 + offset
            #     fill!(ej, 0)
            #     fill!(ei, 0)
            #     ej[j], ei[i] = 1, 1
            #     # compute cn, cd
            #     forward_backward!(u, L, ei, storage) # solves L*L'*u = ej for u
            #     forward_backward!(v, L, ej, storage) # solves L*L'*v = ej for v
            #     cn = dot(u, v)
            #     ldiv!(v, UpperTriangular(L.factors)', ej) # non-allocating version of ldiv!(v, L.L, ej)
            #     ldiv!(u, UpperTriangular(L.factors)', ei) # non-allocating version of ldiv!(u, L.L, ej)
            #     cd = dot(u, v)
            #     # compute kn, kd
            #     forward_backward!(u, C, ei, storage)
            #     forward_backward!(v, C, ej, storage)
            #     kn = dot(u, v)
            #     ldiv!(v, UpperTriangular(C.factors)', ej)
            #     ldiv!(u, UpperTriangular(C.factors)', ei)
            #     kd = dot(u, v)
            #     @show cn, cd, kn, kd
            #     # 1st order optimality condition
            #     δ = solve_group_quadratic(cn, cd, kn, kd)
            #     @show δ
            #     abs(δ) < 1e-15 && continue
            #     S[i, j] += δ
            #     S[j, i] += δ
            #     # update cholesky factor via 3 rank-1 updates
            #     fill!(storage, 0); fill!(ei, 0); fill!(ej, 0)
            #     storage[i] = storage[j] = ei[i] = ej[j] = sqrt(abs(δ))
            #     if δ > 0
            #         lowrankdowndate!(L, storage)
            #         lowrankupdate_turbo!(L, ei)
            #         lowrankupdate_turbo!(L, ej)
            #         lowrankupdate!(C, storage)
            #         lowrankdowndate_turbo!(C, ei)
            #         lowrankdowndate_turbo!(C, ej)
            #     else
            #         lowrankupdate!(L, storage)
            #         lowrankdowndate_turbo!(L, ei)
            #         lowrankdowndate_turbo!(L, ej)
            #         lowrankdowndate!(C, storage)
            #         lowrankupdate_turbo!(C, ei)
            #         lowrankupdate_turbo!(C, ej)
            #     end
            #     # update convergence tol
            #     abs(δ) > max_delta && (max_delta = abs(δ))
            # end
            offset += group_sizes[b]
        end
    end
    return S, Float64[]
end

function solve_group_max_entropy_full(
    Σ::AbstractMatrix{T}, 
    Sblocks::BlockDiagonal;
    niter::Int = 100,
    tol=1e-6, # converges when changes in s are all smaller than tol
    m::Int = 1, # number of knockoffs per variable
    verbose::Bool = false
    ) where T
    p = size(Σ, 1)
    blocks = nblocks(Sblocks)
    group_sizes = size.(Sblocks.blocks, 1)
    # initialize S matrix and compute initial cholesky factor
    # S, _ = solve_group_equi(Σ, Sblocks)
    # S = convert(Matrix{T}, S)
    S = Diagonal(fill(eigmin(Σ), size(Σ, 1))) |> Matrix
    L = cholesky(Symmetric((m+1)/m * Σ - S))
    C = cholesky(Symmetric(S))
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
                ldiv!(x, UpperTriangular(L.factors)', ỹ) # non-allocating version of ldiv!(x, L.L, ỹ)
                x_l2sum = sum(abs2, x)
                # compute zeta and c as in alg 2.2 of askari et al
                ζ = (m+1)/m * Σ[j, j] - S[j, j]
                c = (ζ * x_l2sum) / (ζ + x_l2sum)
                # solve optimality condition in eq 75 of spector et al 2020
                sj_new = ((m+1)/m * Σ[j, j] - c) / 2
                δ = sj_new - S[j, j]
                # compute δ, ensuring S[j, j] + δ is in feasible region
                abs(δ) < 1e-15 && continue
                fill!(ej, 0)
                ej[j] = 1
                ldiv!(u, UpperTriangular(L.factors)', ej) # non-allocating version of ldiv!(u, L.L, ej)
                ldiv!(v, UpperTriangular(C.factors)', ej)
                ub = 1 / sum(abs2, u) - tol
                lb = -1 / sum(abs2, v) + tol
                lb ≥ ub && continue
                δ = clamp(δ, lb, ub)
                # update S
                S[j, j] += δ
                # rank 1 update to cholesky factors
                fill!(x, 0); fill!(ỹ, 0)
                x[j] = ỹ[j] = sqrt(abs(δ))
                if δ > 0
                    lowrankdowndate_turbo!(L, x)
                    lowrankupdate_turbo!(C, ỹ)
                else
                    lowrankupdate_turbo!(L, x)
                    lowrankdowndate_turbo!(C, ỹ)
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
                ldiv!(u, UpperTriangular(L.factors)', ei) # non-allocating version of ldiv!(u, L.L, ei)
                ldiv!(v, UpperTriangular(L.factors)', ej)
                aij, aii, ajj = dot(u, v), dot(u, u), dot(v, v)
                ldiv!(u, UpperTriangular(C.factors)', ei)
                ldiv!(v, UpperTriangular(C.factors)', ej)
                bij, bii, bjj = dot(u, v), dot(u, u), dot(v, v)
                # compute feasible region
                s1 = (aij - sqrt(aii*ajj)) / (aij^2 - aii * ajj)
                s2 = (aij + sqrt(aii*ajj)) / (aij^2 - aii * ajj)
                d1 = (-bij - sqrt(bii*bjj)) / (bij^2 - bii * bjj)
                d2 = (-bij + sqrt(bii*bjj)) / (bij^2 - bii * bjj)
                s1 > s2 && ((s1, s2) = (s2, s1))
                d1 > d2 && ((d1, d2) = (d2, d1))
                # less stringent feasible region criteria
                lb = max(s1, d1, -1 / (bii + 2bij + bjj)) + tol
                ub = min(s2, d2, 1 / (aii + 2aij + ajj)) - tol
                # most stringent feasible region criteria
                # lb = max(s1, d1, -1 / (bii + 2bij + bjj), -1 / (2bij + bjj)) + tol
                # ub = min(s2, d2, 1 / (aii + 2aij + ajj), 1 / (2aij + ajj)) - tol
                if lb ≥ ub
                    println("lb ≥ ub at i=$i, j=$j")
                    continue
                end
                # lb ≥ ub && continue
                # ensure S[i, j] + δ and S[j, i] + δ are in feasible region
                δ = clamp((aij - bij) / (aij^2 + bij^2 - aii*ajj - bii*bjj), lb, ub)
                abs(δ) < 1e-5 && continue
                # f(x) = log((1 - x*aij)^2 - x^2*aii*ajj) + m*log((1 - x+bij)^2 - x^2*bjj*bii)
                # find_zero(f, (lb, ub))
                # update S


                # fill!(ei, 0); fill!(ej, 0)
                # ei[i] = ej[j] = sqrt(abs(δ))
                # @show δ, lb, ub
                # @show eigmin(S + δ*(ei + ej)*(ei + ej)')
                # @show eigmin(S + δ*((ei + ej)*(ei + ej)' - ei*ei'))
                # @show eigmin(S + δ*(ei*ej' + ei*ej'))



                S[i, j] += δ
                S[j, i] += δ
                # update cholesky factor L
                fill!(x, 0); fill!(ei, 0); fill!(ej, 0)
                x[j] = x[i] = ei[i] = ej[j] = sqrt(abs(δ))
                if δ > 0
                    lowrankdowndate_turbo!(L, x)
                    lowrankupdate_turbo!(L, ei)
                    lowrankupdate_turbo!(L, ej)
                else 
                    lowrankupdate_turbo!(L, x)
                    lowrankdowndate_turbo!(L, ei)
                    lowrankdowndate_turbo!(L, ej)
                end
                # update cholesky factor C
                fill!(x, 0); fill!(ei, 0); fill!(ej, 0)
                x[j] = x[i] = ei[i] = ej[j] = sqrt(abs(δ))
                if δ > 0
                    if i == 49 && j == 46

                        fill!(ei, 0); fill!(ej, 0)
                        ei[i] = ej[j] = 1
                        @show δ, lb, ub
                        @show eigmin(S + δ*(ei + ej)*(ei + ej)')
                        @show eigmin(S + δ*((ei + ej)*(ei + ej)' - ei*ei'))
                        @show eigmin(S + δ*(ei*ej' + ei*ej'))

                        @show eigmin(C.L * C.U + δ*(ei + ej)*(ei + ej)')
                        @show eigmin(C.L * C.U + δ*((ei + ej)*(ei + ej)' - ei*ei'))
                        @show eigmin(C.L * C.U + δ*(ei*ej' + ei*ej'))

                        lowrankupdate_turbo!(C, x)

                        # @show δ
                        # @show ei
                        # @show C.L * C.U

                        @show δ
                        @show eigmin(C.L * C.U - δ * ei*ei')

                        lowrankdowndate_turbo!(C, ei)

                        fff
                    end
                    try
                        lowrankupdate_turbo!(C, x)
                        lowrankdowndate_turbo!(C, ei)
                        lowrankdowndate_turbo!(C, ej)
                    catch
                        @show i, j
                        fdsa
                    end
                    # lowrankupdate_turbo!(C, x)
                    # lowrankdowndate_turbo!(C, ei)
                    # lowrankdowndate_turbo!(C, ej)
                else
                    lowrankdowndate_turbo!(C, x)
                    lowrankupdate_turbo!(C, ei)
                    lowrankupdate_turbo!(C, ej)
                end
                # update convergence tol
                abs(δ) > max_delta && (max_delta = abs(δ))
            end
            offset += group_sizes[b]
        end
        verbose && println("Iter $l: δ = $max_delta")
        max_delta < tol && break 
    end
    return S, Float64[]
end

# rank 1 update to cholesky factor
function rank1_update_linesearch(δj, C, L)
    δ = sqrt(abs(δj))
    # update C via linesearch
    for i in 1:50
        try
            ej[j] = δ
            if δj > 0
                lowrankdowndate_turbo!(L, ej)
                lowrankupdate_turbo!(C, ej)
            else
                lowrankupdate_turbo!(L, ej)
                lowrankdowndate_turbo!(C, ej)
            end
            break
        catch
            δ /= 2
            continue
        end
    end
    if δj > 0
        lowrankdowndate_turbo!(L, ej)
    else
        lowrankupdate_turbo!(L, ej)
    end
    S[j, j] += δ
end

function solve_group_quadratic(cn, cd, kn, kd)
    a = cn*kd^2 - cd^2*kn
    b = 2*(cn*kd - cd*kn)
    c = cn - kd
    a == c == 0 && return 0 # quick return; when a = c = 0, only solution is δ = 0
    x1 = (-b + sqrt(b^2 - 4*a*c)) / (2a)
    x2 = (-b - sqrt(b^2 - 4*a*c)) / (2a)
    δ = inv(cd) < x1 < inv(kd) ? x1 : x2
    # @show x1, x2, inv(cd), inv(kd) 
    isinf(δ) && error("δ is Inf, aborting")
    isnan(δ) && error("δ is NaN, aborting")
    return δ
end

"""
    solve_s_group(Σ, Sblocks, groups, [method=:equi]; kwargs...)

Solves the group knockoff problem, returns block diagonal matrix S
satisfying `2Σ - S ⪰ 0` and the constant(s) γ.

# Inputs 
+ `Σ`: A covariance matrix that has been scaled to a correlation matrix.
+ `Sblocks`: A `BlockDiagonal` matrix that approximates `Σ` using group
    structure. Each block should be `pi × pi` where `pi` is number of variables
    in group `i`
+ `groups`: Vector of group membership
+ `method`: Method for constructing knockoffs. Options are `:equi` or `:sdp`
"""
function solve_s_group(
    Σ::AbstractMatrix, 
    Sblocks::BlockDiagonal, 
    groups::Vector{Int},
    method::Symbol=:equi;
    kwargs...)
    # check for error first
    all(x -> x ≈ 1, diag(Σ)) || error("Σ must be scaled to a correlation matrix first.")
    for block in Sblocks.blocks
        all(x -> x ≈ 1, diag(block)) || 
            error("Sblocks must be scaled to a correlation matrix first.")
    end
    # solve optimization problem
    if method == :equi
        S, γs = solve_group_equi(Σ, Sblocks)
    elseif method == :sdp
        S, γs = solve_group_SDP(Σ, Sblocks)
    elseif method == :sdp_full
        S, γs = solve_group_SDP_full(Σ, Sblocks)
    elseif method == :maxent
        S, γs = solve_group_max_entropy_equi(Σ, Sblocks)
    elseif method == :maxent_sdp
        S, γs = solve_group_max_entropy_sdp(Σ, Sblocks)
    elseif method == :mvr_full
        S, γs = solve_group_MVR_full(Σ, Sblocks)
    elseif method == :maxent_full
        S, γs = solve_group_max_entropy_full(Σ, Sblocks; kwargs...)
    else
        error("Method can only be :equi, :sdp, or :maxent, but was $method")
    end
    return S, γs
end

"""
    modelX_gaussian_group_knockoffs(X, groups, method, Σ, μ)
    modelX_gaussian_group_knockoffs(X, groups, method; [covariance_approximator])

Constructs Gaussian model-X group knockoffs. If the covariance `Σ` and mean `μ` 
are not specified, they will be estimated from data, i.e. we will make second-order
group knockoffs. To incorporate group structure, the (true or estimated) covariance 
matrix is block-diagonalized according to `groups` membership to solve a relaxed 
optimization problem. See reference paper and Knockoffs.jl docs for more details. 

# Inputs
+ `X`: A `n × p` design matrix. Each row is a sample, each column is a feature.
+ `groups`: Vector of group membership
+ `method`: Method for constructing knockoffs. Options are `:equi` or `:sdp`
+ `Σ`: A `p × p` covariance matrix for columns of `X`
+ `μ`: A length `p` vector storing the true column means of `X`
+ `covariance_approximator`: A covariance estimator, defaults to 
    `LinearShrinkage(DiagonalUnequalVariance(), :lw)`. See CovarianceEstimation.jl 
    for more options.

# Reference
Dai & Barber 2016, The knockoff filter for FDR control in group-sparse and multitask regression
"""
function modelX_gaussian_group_knockoffs(
    X::Matrix, 
    groups::AbstractVector{Int},
    method::Symbol;
    covariance_approximator=LinearShrinkage(DiagonalUnequalVariance(), :lw),
    kwargs...
    )
    # first check errors
    length(groups) == size(X, 2) || 
        error("Expected length(groups) == size(X, 2). Each variable in X needs a group membership.")
    issorted(groups) || 
        error("groups not sorted. Currently group memberships must be non-overlapping and contiguous")
    # approximate covariance matrix
    Σapprox = cov(covariance_approximator, X)
    # mean component is just column means
    μ = vec(mean(X, dims=1))
    return modelX_gaussian_group_knockoffs(X, groups, method, Σapprox, μ)
end

function modelX_gaussian_group_knockoffs(
    X::Matrix, 
    groups::AbstractVector{Int},
    method::Symbol,
    Σ::AbstractMatrix,
    μ::AbstractVector;
    kwargs...
    )
    # first check errors
    length(groups) == size(X, 2) || 
        error("Expected length(groups) == size(X, 2). Each variable in X needs a group membership.")
    issorted(groups) || 
        error("groups not sorted. Currently group memberships must be non-overlapping and contiguous")
    # Scale covariance to correlation matrix
    σs = sqrt.(diag(Σ))
    iscor = all(x -> x ≈ 1, σs)
    Σcor = iscor ? Σ : StatsBase.cov2cor!(Matrix(Σ), σs)
    # define group-blocks
    Sblocks = Matrix{eltype(X)}[]
    for g in unique(groups)
        idx = findall(x -> x == g, groups)
        push!(Sblocks, Σcor[idx, idx])
    end
    S = BlockDiagonal(Sblocks)
    # compute block diagonal S matrix using the specified knockoff method
    S, γs = solve_s_group(Σcor, S, groups, method; kwargs...)
    # rescale S back to the result for a covariance matrix   
    iscor || StatsBase.cor2cov!(S, σs)
    # generate knockoffs
    X̃ = condition(X, μ, Σ, S)
    return GaussianGroupKnockoff(X, X̃, S, γs, Symmetric(Σ), method)
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
Handle PLINK files with multiple chromosomes and multiple plink files each storing a chromosome
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
