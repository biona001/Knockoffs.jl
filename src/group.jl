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
`Σ` is the true covariance matrix and `Σblocks` is the 
block-diagonal covariance matrix where each block corresponds
to groups.

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
    return S
end

function solve_group_SDP(Σ::AbstractMatrix, Σblocks::BlockDiagonal)
    model = Model(() -> Hypatia.Optimizer(verbose=false))
    n = nblocks(Σblocks)
    @variable(model, 0 <= γ[1:n] <= 1)
    blocks = BlockDiagonal([γ[i] * Σblocks.blocks[i] for i in 1:n]) |> Matrix
    @constraint(model, Symmetric(2Σ - blocks) in PSDCone())
    JuMP.optimize!(model)
    γs = clamp!(JuMP.value.(γ), 0, 1)
    S = BlockDiagonal(γs .* Σblocks.blocks)
    return S
end
# Σ = 2 .* (0.5 * Matrix(I, 100, 100) + 0.5 * ones(100, 100))
# Σblocks = BlockDiagonal([0.5 * Matrix(I, 10, 10) + 0.5 * ones(10, 10) for _ in 1:10])
# Sequi = solve_group_equi(Σ, Σblocks)
# Ssdp = solve_group_SDP(Σ, Σblocks)

function solve_s_group(
    Σ::AbstractMatrix, 
    groups::Vector{Int},
    method::Symbol=:equi;
    kwargs...)
    # create correlation matrix
    σs = sqrt.(diag(Σ))
    iscor = all(x -> x ≈ 1, σs)
    Σcor = iscor ? Σ : StatsBase.cov2cor!(Matrix(Σ), σs)
    # define group-blocks
    Σblocks = Matrix{eltype(Σ)}[]
    for g in unique(groups)
        idx = findall(x -> x == g, groups)
        push!(Σblocks, Σcor[idx, idx])
    end
    Σblocks = BlockDiagonal(Σblocks)
    # solve optimization problem
    if method == :equi
        S = solve_group_equi(Σcor, Σblocks)
    elseif method == :sdp
        S = solve_group_SDP(Σcor, Σblocks)
    else
        error("Method can only be :equi or :sdp, but was $method")
    end
    # rescale S back to the result for a covariance matrix   
    iscor || StatsBase.cor2cov!(S, σs)
    return S
end

function modelX_gaussian_group_knockoffs(
    X::Matrix, 
    groups::AbstractVector{Int},
    method::Symbol;
    covariance_approximator=LinearShrinkage(DiagonalUnequalVariance(), :lw),
    kwargs...
    )
    # approximate Σ
    Σapprox = cov(covariance_approximator, X)
    # mean component is just column means
    μ = vec(mean(X, dims=1))
    return modelX_gaussian_group_knockoffs(X, groups, method, μ, Σapprox; kwargs...)
end

function modelX_gaussian_group_knockoffs(
    X::Matrix, 
    groups::AbstractVector{Int},
    method::Symbol, 
    μ::AbstractVector, 
    Σ::AbstractMatrix; 
    kwargs...)
    # compute block diagonal S matrix using the specified method
    S = solve_s_group(Σ, groups, method; kwargs...) :: BlockDiagonal
    # generate knockoffs
    X̃ = condition(X, μ, inv(Σ), S)
    return GaussianGroupKnockoff(X, X̃, S, Symmetric(Σ), method)
end
