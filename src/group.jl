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
    return S, [γ]
end

function solve_group_SDP(Σ::AbstractMatrix, Σblocks::BlockDiagonal)
    model = Model(() -> Hypatia.Optimizer(verbose=false))
    n = nblocks(Σblocks)
    block_sizes = size.(Σblocks.blocks, 1)
    @variable(model, 0 <= γ[1:n] <= 1)
    blocks = BlockDiagonal([γ[i] * Σblocks.blocks[i] for i in 1:n]) |> Matrix
    @objective(model, Max, block_sizes' * γ)
    @constraint(model, Symmetric(2Σ - blocks) in PSDCone())
    JuMP.optimize!(model)
    γs = clamp!(JuMP.value.(γ), 0, 1)
    S = BlockDiagonal(γs .* Σblocks.blocks)
    return S, γs
end

function solve_s_group(
    Σ::AbstractMatrix, 
    Sblocks::BlockDiagonal,
    groups::Vector{Int},
    method::Symbol=:equi;
    kwargs...)
    # create correlation matrix
    σs = sqrt.(diag(Σ))
    iscor = all(x -> x ≈ 1, σs)
    Σcor = iscor ? Σ : StatsBase.cov2cor!(Matrix(Σ), σs)
    # solve optimization problem
    if method == :equi
        S, γs = solve_group_equi(Σcor, Sblocks)
    elseif method == :sdp
        S, γs = solve_group_SDP(Σcor, Sblocks)
    else
        error("Method can only be :equi or :sdp, but was $method")
    end
    # rescale S back to the result for a covariance matrix   
    iscor || StatsBase.cor2cov!(S, σs)
    return S, γs
end

function modelX_gaussian_group_knockoffs(
    X::Matrix, 
    groups::AbstractVector{Int},
    method::Symbol;
    covariance_approximator=LinearShrinkage(DiagonalUnequalVariance(), :lw),
    kwargs...
    )
    length(groups) == size(X, 2) || error("Each variable in X needs a group membership")
    issorted(groups) || error("groups not sorted. Currently group memberships must be non-overlapping and contiguous")
    # approximate Σ
    Σapprox = cov(covariance_approximator, X)
    # define group-blocks
    Sblocks = Matrix{eltype(X)}[]
    for g in unique(groups)
        idx = findall(x -> x == g, groups)
        push!(Sblocks, cov(covariance_approximator, @view(X[:, idx])))
    end
    Sblocks = BlockDiagonal(Sblocks)
    # mean component is just column means
    μ = vec(mean(X, dims=1))
    return modelX_gaussian_group_knockoffs(X, groups, method, μ, Σapprox, Sblocks; kwargs...)
end

function modelX_gaussian_group_knockoffs(
    X::Matrix, 
    groups::AbstractVector{Int},
    method::Symbol, 
    μ::AbstractVector, 
    Σ::AbstractMatrix,
    Sblocks::BlockDiagonal; 
    kwargs...)
    # compute block diagonal S matrix using the specified method
    S, γs = solve_s_group(Σ, Sblocks, groups, method; kwargs...)
    # generate knockoffs
    X̃ = condition(X, μ, inv(Σ), S)
    return GaussianGroupKnockoff(X, X̃, S, γs, Symmetric(Σ), method)
end
