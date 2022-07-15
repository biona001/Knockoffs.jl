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
Solves the equi-correlated group knockoff problem, detailed in
Dai & Barber 2016, The knockoff filter for FDR control in group-sparse and multitask regression
"""
function solve_Sb_equi(Σb::BlockDiagonal)
    Db = Matrix{eltype(Σ)}[]
    for Σbi in Σb.blocks
        push!(Db, inverse_mat_sqrt(Symmetric(Σbi)))
    end
    Db = BlockDiagonal(Db)
    λmin = Symmetric(Db * Σb * Db) |> eigvals |> minimum
    γb = min(1, 2λmin)
    Sb = BlockDiagonal(γb .* Σb.blocks)
    return Sb
end

function solve_s_group(Σ::BlockDiagonal, method=:equi; kwargs...)
    S = BlockDiagonal{eltype(Σ)}[]
    for Σb in Σ.blocks
        push!(S, solve_Sb_equi(Σb))
    end
    return BlockDiagonal(S)
end

function modelX_gaussian_group_knockoffs(
    X::Matrix, 
    groups::AbstractVector{Int};
    method::Symbol,
    covariance_approximator=LinearShrinkage(DiagonalUnequalVariance(), :lw),
    kwargs...
    )
    # approximate Σ
    Σapprox = BlockDiagonal{eltype(X)}[]
    for g in unique(groups)
        idx = findall(x -> x == g, groups)
        push!(Σapprox, cov(covariance_approximator, @view(X[:, idx])))
    end
    Σapprox = BlockDiagonal(BlockDiagonal)
    # mean component is just column means
    μ = vec(mean(X, dims=1))
    return modelX_gaussian_group_knockoffs(X, method, μ, Σapprox; kwargs...)
end

function modelX_gaussian_group_knockoffs(X::Matrix, method::Symbol, μ::AbstractVector, Σ::AbstractMatrix; kwargs...)
    # compute s vector using the specified method
    S = solve_s_group(Σ, method; kwargs...) :: BlockDiagonal
    # generate knockoffs
    X̃ = condition(X, μ, inv(Σ), S)
    return GaussianGroupKnockoff(X, X̃, S, Σ, method)
end
