using Knockoffs
using Test
using LinearAlgebra
using Random
using StatsBase

@testset "equi-correlated knockoffs" begin
    Random.seed!(2021)

    # simulate matrix and normalize columns
    n = 3000
    p = 1000
    X = randn(n, p)
    normalize_col!(X)

    # construct knockoff struct
    knockoff = knockoff_equi(X)
    X = knockoff.X
    X̃ = knockoff.X̃
    s = knockoff.s
    C = knockoff.C
    Ũ = knockoff.Ũ
    Σ = knockoff.Σ
    Σinv = knockoff.Σinv

    @test all(X' * X .≈ Σ) # good accuracy
    @test all(isapprox.(X̃' * X̃, Σ, atol=5e-2)) # numerical accuracy not good?
    @test all(s .≥ 0)
    λ = eigvals(2Σ - Diagonal(s))
    for λi in λ
        @test λi ≥ 0 || λi ≈ 0
    end
    @test all(isapprox.(Ũ' * X, 0, atol=1e-10))
    for i in 1:p, j in 1:p
        if i == j
            @test isapprox(dot(X[:, i], X̃[:, i]), Σ[i, i] - s[i])
            @test isapprox(dot(X[:, i], X̃[:, i]), 1 - s[i], atol=5e-2) # numerical accuracy not good?
        else
            @test dot(X[:, i], X̃[:, j]) ≈ dot(X[:, i], X[:, j])
        end
    end
end
