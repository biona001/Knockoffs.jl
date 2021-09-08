using Knockoffs
using Test
using LinearAlgebra
using Random
using StatsBase

@testset "equi-correlated knockoffs" begin
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
    λ = eigvals(2Σ - Diagonal(s)); @test all(λ .≥ 0)
    @test all(isapprox.(Ũ' * X, 0, atol=1e-10))
    for i in 1:p-1
        @test dot(X[:, i], X̃[:, i+1]) ≈ dot(X[:, i], X[:, i+1])
    end
    for i in 1:p
        @test isapprox(dot(X[:, i], X̃[:, i]), Σ[i, i] - s[i])
        @test isapprox(dot(X[:, i], X̃[:, i]), 1 - s[i], atol=5e-2) # numerical accuracy not good?
    end
end
