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

    @test all(X' * X .≈ Σ)
    @test all(isapprox.(X̃' * X̃, Σ, atol=5e-2)) # numerical accuracy not good?
    @test all(isapprox.(X' * X, Σ, atol=1e-10)) # good accuracy
    @test all(s .> 0)
    for i in 1:p-1
        @test dot(X[:, i], X̃[:, i+1]) ≈ dot(X[:, i], X[:, i+1])
    end
    for i in 1:p
        @test isapprox(dot(X[:, i], X̃[:, i]), Σ[i, i] - s[i])
        @test isapprox(dot(X[:, i], X̃[:, i]), 1 - s[i], atol=5e-2) # numerical accuracy not good?
    end
end
