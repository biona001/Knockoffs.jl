using Knockoffs
using Test
using LinearAlgebra
using Random
using StatsBase
using Statistics

@testset "fixed equi knockoffs" begin
    Random.seed!(2021)

    # simulate matrix and normalize columns
    n = 3000
    p = 1000
    X = randn(n, p)
    zscore!(X, mean(X, dims=1), std(X, dims=1)) # center/scale Xj to mean 0 var 1
    normalize_col!(X) # normalize Xj to unit length

    # construct knockoff struct
    @time knockoff = knockoff_equi(X)
    X = knockoff.X
    X̃ = knockoff.X̃
    s = knockoff.s
    C = knockoff.C
    Ũ = knockoff.Ũ
    Σ = knockoff.Σ
    Σinv = knockoff.Σinv

    @test all(X' * X .≈ Σ)
    @test all(isapprox.(X̃' * X̃, Σ, atol=5e-2)) # numerical accuracy not good?
    @test all(s .≥ 0)
    # λ = eigvals(2Σ - Diagonal(s))
    # for λi in λ
    #     @test λi ≥ 0 || λi ≈ 0
    # end
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

@testset "fixed SDP knockoffs" begin
    Random.seed!(2021)

    # simulate matrix and normalize columns
    n = 1000
    p = 100
    X = randn(n, p)
    zscore!(X, mean(X, dims=1), std(X, dims=1)) # center/scale Xj to mean 0 var 1
    normalize_col!(X) # normalize Xj to unit length

    # construct knockoff struct
    @time knockoff = knockoff_sdp(X)
    X = knockoff.X
    X̃ = knockoff.X̃
    s = knockoff.s
    C = knockoff.C
    Ũ = knockoff.Ũ
    Σ = knockoff.Σ
    Σinv = knockoff.Σinv

    @test all(X' * X .≈ Σ)
    @test all(isapprox.(X̃' * X̃, Σ, atol=5e-1)) # numerical accuracy not good?
    @test all(s .≥ 0)
    # λ = eigvals(2Σ - Diagonal(s))
    # for λi in λ
    #     @test λi ≥ 0 || λi ≈ 0
    # end
    @test all(isapprox.(Ũ' * X, 0, atol=1e-10))
    for i in 1:p, j in 1:p
        if i == j
            @test isapprox(dot(X[:, i], X̃[:, i]), Σ[i, i] - s[i], atol=1e-8)
            @test isapprox(dot(X[:, i], X̃[:, i]), 1 - s[i], atol=1e-8)
        else
            @test dot(X[:, i], X̃[:, j]) ≈ dot(X[:, i], X[:, j])
        end
    end
end
