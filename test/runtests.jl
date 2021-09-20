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

    # equi-correlated knockoff
    @time knockoff = fixed_knockoffs(X, method=:equi)
    X = knockoff.X
    X̃ = knockoff.X̃
    s = knockoff.s
    Σ = knockoff.Σ
    Σinv = knockoff.Σinv

    @test all(X' * X .≈ Σ)
    @test all(isapprox.(X̃' * X̃, Σ, atol=5e-2)) # numerical accuracy not good?
    @test all(s .≥ 0)
    @test all(1 .≥ s)
    # λ = eigvals(2Σ - Diagonal(s))
    # for λi in λ
    #     @test λi ≥ 0 || λi ≈ 0
    # end
    # @test all(isapprox.(Ũ' * X, 0, atol=1e-10))
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

    # SDP knockoff
    @time knockoff = fixed_knockoffs(X, method=:sdp)
    X = knockoff.X
    X̃ = knockoff.X̃
    s = knockoff.s
    Σ = knockoff.Σ
    Σinv = knockoff.Σinv

    @test all(X' * X .≈ Σ)
    @test all(isapprox.(X̃' * X̃, Σ, atol=5e-1)) # numerical accuracy not good?
    @test all(s .≥ 0)
    @test all(1 .≥ s)
    # λ = eigvals(2Σ - Diagonal(s))
    # for λi in λ
    #     @test λi ≥ 0 || λi ≈ 0
    # end
    # @test all(isapprox.(Ũ' * X, 0, atol=1e-10))
    for i in 1:p, j in 1:p
        if i == j
            @test isapprox(dot(X[:, i], X̃[:, i]), Σ[i, i] - s[i], atol=1e-8)
            @test isapprox(dot(X[:, i], X̃[:, i]), 1 - s[i], atol=1e-8)
        else
            @test dot(X[:, i], X̃[:, j]) ≈ dot(X[:, i], X[:, j])
        end
    end
end

@testset "Knockoff data structure" begin
    Random.seed!(2021)

    # simulate matrix and normalize columns
    n = 1000
    p = 100
    X = randn(n, p)
    zscore!(X, mean(X, dims=1), std(X, dims=1)) # center/scale Xj to mean 0 var 1
    normalize_col!(X) # normalize Xj to unit length

    # construct knockoff struct and the real [A Ã]
    @time A = fixed_knockoffs(X, method=:sdp)
    Atrue = [A.X A.X̃]

    # array operations
    @test size(Atrue) == size(A)
    @test eltype(Atrue) == eltype(A)
    @test getindex(Atrue, 127) == getindex(A, 127)
    @test getindex(Atrue, 2, 19) == getindex(A, 2, 19)
    @test getindex(Atrue, 900, 110) == getindex(A, 900, 110)
    @test all(@view(Atrue[:, 1]) .== @view(A[:, 1]))
    @test all(@view(Atrue[1:2:end, 1:5:end]) .== @view(A[1:2:end, 1:5:end]))

    # matrix-vector multiplication
    b = randn(2p)
    ctrue, c = zeros(n), zeros(n)
    mul!(ctrue, Atrue, b)
    mul!(c, A, b)
    @test all(ctrue .≈ c)

    # matrix-matrix multiplication 
    B = randn(2p, n)
    Ctrue, C = zeros(n, n), zeros(n, n)
    mul!(Ctrue, Atrue, B)
    mul!(C, A, B)
    @test all(Ctrue .≈ C)
end
