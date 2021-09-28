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
    @time knockoff = fixed_knockoffs(X, :equi)
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
    @time knockoff = fixed_knockoffs(X, :sdp)
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
    @time A = fixed_knockoffs(X, :sdp)
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

@testset "model X Guassian Knockoffs" begin
    Random.seed!(2021)

    # simulate matrix (but manually normalizing columns leads to rank deficiency for some reason)
    n = 300
    p = 600
    X = randn(n, p)
    # zscore!(X, mean(X, dims=1), std(X, dims=1))
    # normalize_col!(X)
    # @show rank(X)

    # generate knockoff
    @time knockoff = modelX_gaussian_knockoffs(X, :sdp, zeros(p));
    X = knockoff.X
    X̃ = knockoff.X̃
    s = knockoff.s
    Σ = knockoff.Σ
    Σinv = knockoff.Σinv

    # test properties
    @test all(X' * X .≈ Σ)
    @test all(isapprox.(X̃' * X̃, Σ, atol=0.5)) # numerical accuracy not good?
    @test all(s .≥ 0)
    @test all(1 .≥ s)
    for i in 1:p, j in 1:p
        if i == j
            @test isapprox(dot(X[:, i], X̃[:, i]), Σ[i, i] - s[i], atol=1.0) # numerical accuracy not good?
        else
            @test isapprox(dot(X[:, i], X̃[:, j]), dot(X[:, i], X[:, j]), atol=1.0) # numerical accuracy not good?
        end
    end
end

@testset "threshold functions" begin
    w = [0.1, 1.9, 1.3, 1.8, 0.8, -0.7, -0.1]
    τ = threshold(w, 0.2)
    @test τ == 0.8
    τ = threshold(w, 0.2, :knockoff_plus)

    w = [0.27, 0.76, 0.21, 0.1, -0.38, -0.01]
    τ = threshold(w, 0.4)
    @test τ == 0.1
    τ = threshold(w, 0.5, :knockoff_plus)
    @test τ == 0.1

    w = [0.74, -0.65, -0.83, -0.27, -0.19, 0.4]
    τ = threshold(w, 0.25)
    @test τ == Inf
end

@testset "coefficient_diff" begin
    # knockoffs are at the end (e.g. [XX̃])
    β = [1.0, 0.2, -0.3, 0.8, -0.1, 0.5]
    w = coefficient_diff(β, :concatenated)
    @test length(w) == 3
    @test w[1] ≈ 0.2
    @test w[2] ≈ 0.1
    @test w[3] ≈ -0.2

    # knockoffs are interleaved (e.g. [x₁x̃₁x₂x̃₂...])
    β = [1.0, 0.2, -0.3, 0.8, -0.1, 0.5]
    w = coefficient_diff(β, :interleaved)
    @test length(w) == 3
    @test w[1] ≈ 0.8
    @test w[2] ≈ -0.5
    @test w[3] ≈ -0.4

    # knockoffs are at the end (e.g. [XX̃])
    B = [[1.0, 2.0, -0.3, 0.8, -0.1, 0.5] [-0.4, 0.6, 1.8, -0.3, 1.4, 0.3]]
    w = coefficient_diff(B, :concatenated)
    @test length(w) == 3
    @test w[1] ≈ 0.3
    @test w[2] ≈ 1.1
    @test w[3] ≈ 1.3

    # knockoffs are interleaved (e.g. [x₁x̃₁x₂x̃₂...])
    B = [[1.0, 2.0, -0.3, 0.8, -0.1, 0.5] [-0.4, 0.6, 1.8, -0.3, 1.4, 0.3]]
    w = coefficient_diff(B, :interleaved)
    @test length(w) == 3
    @test w[1] ≈ -1.2
    @test w[2] ≈ 1.0
    @test w[3] ≈ 0.7
end
