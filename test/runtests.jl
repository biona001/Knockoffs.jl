using Knockoffs
using Test
using LinearAlgebra
using Random
using StatsBase
using Statistics
using Distributions

@testset "fixed equi knockoffs" begin
    Random.seed!(2021)

    # simulate matrix and normalize columns
    n = 3000
    p = 1000
    X = randn(n, p)
    zscore!(X, mean(X, dims=1), std(X, dims=1)) # center/scale Xj to mean 0 var 1
    normalize_col!(X) # normalize columns 

    # equi-correlated knockoff
    @time knockoff = fixed_knockoffs(X, :equi)
    X = knockoff.X
    X̃ = knockoff.X̃
    s = knockoff.s
    Σ = knockoff.Σ
    Σinv = knockoff.Σinv

    @test all(isapprox.(X' * X, Σ, atol=1e-10))
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
            @test isapprox(dot(X[:, i], X̃[:, j]), dot(X[:, i], X[:, j]), atol=1e-8)
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
    normalize_col!(X) # normalize columns 

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

# @testset "model X Guassian Knockoffs" begin
#     Random.seed!(2021)

#     # simulate matrix
#     n = 300
#     p = 600
#     X = randn(n, p)
#     zscore!(X, mean(X, dims=1), std(X, dims=1)) # center/scale Xj to mean 0 var 1
#     # normalize_col!(X) # normalize columns 

#     # generate knockoff
#     true_μ = zeros(p)
#     true_var = Matrix{Float64}(I, p, p)
#     @time knockoff = modelX_gaussian_knockoffs(X, :sdp, true_μ, true_var)
#     X = knockoff.X
#     X̃ = knockoff.X̃
#     s = knockoff.s

#     # test properties
#     @test all(X' * X .≈ Σ)
#     @test all(isapprox.(X̃' * X̃, Σ, atol=0.5)) # numerical accuracy not good?
#     @test all(s .≥ 0)
#     @test all(1 .≥ s)
#     for i in 1:p, j in 1:p
#         if i == j
#             @test isapprox(dot(X[:, i], X̃[:, i]), Σ[i, i] - s[i], atol=1.0) # numerical accuracy not good?
#         else
#             @test isapprox(dot(X[:, i], X̃[:, j]), dot(X[:, i], X[:, j]), atol=1.0) # numerical accuracy not good?
#         end
#     end
# end

@testset "threshold functions" begin
    w = [0.1, 1.9, 1.3, 1.8, 0.8, -0.7, -0.1]
    @test threshold(w, 0.2) == 0.8
    @test threshold(w, 0.2, :knockoff_plus) == Inf

    w = [0.27, 0.76, 0.21, 0.1, -0.38, -0.01]
    @test threshold(w, 0.4) == 0.1
    @test threshold(w, 0.5, :knockoff_plus) == 0.1

    w = [0.74, -0.65, -0.83, -0.27, -0.19, 0.4]
    @test threshold(w, 0.25) == Inf
    @test threshold(w, 0.25, :knockoff_plus) == Inf
end

@testset "coefficient_diff" begin
    # knockoffs and original are randomly swapped
    β = [1.0, 0.2, -0.3, 0.8, -0.1, 0.5]
    original = [1, 4, 6]
    knockoff = [2, 3, 5]
    w = coefficient_diff(β, original, knockoff)
    @test length(w) == 3
    @test w[1] ≈ 0.8
    @test w[2] ≈ 0.5
    @test w[3] ≈ 0.4

    # group knockoffs
    β = [1.0, 0.2, -0.3, 0.8, -0.1, 0.5, 0.5, -0.1]
    groups = [1, 1, 1, 1, 2, 2, 2, 2]
    original = [1, 4, 6, 7]
    knockoff = [2, 3, 5, 8]
    w = coefficient_diff(β, groups, original, knockoff)
    @test length(w) == 2
    @test w[1] ≈ 1.8 - 0.5
    @test w[2] ≈ 1.0 - 0.2
end

function sample_DMC(q, Q; n=1)
    p = size(Q, 3)
    d = Categorical(q)
    X = zeros(Int, n, p)
    for i in 1:n
        X[i, 1] = rand(d)
        for j in 2:p
            d.p .= @view(Q[X[i, j-1], :, j])
            X[i, j] = rand(d)
        end
    end
    return X
end

# from https://github.com/msesia/snpknock/blob/master/tests/testthat/test_knockoffs.R
@testset "Markov chain knockoffs have the right correlation structure" begin
    p = 20 # Number of states in markov chain
    K = 3  # Number of possible states for each variable
    q = 1/K .* ones(K) # Marginal distribution for the first variable
    samples = 1000000 # number of samples in this example

    # form random transition matrices
    Q = rand(K, K, p)
    for j in 1:p
        Qj = @view(Q[:, :, j])
        Qj ./= sum(Qj, dims=2)
    end
    fill!(@view(Q[:, :, 1]), NaN)

    # sample a bunch of markov chains
    X = sample_DMC(q, Q, n=samples)

    # sample knockoff of the markov chains
    X̃ = zeros(Int, samples, p)
    N = zeros(p, K)
    d = Categorical([1 / K for _ in 1:K])
    for i in 1:samples
        markov_knockoffs!(@view(X̃[i, :]), @view(X[i, :]), N, d, Q, q) 
    end

    # Check column means match
    Xmean = mean(X, dims=1)
    X̃mean = mean(X̃, dims=1)
    for i in 2:length(Xmean) # 1st entry might not match to 2 digits for some reason, this is the same in SNPknock
        @test isapprox(Xmean[i], X̃mean[i], atol=1e-2)
    end

    # Check that internal column correlations match
    for i in 2:p-1
        r1 = cor(@view(X[:, i]), @view(X[:, i+1]))
        r2 = cor(@view(X̃[:, i]), @view(X̃[:, i+1]))
        @test isapprox(r1, r2, atol=1e-2)
    end

    # Check that cross column correlations match
    for i in 2:p-1
        r1 = cor(@view(X[:, i]), @view(X[:, i+1]))
        r2 = cor(@view(X[:, i]), @view(X̃[:, i+1]))
        @test isapprox(r1, r2, atol=1e-2)
    end
end
