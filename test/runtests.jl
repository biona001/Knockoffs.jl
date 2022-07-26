using Knockoffs
using Test
using LinearAlgebra
using Random
using StatsBase
using Statistics
using Distributions
using ToeplitzMatrices
# using RCall # for comparing with Matteo's knockoffs

@testset "fixed equi knockoffs" begin
    Random.seed!(2021)

    # simulate matrix and normalize columns
    n = 3000
    p = 1000
    X = randn(n, p)

    # equi-correlated knockoff
    @time knockoff = fixed_knockoffs(X, :equi)
    X = knockoff.X
    X̃ = knockoff.X̃
    s = knockoff.s
    Σ = knockoff.Σ

    @test all(isapprox.(X' * X, Σ, atol=1e-10))
    @test all(isapprox.(X̃' * X̃, Σ, atol=5e-2)) # numerical accuracy not good?
    @test all(s .≥ 0)
    @test all(1 .≥ s)
    λ = eigvals(2Σ - Diagonal(s))
    for λi in λ
        @test λi ≥ 0 || isapprox(λi, 0, atol=1e-8)
    end
    # @test all(isapprox.(Ũ' * X, 0, atol=1e-10))
    for i in 1:p, j in 1:p
        if i == j
            # @test isapprox(dot(X[:, i], X[:, i]), 1, atol=1e-1)
            # @test isapprox(dot(X̃[:, i], X̃[:, i]), 1, atol=1e-1)
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

    # SDP knockoff
    @time knockoff = fixed_knockoffs(X, :sdp)
    X = knockoff.X
    X̃ = knockoff.X̃
    s = knockoff.s
    Σ = knockoff.Σ

    # compare with Matteo's result
    # @rput X
    # R"""
    # library(knockoff)
    # X_ko = create.fixed(X)
    
    # # get s vector (need to load many functions in https://github.com/msesia/knockoff-filter/blob/508ed64d914137d22ae8ad344311e147900fb437/R/knockoff/R/create_fixed.R)
    # X.svd = decompose(X, 'F')
    # tol = 1e-5
    # d = X.svd$d
    # d_inv = 1 / d
    # d_zeros = d <= tol*max(d)
    # if (any(d_zeros)) {
    #   warning(paste('Data matrix is rank deficient.',
    #                 'Model is not identifiable, but proceeding with SDP knockoffs'),immediate.=T)
    #   d_inv[d_zeros] = 0
    # }
    # G = (X.svd$v %*diag% d^2) %*% t(X.svd$v)
    # G_inv = (X.svd$v %*diag% d_inv^2) %*% t(X.svd$v)
    # s_matteo = create.solve_sdp(G)
    # s_matteo[s_matteo <= tol] = 0
    # """
    # @rget X_ko s_matteo
    # [s matteo_s]
    # histogram(vec(X_ko))
    # histogram(vec(X̃))

    @test all(X' * X .≈ Σ)
    @test all(isapprox.(X̃' * X̃, Σ, atol=5e-1)) # numerical accuracy not good?
    @test all(s .≥ 0)
    @test all(1 .≥ s)
    λ = eigvals(2Σ - Diagonal(s))
    for λi in λ
        @test λi ≥ 0 || isapprox(λi, 0, atol=1e-8)
    end
    # @test all(isapprox.(Ũ' * X, 0, atol=1e-10))
    for i in 1:p, j in 1:p
        if i == j
            # @test isapprox(dot(X[:, i], X[:, i]), 1, atol=1e-1)
            # @test isapprox(dot(X̃[:, i], X̃[:, i]), 1, atol=1e-1)
            @test isapprox(dot(X[:, i], X̃[:, i]), Σ[i, i] - s[i], atol=1e-8)
            @test isapprox(dot(X[:, i], X̃[:, i]), 1 - s[i], atol=1e-8)
        else
            @test dot(X[:, i], X̃[:, j]) ≈ dot(X[:, i], X[:, j])
        end
    end
end

@testset "model X Guassian Knockoffs" begin
    # example from https://github.com/msesia/knockoff-filter/blob/master/R/knockoff/R/create_gaussian.R

    # simulate matrix
    Random.seed!(2022)
    n = 100
    p = 200
    ρ = 0.4
    Sigma = Matrix(SymmetricToeplitz(ρ.^(0:(p-1))))
    L = cholesky(Sigma).L
    X = randn(n, p) * L # var(X) = L var(N(0, 1)) L' = var(Σ)

    # generate knockoff
    true_mu = zeros(p)
    @time knockoff = modelX_gaussian_knockoffs(X, :sdp, true_mu, Sigma)
    X = knockoff.X
    X̃ = knockoff.X̃
    s = knockoff.s
    Σ = knockoff.Σ

    # compare with Matteo's result
    # @rput Sigma X true_mu
    # R"""
    # library(knockoff)
    # diag_s = create.solve_sdp(Sigma)
    # X_ko = create.gaussian(X, true_mu, Sigma, method = "sdp", diag_s = diag_s)
    # """
    # @rget diag_s X_ko
    # [diag_s s]
    # histogram(vec(X_ko))
    # histogram(vec(X̃))

    # test properties
    @test all(s .≥ 0)
    @test all(1 .≥ s) # this is true since Σ has diagonal entries 1
    @test isposdef(Σ)
    λmin = eigmin(2Σ - Diagonal(s))
    @test λmin ≥ 0 || isapprox(λmin, 0, atol=1e-8)
end

@testset "model X 2nd order Knockoffs" begin
    # example from https://github.com/msesia/knockoff-filter/blob/master/R/knockoff/R/create_gaussian.R

    # simulate matrix
    Random.seed!(2022)
    n = 100
    p = 200
    ρ = 0.4
    Sigma = Matrix(SymmetricToeplitz(ρ.^(0:(p-1))))
    L = cholesky(Sigma).L
    X = randn(n, p) * L # var(X) = L var(N(0, 1)) L' = var(Σ)

    # generate knockoff
    true_mu = zeros(p)
    @time knockoff = modelX_gaussian_knockoffs(X, :sdp)
    X = knockoff.X
    X̃ = knockoff.X̃
    s = knockoff.s
    Σ = knockoff.Σ

    # compare with Matteo's result
    # @rput Sigma X true_mu
    # R"""
    # library(knockoff)
    # diag_s = create.solve_sdp(Sigma)
    # X_ko = create.gaussian(X, true_mu, Sigma, method = "sdp", diag_s = diag_s)
    # """
    # @rget diag_s X_ko
    # [diag_s s]
    # histogram(vec(X_ko))
    # histogram(vec(X̃))

    # test properties
    @test all(s .≥ 0)
    @test isposdef(Σ)
    λmin = eigmin(2Σ - Diagonal(s))
    @test λmin ≥ 0 || isapprox(λmin, 0, atol=1e-8)
end

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
    X = Knockoffs.sample_DMC(q, Q, n=samples)

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

@testset "SDP vs MVR vs ME knockoffs" begin
    # This is example 1 from https://amspector100.github.io/knockpy/mrcknock.html 
    # SDP knockoffs are provably powerless in this situation, while MVR and ME knockoffs have high power

    seed = 2022

    # simulate X
    Random.seed!(seed)
    n = 600
    p = 300
    ρ = 0.5
    Σ = (1-ρ) * I + ρ * ones(p, p)
    μ = zeros(p)
    L = cholesky(Σ).L
    X = randn(n, p) * L # var(X) = L var(N(0, 1)) L' = var(Σ)

    # simulate y
    Random.seed!(seed)
    k = Int(0.2p)
    βtrue = zeros(p)
    βtrue[1:k] .= rand(-1:2:1, k) .* rand(Uniform(0.5, 1), k)
    shuffle!(βtrue)
    correct_position = findall(!iszero, βtrue)
    y = X * βtrue + randn(n)

    # solve s vector
    @time Xko_sdp = modelX_gaussian_knockoffs(X, :sdp, μ, Σ)
    @time Xko_maxent = modelX_gaussian_knockoffs(X, :maxent, μ, Σ)
    @time Xko_mvr = modelX_gaussian_knockoffs(X, :mvr, μ, Σ)

    # run lasso and then apply knockoff-filter to default FDR = 0.01, 0.05, 0.1, 0.25, 0.5
    @time sdp_filter = fit_lasso(y, Xko_sdp.X, Xko_sdp.X̃, debias=nothing)
    @time mvr_filter = fit_lasso(y, Xko_mvr.X, Xko_mvr.X̃, debias=nothing)
    @time me_filter = fit_lasso(y, Xko_maxent.X, Xko_maxent.X̃, debias=nothing)

    sdp_power, mvr_power, me_power = Float64[], Float64[], Float64[]
    for i in eachindex(sdp_filter.fdr_target)
        # extract beta for current fdr
        βsdp = sdp_filter.βs[i]
        βmvr = mvr_filter.βs[i]
        βme = me_filter.βs[i]
        
        # compute power and false discovery proportion
        push!(sdp_power, length(findall(!iszero, βsdp) ∩ correct_position) / k)
        push!(mvr_power, length(findall(!iszero, βmvr) ∩ correct_position) / k)
        push!(me_power, length(findall(!iszero, βme) ∩ correct_position) / k)
        # fdp = length(setdiff(findall(!iszero, βsdp), correct_position)) / max(count(!iszero, βsdp), 1)
        # push!(empirical_fdr, fdp)
    end

    @test all(mvr_power .> sdp_power)
    @test all(me_power .> sdp_power)
end

@testset "SDP vs SDP fast" begin
    seed = 2022

    # simulate X
    Random.seed!(seed)
    n = 400
    p = 300
    ρ = 0.4
    Sigma = Matrix(SymmetricToeplitz(ρ.^(0:(p-1)))) # true covariance matrix
    mu = zeros(p)
    L = cholesky(Sigma).L
    X = randn(n, p) * L # var(X) = L var(N(0, 1)) L' = var(Σ)

    @time Xko_sdp = modelX_gaussian_knockoffs(X, :sdp, mu, Sigma);
    @time Xko_sdp_fast = modelX_gaussian_knockoffs(X, :sdp_fast, mu, Sigma)

    @test all(isapprox.(Xko_sdp.s, Xko_sdp_fast.s, atol=0.05))
end

@testset "keyword arguments" begin
    seed = 2022

    # simulate X
    Random.seed!(seed)
    n = 400
    p = 300
    ρ = 0.4
    Sigma = Matrix(SymmetricToeplitz(ρ.^(0:(p-1)))) # true covariance matrix
    mu = zeros(p)
    L = cholesky(Sigma).L
    X = randn(n, p) * L # var(X) = L var(N(0, 1)) L' = var(Σ)

    # try supplying arguments to modelX_gaussian_knockoffs and fixed_knockoffs
    @time Xko_sdp_fast1 = modelX_gaussian_knockoffs(X, :sdp_fast, mu, Sigma, λ = 0.7, μ = 0.7)
    @time Xko_sdp_fast2 = modelX_gaussian_knockoffs(X, :sdp_fast, mu, Sigma, λ = 0.9, μ = 0.9)
    @test all(isapprox.(Xko_sdp_fast1.s, Xko_sdp_fast2.s, atol=0.05))
end

@testset "debiasing preserves sparsity pattern" begin
    seed = 2022

    # simulate x
    n = 1000
    p = 500
    Random.seed!(seed)
    ρ = 0.4
    Σ = Matrix(SymmetricToeplitz(ρ.^(0:(p-1)))) # true covariance matrix
    μ = zeros(p) # true mean parameters
    L = cholesky(Σ).L
    X = randn(n, p) * L # var(X) = L var(N(0, 1)) L' = var(Σ)
    # X = zscore(X, mean(X, dims=1), std(X, dims=1)) # center/scale Xj to mean 0 var 1

    # simulate y
    Random.seed!(seed)
    k = 50
    ϵ = Normal(0, 1)
    d = Normal(0, 1)
    β = zeros(p)
    β[1:k] .= rand(d, k)
    shuffle!(β)
    y = X * β + rand(ϵ, n) |> Vector{eltype(X)}

    # generate knockoffs
    @time Xko = modelX_gaussian_knockoffs(X, :sdp, μ, Σ)

    # run lasso, followed up by debiasing
    Random.seed!(seed)
    @time nodebias = fit_lasso(y, Xko.X, Xko.X̃, debias=nothing)
    Random.seed!(seed)
    @time yesdebias = fit_lasso(y, Xko.X, Xko.X̃, debias=:ls)

    # check that debiased result have same support as not debiasing
    for i in eachindex(nodebias.fdr_target)
        @test issubset(findall(!iszero, yesdebias.βs[i]), findall(!iszero, nodebias.βs[i]))
    end
end

@testset "approximate constructions" begin
    # simulate data
    Random.seed!(2022)
    n = 100
    p = 500
    ρ = 0.4
    Sigma = Matrix(SymmetricToeplitz(ρ.^(0:(p-1))))
    L = cholesky(Sigma).L
    X = randn(n, p) * L # var(X) = L var(N(0, 1)) L' = var(Σ)
    true_mu = zeros(p)

    # ASDP (fixed window ranges)
    @time asdp = approx_modelX_gaussian_knockoffs(X, :sdp, windowsize = 99)
    λmin = eigmin(2*asdp.Σ - Diagonal(asdp.s))
    @test λmin ≥ 0 || isapprox(λmin, 0, atol=1e-8)

    # AMVR (arbitrary window ranges)
    window_ranges = [1:99, 100:121, 122:444, 445:500]
    @time amvr = approx_modelX_gaussian_knockoffs(X, :mvr, window_ranges);
    λmin = eigmin(2*amvr.Σ - Diagonal(amvr.s))
    @test λmin ≥ 0 || isapprox(λmin, 0, atol=1e-8)
end

@testset "fit lasso Gaussian" begin
    # simulate data
    Random.seed!(2022)
    n = 100 # sample size
    p = 500 # number of predictors
    k = 10 # number of causal predictors
    X = randn(n, p)
    b = zeros(p)
    b[1:k] .= randn(k)
    shuffle!(b)
    y = X * b + randn(n)

    # debias with least squares
    ko = fit_lasso(y, X, debias=:ls)
    @test length(ko.βs) == length(ko.a0)
    for i in 1:length(ko.βs)
        # println(norm(ko.βs[i] - b))
        @test norm(ko.βs[i] - b) < 1.5
    end
    # idx = findall(!iszero, b)
    # [ko.βs[5][idx] b[idx]]
    
    # debias with lasso
    ko = fit_lasso(y, X, debias=:lasso)
    @test length(ko.βs) == length(ko.a0)
    for i in 1:length(ko.βs)
        # println(norm(ko.βs[i] - b))
        @test norm(ko.βs[i] - b) < 1.1
    end

    # no debias
    ko = fit_lasso(y, X, debias=nothing)
    @test length(ko.βs) == length(ko.a0)
    for i in 1:length(ko.βs)
        # println(norm(ko.βs[i] - b))
        @test norm(ko.βs[i] - b) < 2
    end
end

@testset "fit lasso logistic" begin
    # simulate data
    Random.seed!(2022)
    n = 1000 # sample size
    p = 500 # number of predictors
    k = 10 # number of causal predictors
    X = randn(n, p)
    b = zeros(p)
    b[1:k] .= randn(k)
    shuffle!(b)
    μ = GLM.linkinv.(LogitLink(), X * b)
    y = [rand(Bernoulli(μi)) for μi in μ] |> Vector{Float64}

    # debias with least squares
    ls_ko = fit_lasso(y, X, d = Binomial(), debias=:ls)
    @test length(ls_ko.βs) == length(ls_ko.a0)
    for i in 1:length(ls_ko.βs)
        @test norm(ls_ko.βs[i] - b) < 1
    end
    
    # debias with lasso
    lasso_ko = fit_lasso(y, X, d = Binomial(), debias=:lasso)
    @test length(lasso_ko.βs) == length(lasso_ko.a0)
    for i in 1:length(lasso_ko.βs)
        @test norm(lasso_ko.βs[i] - b) < 5
    end

    # no debias
    nodebias_ko = fit_lasso(y, X, d = Binomial(), debias=nothing)
    @test length(nodebias_ko.βs) == length(nodebias_ko.a0)
    for i in 1:length(nodebias_ko.βs)
        @test norm(nodebias_ko.βs[i] - b) < 5
    end

    # visually compare estimated effect sizes (least squares > nodebias > lasso)
    # idx = findall(!iszero, b)
    # [ls_ko.βs[5][idx] lasso_ko.βs[5][idx] nodebias_ko.βs[5][idx] b[idx]]
end

@testset "predict via knockoff-filter" begin
    # simulate data
    Random.seed!(2022)
    n = 200 # sample size
    p = 500 # number of predictors
    k = 10 # number of causal predictors
    X = randn(n, p)
    b = zeros(p)
    b[1:k] .= randn(k)
    shuffle!(b)
    y = X * b + randn(n)
    Xtest = randn(n, p)
    ytest = Xtest * b + randn(n)

    # generate knockoffs and predict with debiased beta for each target FDR
    ko = fit_lasso(y, X, debias=:ls)
    ŷs = Knockoffs.predict(ko, Xtest)
    for i in 1:length(ko.βs)
        # println("R2 = $(R2(ŷs[i], ytest))")
        @test R2(ŷs[i], ytest) > 0.5
    end

    ko = fit_lasso(y, X, debias=:lasso)
    ŷs = Knockoffs.predict(ko, Xtest)
    for i in 1:length(ko.βs)
        # println("R2 = $(R2(ŷs[i], ytest))")
        @test R2(ŷs[i], ytest) > 0.5
    end

    ko = fit_lasso(y, X, debias=nothing)
    ŷs = Knockoffs.predict(ko, Xtest)
    for i in 1:length(ko.βs)
        # println("R2 = $(R2(ŷs[i], ytest))")
        @test R2(ŷs[i], ytest) > 0.5
    end
end

@testset "group knockoffs" begin
    # test if inverse_mat_sqrt is working
    x = rand(10, 10)
    A = Symmetric(x' * x)
    Ainvsqrt = Knockoffs.inverse_mat_sqrt(A)
    @test all(isapprox.(Ainvsqrt^2 * A - Matrix(I, 10, 10), 0, atol=1e-10))

    # simulate some data
    m = 200 # number of groups
    pi = 5  # features per group
    k = 20  # number of causal groups
    ρ = 0.4 # within group correlation
    γ = 0.2 # between group correlation
    p = m * pi # number of features
    n = 1000 # sample size
    groups = repeat(1:m, inner=5)
    Σ = simulate_block_covariance(groups, ρ, γ)
    true_mu = zeros(p)
    L = cholesky(Σ).L
    X = randn(n, p) * L
    zscore!(X, mean(X, dims=1), std(X, dims=1));

    # exact group knockoffs
    @time ko_equi = modelX_gaussian_group_knockoffs(X, groups, :equi, Σ, true_mu)
    S = ko_equi.S
    @test all(x -> x ≥ 0 || x ≈ 0, eigvals(Matrix(2Σ - S)))
    @test all(x -> x == (pi, pi), size.(S.blocks))

    @time ko_sdp = modelX_gaussian_group_knockoffs(X, groups, :sdp, Σ, true_mu)
    S = ko_sdp.S
    @test all(x -> x ≥ 0 || x ≈ 0, eigvals(Matrix(2Σ - S)))
    @test all(x -> x == (pi, pi), size.(S.blocks))

    # second order knockoffs
    @time ko_equi = modelX_gaussian_group_knockoffs(X, groups, :equi)
    S = ko_equi.S
    @test all(x -> x ≥ 0 || x ≈ 0, eigvals(Matrix(2Σ - S)))
    @test all(x -> x == (pi, pi), size.(S.blocks))

    @time ko_sdp = modelX_gaussian_group_knockoffs(X, groups, :sdp)
    S = ko_sdp.S
    @test all(x -> x ≥ 0 || x ≈ 0, eigvals(Matrix(2Σ - S)))
    @test all(x -> x == (pi, pi), size.(S.blocks))
end

@testset "group fit_lasso" begin
    # simulate some data
    m = 200 # number of groups
    pi = 5  # features per group
    k = 20  # number of causal groups
    ρ = 0.4 # within group correlation
    γ = 0.2 # between group correlation
    p = m * pi # number of features
    n = 1000 # sample size

    groups = repeat(1:m, inner=5)
    Σ = simulate_block_covariance(groups, ρ, γ)
    true_mu = zeros(p)
    L = cholesky(Σ).L
    X = randn(n, p) * L
    zscore!(X, mean(X, dims=1), std(X, dims=1));

    βtrue = zeros(m*pi)
    βtrue[1:k] .= rand(-1:2:1, k) .* 3.5
    shuffle!(βtrue)
    ϵ = randn(n)
    y = X * βtrue + ϵ;

    # no debias
    Random.seed!(2022)
    @time ko_filter1 = fit_lasso(y, X, method=:equi, groups=groups, debias=nothing)
    # debias with least squares (stringent)
    Random.seed!(2022)
    @time ko_filter2 = fit_lasso(y, X, method=:equi, groups=groups, debias=:ls, stringent=true)
    # debias with lasso (not stringent
    Random.seed!(2022)
    @time ko_filter3 = fit_lasso(y, X, method=:equi, groups=groups, debias=:lasso, stringent=false)

    nz_1 = count.(!iszero, ko_filter1.βs)
    nz_2 = count.(!iszero, ko_filter2.βs)
    nz_3 = count.(!iszero, ko_filter3.βs)
    @test all(nz_1 .== nz_2 .≤ nz_3)
end
