using Knockoffs
using Test
using LinearAlgebra
using Random
using StatsBase
using Statistics
using Distributions
using ToeplitzMatrices
# using RCall # for comparing with Matteo's knockoffs

@testset "fixed knockoffs" begin
    Random.seed!(2021)

    # simulate matrix and normalize columns
    n = 1000
    p = 100
    X = randn(n, p)

    # SDP knockoff
    @time knockoff = fixed_knockoffs(X, :sdp)
    X = knockoff.X
    Xko = knockoff.Xko
    s = knockoff.s
    Sigma = knockoff.Sigma

    # basic knockoff properties
    all(isapprox.(X'*X, Sigma, atol=1e-8))
    all(isapprox.(Xko'*Xko, Sigma, atol=1e-8))
    all(isapprox.(X'*Xko, Sigma - Diagonal(s), atol=1e-8))

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
    # histogram(vec(Xko))

    @test all(X' * X .≈ Sigma)
    @test all(isapprox.(Xko' * Xko, Sigma, atol=5e-1))
    @test all(s .≥ 0)
    @test all(1 .≥ s)
    λ = eigvals(2Sigma - Diagonal(s))
    for λi in λ
        @test λi ≥ 0 || isapprox(λi, 0, atol=1e-8)
    end
    # @test all(isapprox.(Ũ' * X, 0, atol=1e-10))
    for i in 1:p, j in 1:p
        if i == j
            # @test isapprox(dot(X[:, i], X[:, i]), 1, atol=1e-1)
            # @test isapprox(dot(Xko[:, i], Xko[:, i]), 1, atol=1e-1)
            @test isapprox(dot(X[:, i], Xko[:, i]), Sigma[i, i] - s[i], atol=1e-8)
            @test isapprox(dot(X[:, i], Xko[:, i]), 1 - s[i], atol=1e-8)
        else
            @test dot(X[:, i], Xko[:, j]) ≈ dot(X[:, i], X[:, j])
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
    true_mu = zeros(p)
    X = rand(MvNormal(true_mu, Sigma), n)' |> Matrix

    # generate knockoff
    @time knockoff = modelX_gaussian_knockoffs(X, :sdp, true_mu, Sigma)
    X = knockoff.X
    Xko = knockoff.Xko
    s = knockoff.s
    Sigma = knockoff.Sigma

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
    # histogram(vec(Xko))

    # test properties
    @test all(s .≥ 0)
    @test all(1 .≥ s) # this is true since Sigma has diagonal entries 1
    @test isposdef(Sigma)
    λmin = eigmin(2Sigma - Diagonal(s))
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
    true_mu = zeros(p)
    X = rand(MvNormal(true_mu, Sigma), n)' |> Matrix

    # generate knockoff
    @time knockoff = modelX_gaussian_knockoffs(X, :sdp)
    X = knockoff.X
    Xko = knockoff.Xko
    s = knockoff.s
    Sigma = knockoff.Sigma

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
    # histogram(vec(Xko))

    # test properties
    @test all(s .≥ 0)
    @test isposdef(Sigma)
    λmin = eigmin(2Sigma - Diagonal(s))
    @test λmin ≥ 0 || isapprox(λmin, 0, atol=1e-8)
end

@testset "utility functions" begin
    # thresholds
    w = [0.1, 1.9, 1.3, 1.8, 0.8, -0.7, -0.1]
    @test threshold(w, 0.2, :knockoff) == 0.8
    @test threshold(w, 0.2, :knockoff_plus) == Inf
    w = [0.27, 0.76, 0.21, 0.1, -0.38, -0.01]
    @test threshold(w, 0.4, :knockoff) == 0.1
    @test threshold(w, 0.5, :knockoff_plus) == 0.1
    w = [0.74, -0.65, -0.83, -0.27, -0.19, 0.4]
    @test threshold(w, 0.25, :knockoff) == Inf
    @test threshold(w, 0.25, :knockoff_plus) == Inf

    # merge_knockoffs_with_original
    # X = randn(500, 200)
    # Xko = randn(500, 800)
    # # Xfull, original, knockoff = merge_knockoffs_with_original(X, Xko)
    # merged = merge_knockoffs_with_original(X, Xko)
    # @test size(merged.XXko) == (500, 1000)
    # @test length(merged.original) == 200
    # @test length(merged.knockoff) == 200
    # @test all(merged.XXko[:, merged.original] .== X)

    # hc_partition_groups: based on X or Sigma
    n = 500
    p = 500
    Sigma = simulate_AR1(p, a=3, b=1)
    X = rand(MvNormal(Sigma), n)' |> Matrix
    groups = hc_partition_groups(Symmetric(Sigma), cutoff=1)
    @test all(groups .== collect(1:p))
    groups = hc_partition_groups(X, cutoff=0)
    @test all(groups .== 1)
    groups = hc_partition_groups(Symmetric(Sigma), cutoff=0.7, linkage=:single)
    # check that between group correlation is below h (must be the case with single linkage)
    unique_groups = unique(groups)
    for g1 in unique_groups, g2 in unique_groups
        g1 == g2 && continue
        idx1 = findall(x -> x == g1, groups)
        idx2 = findall(x -> x == g2, groups)
        for u in idx1, v in idx2
            @test Sigma[u, v] ≤ 0.7
        end
    end

    # id_partition_groups: based on X or Sigma
    groups = id_partition_groups(Symmetric(Sigma))
    groups = id_partition_groups(X)
    @test length(groups) == size(X, 2)
end

@testset "MK_statistics" begin
    # single knockoffs
    beta = [1.0, 0.2, -0.3, 0.8, -0.1, 0.5]
    betako = [0.8, 0.4, -0.2, 0.8, 0.1, 0.0]
    w = MK_statistics(beta, betako)
    @test length(w) == 6
    @test w[1] ≈ 0.2
    @test w[2] ≈ -0.2
    @test w[3] ≈ 0.1
    @test w[4] ≈ 0.0
    @test w[5] ≈ 0.0
    @test w[6] ≈ 0.5

    # multiple knockoffs
    T0 = [1.0, 0.2, -0.3, 0.8, -0.1, 0.0]
    T1 = [0.8, -0.4, 0.2, -0.6, 0.1, 0.0]
    T2 = [0.5, 0.2, -0.3, 0.3, -0.5, 0.0]
    T3 = [0.6, 0.4, -0.3, 0.3, 0.1, 0.2]
    κ, τ, w = MK_statistics(T0, [T1, T2, T3])
    @test length(κ) == length(τ) == length(w) == 6
    @test w[1] ≈ 0.4 # 1 - median(0.8, 0.5, 0.6)
    @test w[2] ≈ 0.0
    @test w[3] ≈ 0.0
    @test w[4] ≈ 0.5
    @test w[5] ≈ 0.0
    @test w[6] ≈ 0.0
    @test all(κ .== [0, 3, 0, 0, 2, 3])
    @test all(τ .== [0.4, 0.2, 0.0, 0.5, 0.4, 0.2])
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
    Xko = zeros(Int, samples, p)
    N = zeros(p, K)
    d = Categorical([1 / K for _ in 1:K])
    for i in 1:samples
        markov_knockoffs!(@view(Xko[i, :]), @view(X[i, :]), N, d, Q, q) 
    end

    # Check column means match
    Xmean = mean(X, dims=1)
    Xkomean = mean(Xko, dims=1)
    for i in 2:length(Xmean) # 1st entry might not match to 2 digits for some reason, this is the same in SNPknock
        @test isapprox(Xmean[i], Xkomean[i], atol=1e-2)
    end

    # Check that internal column correlations match
    for i in 2:p-1
        r1 = cor(@view(X[:, i]), @view(X[:, i+1]))
        r2 = cor(@view(Xko[:, i]), @view(Xko[:, i+1]))
        @test isapprox(r1, r2, atol=1e-2)
    end

    # Check that cross column correlations match
    for i in 2:p-1
        r1 = cor(@view(X[:, i]), @view(X[:, i+1]))
        r2 = cor(@view(X[:, i]), @view(Xko[:, i+1]))
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
    Sigma = (1-ρ) * I + ρ * ones(p, p)
    μ = zeros(p)
    X = rand(MvNormal(μ, Sigma), n)' |> Matrix

    # simulate y
    Random.seed!(seed)
    k = Int(0.2p)
    betatrue = zeros(p)
    betatrue[1:k] .= rand(-1:2:1, k) .* rand(Uniform(0.5, 1), k)
    shuffle!(betatrue)
    correct_position = findall(!iszero, betatrue)
    y = X * betatrue + randn(n)

    # solve s vector
    @time Xko_sdp = modelX_gaussian_knockoffs(X, :sdp, μ, Sigma)
    @time Xko_maxent = modelX_gaussian_knockoffs(X, :maxent, μ, Sigma)
    @time Xko_mvr = modelX_gaussian_knockoffs(X, :mvr, μ, Sigma)

    # run lasso and then apply knockoff-filter to default FDR = 0.01, 0.05, 0.1, 0.25, 0.5
    @time sdp_filter = fit_lasso(y, Xko_sdp, debias=nothing)
    @time mvr_filter = fit_lasso(y, Xko_mvr, debias=nothing)
    @time me_filter = fit_lasso(y, Xko_maxent, debias=nothing)

    sdp_power, mvr_power, me_power = Float64[], Float64[], Float64[]
    for i in eachindex(sdp_filter.fdr_target)
        # extract beta for current fdr
        betasdp = sdp_filter.betas[i]
        betamvr = mvr_filter.betas[i]
        betame = me_filter.betas[i]
        
        # compute power and false discovery proportion
        push!(sdp_power, length(findall(!iszero, betasdp) ∩ correct_position) / k)
        push!(mvr_power, length(findall(!iszero, betamvr) ∩ correct_position) / k)
        push!(me_power, length(findall(!iszero, betame) ∩ correct_position) / k)
        # fdp = length(setdiff(findall(!iszero, betasdp), correct_position)) / max(count(!iszero, betasdp), 1)
        # push!(empirical_fdr, fdp)
    end

    @test all(mvr_power .≥ sdp_power)
    @test all(me_power .≥ sdp_power)
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
    X = rand(MvNormal(mu, Sigma), n)' |> Matrix

    @time Xko_sdp = modelX_gaussian_knockoffs(X, :sdp, mu, Sigma);
    @time Xko_sdp_fast = modelX_gaussian_knockoffs(X, :sdp_ccd, mu, Sigma)

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
    X = rand(MvNormal(mu, Sigma), n)' |> Matrix

    # try supplying arguments to modelX_gaussian_knockoffs and fixed_knockoffs
    @time Xko_sdp_fast1 = modelX_gaussian_knockoffs(X, :sdp_ccd, mu, Sigma, λ = 0.7, μ = 0.7)
    @time Xko_sdp_fast2 = modelX_gaussian_knockoffs(X, :sdp_ccd, mu, Sigma, λ = 0.9, μ = 0.9)
    @test all(isapprox.(Xko_sdp_fast1.s, Xko_sdp_fast2.s, atol=0.05))
end

@testset "debiasing preserves sparsity pattern" begin
    seed = 2022

    # simulate x
    n = 1000
    p = 500
    Random.seed!(seed)
    ρ = 0.4
    Sigma = Matrix(SymmetricToeplitz(ρ.^(0:(p-1)))) # true covariance matrix
    μ = zeros(p) # true mean parameters
    X = rand(MvNormal(μ, Sigma), n)' |> Matrix

    # simulate y
    Random.seed!(seed)
    k = 50
    ϵ = Normal(0, 1)
    d = Normal(0, 1)
    beta = zeros(p)
    beta[1:k] .= rand(d, k)
    shuffle!(beta)
    y = X * beta + rand(ϵ, n) |> Vector{eltype(X)}

    # generate knockoffs
    @time Xko = modelX_gaussian_knockoffs(X, :maxent, μ, Sigma)

    # run lasso, followed up by debiasing
    Random.seed!(seed)
    @time nodebias = fit_lasso(y, Xko, debias=nothing)
    Random.seed!(seed)
    @time yesdebias = fit_lasso(y, Xko, debias=:ls)

    # check that debiased result have same support as not debiasing
    for i in eachindex(nodebias.fdr_target)
        @test issubset(findall(!iszero, yesdebias.betas[i]), findall(!iszero, nodebias.betas[i]))
    end
end

@testset "approximate constructions" begin
    # simulate data
    Random.seed!(2022)
    n = 100
    p = 500
    ρ = 0.4
    Sigma = Matrix(SymmetricToeplitz(ρ.^(0:(p-1))))
    true_mu = zeros(p)
    X = rand(MvNormal(true_mu, Sigma), n)' |> Matrix

    # ASDP (fixed window ranges)
    @time asdp = approx_modelX_gaussian_knockoffs(X, :sdp, windowsize = 99)
    λmin = eigvals(2*asdp.Sigma - Diagonal(asdp.s)) |> minimum
    @test λmin ≥ 0 || isapprox(λmin, 0, atol=1e-8)

    # AMVR (arbitrary window ranges)
    window_ranges = [1:99, 100:121, 122:444, 445:500]
    @time amvr = approx_modelX_gaussian_knockoffs(X, :mvr, window_ranges);
    λmin = eigvals(2*amvr.Sigma - Diagonal(amvr.s)) |> minimum
    @test λmin ≥ 0 || isapprox(λmin, 0, atol=1e-8)

    # AMVR (arbitrary window ranges, m=5)
    m = 5
    window_ranges = [1:99, 100:121, 122:444, 445:500]
    @time amvr = approx_modelX_gaussian_knockoffs(X, :mvr, window_ranges, m=m);
    λmin = eigvals((m+1)/m*amvr.Sigma - Diagonal(amvr.s)) |> minimum
    @test λmin ≥ 0 || isapprox(λmin, 0, atol=1e-8)
    @test eigmin(Diagonal(amvr.s)) ≥ 0
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
    ko = fit_lasso(y, X, debias=:ls);
    @test length(ko.betas) == length(ko.a0)
    for i in 1:length(ko.betas)
        @show norm(ko.betas[i] - b) # second best
    end
    # idx = findall(!iszero, b)
    # [ko.betas[5][idx] b[idx]]
    
    # debias with lasso
    ko = fit_lasso(y, X, debias=:lasso);
    @test length(ko.betas) == length(ko.a0)
    for i in 1:length(ko.betas)
        @show norm(ko.betas[i] - b) # best
    end

    # no debias
    ko = fit_lasso(y, X, debias=nothing);
    @test length(ko.betas) == length(ko.a0)
    for i in 1:length(ko.betas)
        @show norm(ko.betas[i] - b) # worst
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
    @test length(ls_ko.betas) == length(ls_ko.a0)
    for i in 1:length(ls_ko.betas)
        @show norm(ls_ko.betas[i] - b) # best
    end
    
    # debias with lasso
    lasso_ko = fit_lasso(y, X, d = Binomial(), debias=:lasso)
    @test length(lasso_ko.betas) == length(lasso_ko.a0)
    for i in 1:length(lasso_ko.betas)
        @show norm(lasso_ko.betas[i] - b) # second best
    end

    # no debias
    nodebias_ko = fit_lasso(y, X, d = Binomial(), debias=nothing)
    @test length(nodebias_ko.betas) == length(nodebias_ko.a0)
    for i in 1:length(nodebias_ko.betas)
        @show norm(nodebias_ko.betas[i] - b) # worst
    end

    # visually compare estimated effect sizes (least squares > nodebias > lasso)
    # idx = findall(!iszero, b)
    # [ls_ko.betas[5][idx] lasso_ko.betas[5][idx] nodebias_ko.betas[5][idx] b[idx]]
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
    ko = fit_lasso(y, X, debias=:ls, filter_method=:knockoff)
    ŷs = Knockoffs.predict(ko, Xtest)
    for i in 1:length(ko.betas)
        # println("R2 = $(R2(ŷs[i], ytest))")
        @test R2(ŷs[i], ytest) > 0.5
    end

    ko = fit_lasso(y, X, debias=:lasso, filter_method=:knockoff)
    ŷs = Knockoffs.predict(ko, Xtest)
    for i in 1:length(ko.betas)
        # println("R2 = $(R2(ŷs[i], ytest))")
        @test R2(ŷs[i], ytest) > 0.5
    end

    ko = fit_lasso(y, X, debias=nothing, filter_method=:knockoff)
    ŷs = Knockoffs.predict(ko, Xtest)
    for i in 1:length(ko.betas)
        # println("R2 = $(R2(ŷs[i], ytest))")
        @test R2(ŷs[i], ytest) > 0.5
    end
end

@testset "group knockoff optimization" begin
    # simulate some data
    groups = 10 # number of groups
    pi = 5  # features per group
    k = 10  # number of causal groups
    ρ = 0.4 # within group correlation
    γ = 0.2 # between group correlation
    p = groups * pi # number of features
    n = 1000 # sample size
    m = 5 # number of knockoffs per feature
    groups = repeat(1:groups, inner=5)
    Sigma = simulate_block_covariance(groups, ρ, γ)
    true_mu = zeros(p)
    X = rand(MvNormal(true_mu, Sigma), n)' |> Matrix
    zscore!(X, mean(X, dims=1), std(X, dims=1))
    Sigmacopy = copy(Sigma)
    groups_copy = copy(groups)

    # equi
    @time equi = modelX_gaussian_group_knockoffs(X, :equi, groups, true_mu, Sigma, m=m)
    @test all(x -> x ≥ 0 || x ≈ 0, eigvals(Symmetric((m+1)/m*Sigma - equi.S)))
    @test all(x -> x ≥ 0 || x ≈ 0, eigvals(Symmetric(equi.S)))

    # CCD (exact Gaussian constructions)
    inner_pca_iter = [1, 10, 1]
    inner_ccd_iter = [0, 5, 1]
    tol = 0.01
    for method in [:sdp, :mvr, :maxent]
        for (pca, ccd) in zip(inner_pca_iter, inner_ccd_iter)
            @time ko = modelX_gaussian_group_knockoffs(X, method, groups, 
                true_mu, Sigma, m=m, inner_pca_iter=pca, inner_ccd_iter=ccd, 
                tol = tol, verbose=true)
            # check constraints (compensating for numerical error)
            @test all(x -> x ≥ -1e-7, eigvals(Symmetric((m+1)/m*Sigma - ko.S)))
            @test all(x -> x ≥ -1e-7, eigvals(Symmetric(ko.S)))
            # check data integrity
            @test all(Sigma .== Sigmacopy)
            @test all(groups_copy .== groups)
            # check S has group-block-diagonal structure
            for idx in findall(!iszero, ko.S)
                i, j = getindex(idx, 1), getindex(idx, 2)
                @test groups[i] == groups[j]
            end
        end
    end

    # block updates (2nd order constructions)
    tol = 0.01
    for method in [:sdp_block, :mvr_block, :maxent_block]
        @time ko = modelX_gaussian_group_knockoffs(X, method, groups, 
            m=m, tol = tol, verbose=true)
        # check constraints (compensating for numerical error)
        @test all(x -> x ≥ -1e-7, eigvals(Symmetric((m+1)/m*Sigma - ko.S)))
        @test all(x -> x ≥ -1e-7, eigvals(Symmetric(ko.S)))
        # check data integrity
        @test all(Sigma .== Sigmacopy)
        @test all(groups_copy .== groups)
        # check S has group-block-diagonal structure
        for idx in findall(!iszero, ko.S)
            i, j = getindex(idx, 1), getindex(idx, 2)
            @test groups[i] == groups[j]
        end
    end

    # suboptimal
    for method in [:sdp_subopt, :sdp_subopt_correct]
        @time ko = modelX_gaussian_group_knockoffs(X, method, groups, true_mu, Sigma, m=m)
        # check constraints (compensating for numerical error)
        @test all(x -> x ≥ -1e-7, eigvals(Symmetric((m+1)/m*Sigma - ko.S)))
        @test all(x -> x ≥ -1e-7, eigvals(Symmetric(ko.S)))
        # check data integrity
        @test all(Sigma .== Sigmacopy)
        @test all(groups_copy .== groups)
        # check S has group-block-diagonal structure
        for idx in findall(!iszero, ko.S)
            i, j = getindex(idx, 1), getindex(idx, 2)
            @test groups[i] == groups[j]
        end
    end
end

@testset "block descent for a single block" begin
    p = 15
    groups = repeat(1:3, inner=5) # each group has 5 variables
    Sigma = Matrix(SymmetricToeplitz(0.4.^(0:(p-1)))) # true covariance matrix
    m = 1 # make just 1 knockoff per variable

    # initialize with equicorrelated solution
    Sequi, γ = solve_s_group(Symmetric(Sigma), groups, :equi)
    
    # form constraints for block 1
    Sigma11 = Sigma[1:5, 1:5]
    A = (m+1)/m * Sigma
    D = A - Sequi
    A11 = @view(A[1:5, 1:5])
    D12 = @view(D[1:5, 6:end])
    D22 = @view(D[6:end, 6:end])
    ub = A11 - D12 * inv(D22) * D12'
    
    # solve first block
    @time S1_new, success = Knockoffs.solve_group_SDP_single_block(Sigma11, ub)
    λmin = eigmin(S1_new)
    @test λmin ≥ 0 || isapprox(λmin, 0, atol=1e-8)
    λmin = eigmin(ub - S1_new)
    @test λmin ≥ 0 || isapprox(λmin, 0, atol=1e-8)

    # eyeball result
    # @show S1_new
    # @show sum(abs.(Sigma11 - S1_new))
end

@testset "group knockoff utilities" begin
    # test if inverse_mat_sqrt is working
    x = rand(10, 10)
    A = Symmetric(x' * x)
    Ainvsqrt = Knockoffs.inverse_mat_sqrt(A, tol=0)
    @test all(isapprox.(Ainvsqrt^2 * A - Matrix(I, 10, 10), 0, atol=1e-8))

    # test adjacency constrained hierachical clustering
    distmat = rand(4, 4)
    LinearAlgebra.copytri!(distmat, 'U')
    group1 = [1, 2]
    group2 = [3, 4]
    val, pos = findmin(distmat[group1, group2])
    @test val == Knockoffs.single_linkage_distance(distmat, group1, group2)

    # data for hc_partition_groups and id_partition_groups
    n = 100
    p = 500
    μ = zeros(p)
    Sigma = simulate_AR1(p, a=3, b=1)
    X = rand(MvNormal(μ, Sigma), n)' |> Matrix
    zscore!(X, mean(X, dims=1), std(X, dims=1))

    # defining groups: hierarchical clustering
    for force_contiguous in [true, false]
        groups1, reps1 = hc_partition_groups(X, 
            force_contiguous=force_contiguous, linkage=:single)
        groups2, reps2 = hc_partition_groups(Symmetric(Sigma), 
            force_contiguous=force_contiguous, linkage=:single)
        @test length(unique(groups1)) == length(reps1)
        @test length(unique(groups2)) == length(reps2)
        # test contiguity
        if force_contiguous
            for g in unique(groups1)
                idx = findall(x -> x == g, groups1)
                @test all(diff(idx) .== 1)
            end
        end
    end

    # defining groups: interpolative decomposition
    for force_contiguous in [true, false]
        groups1, reps1 = id_partition_groups(X, 
            force_contiguous=force_contiguous)
        groups2, reps2 = id_partition_groups(Symmetric(Sigma), 
            force_contiguous=force_contiguous)
        @test length(unique(groups1)) == length(reps1)
        @test length(unique(groups2)) == length(reps2)
        # test contiguity
        if force_contiguous
            for g in unique(groups1)
                idx = findall(x -> x == g, groups1)
                @test all(diff(idx) .== 1)
            end
        end
    end
end

@testset "representative group knockoffs" begin
    # simulate data
    p = 500
    k = 50
    n = 250
    Sigma = simulate_AR1(p, a=3, b=1)
    true_mu = zeros(p)
    X = rand(MvNormal(true_mu, Sigma), n)' |> Matrix
    zscore!(X, mean(X, dims=1), std(X, dims=1))

    # tests defining groups by ID and choosing representatives
    groups1 = id_partition_groups(X)
    groups2 = id_partition_groups(Symmetric(cor(X)))
    @test length(groups1) == length(groups2)
    group_reps1 = choose_group_reps(Symmetric(Sigma), groups1, threshold=0.5)
    group_reps2 = choose_group_reps(Symmetric(Sigma), groups1, threshold=0.7)
    group_reps3 = choose_group_reps(Symmetric(Sigma), groups1, threshold=0.9)
    @test issubset(group_reps1, group_reps2)
    @test issubset(group_reps2, group_reps3)

    # tests defining groups by ID and choosing representatives
    groups1 = hc_partition_groups(X)
    groups2 = hc_partition_groups(Symmetric(cor(X)))
    @test length(groups1) == length(groups2)
    group_reps1 = choose_group_reps(Symmetric(Sigma), groups1, threshold=0.5)
    group_reps2 = choose_group_reps(Symmetric(Sigma), groups1, threshold=0.7)
    group_reps3 = choose_group_reps(Symmetric(Sigma), groups1, threshold=0.9)
    @test issubset(group_reps1, group_reps2)
    @test issubset(group_reps2, group_reps3)

    # representative knockoffs based on conditional independence assumption
    rme = modelX_gaussian_rep_group_knockoffs(
        X, :maxent, groups1, true_mu, Sigma
    )
    @test size(rme.S11, 1) ≤ size(rme.S, 1) == p
    @test length(rme.group_reps) ≤ p

    # enforcing conditional independent assumption
    rme = modelX_gaussian_rep_group_knockoffs(
        X, :maxent, groups1, true_mu, Sigma, enforce_cond_indep=true
    )
    @test typeof(rme.S) <: AbstractMatrix
    Sblocks = Knockoffs.block_diagonalize(rme.S, groups1)
    @test count(x -> abs(x) ≥ 1e-8, rme.S - Sblocks) == 0 # test if rme.S is truly block diagonal
end

@testset "multiple knockoffs" begin
    Random.seed!(2022)
    n = 100 # sample size
    p = 500 # number of covariates
    ρ = 0.4
    Sigma = Matrix(SymmetricToeplitz(ρ.^(0:(p-1)))) # true covariance matrix
    μ = zeros(p) # true mean parameters
    X = rand(MvNormal(μ, Sigma), n)' |> Matrix

    # routine for solving s and generating knockoffs satisfy PSD constraint
    mvr_multiple = modelX_gaussian_knockoffs(X, :mvr, μ, Sigma, m=3)
    @test eigmin(4/3 * Sigma - Diagonal(mvr_multiple.s)) ≥ 0
    me_multiple = modelX_gaussian_knockoffs(X, :maxent, μ, Sigma, m=5)
    @test eigmin(6/5 * Sigma - Diagonal(me_multiple.s)) ≥ 0
    sdp_multiple = modelX_gaussian_knockoffs(X, :sdp, μ, Sigma, m=5)
    λmin = eigmin(6/5 * Sigma - Diagonal(sdp_multiple.s))
    @test λmin ≥ 0 || isapprox(λmin, 0, atol=1e-8)
    sdp_fast_multiple = modelX_gaussian_knockoffs(X, :sdp_ccd, μ, Sigma, m=5)
    λmin = eigmin(6/5 * Sigma - Diagonal(sdp_fast_multiple.s))
    @test λmin ≥ 0 || isapprox(λmin, 0, atol=1e-8)

    # Check lasso runs with multiple knockoffs
    k = 15
    betatrue = zeros(p)
    betatrue[1:k] .= randn(k)
    shuffle!(betatrue)
    correct_position = findall(!iszero, betatrue)
    y = X * betatrue + randn(n)
    @time mvr_filter = fit_lasso(y, X, method=:mvr, m=3, filter_method=:knockoff_plus)
    @time me_filter = fit_lasso(y, X, method=:maxent, m=5, filter_method=:knockoff_plus)

    @test size(mvr_filter.X) == (n, p)
    @test size(mvr_filter.ko.Xko) == (n, 3p)
    @test size(me_filter.X) == (n, p)
    @test size(me_filter.ko.Xko) == (n, 5p)
end

#todo: 
# + test multiple group knockoffs (W uses averaged importance score in each group)
# + test cholesky update is correct when v = ei + ej where i and j is far apart
# + test marginal case
# + test when all groups is unique, objective function functions correctly (paula) and the solver does non-grouped knockoff 
# + choosing group reps is correct
# + select_features for groups should work when certain groups are not represented
# + test code for running group knockoff statistics through MK_statistics
# + test that the "groups" vector is never permuted
