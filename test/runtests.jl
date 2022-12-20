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
    @test all(isapprox.(X̃' * X̃, Σ, atol=5e-1))
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
    true_mu = zeros(p)
    X = rand(MvNormal(true_mu, Sigma), n)' |> Matrix

    # generate knockoff
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
    true_mu = zeros(p)
    X = rand(MvNormal(true_mu, Sigma), n)' |> Matrix

    # generate knockoff
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
    X = randn(500, 200)
    X̃ = randn(500, 800)
    # Xfull, original, knockoff = merge_knockoffs_with_original(X, X̃)
    merged = merge_knockoffs_with_original(X, X̃)
    @test size(merged.XX̃) == (500, 1000)
    @test length(merged.original) == 200
    @test length(merged.knockoff) == 200
    @test all(merged.XX̃[:, merged.original] .== X)

    # hc_partition_groups: based on X or Σ
    n = 500
    p = 500
    Σ = simulate_AR1(p, a=3, b=1)
    groups, rep_variables = hc_partition_groups(Symmetric(Σ), cutoff=1)
    @test all(groups .== collect(1:p))
    groups, rep_variables = hc_partition_groups(Symmetric(Σ), cutoff=0)
    @test all(groups .== 1)
    groups, rep_variables = hc_partition_groups(Symmetric(Σ), cutoff=0.7)
    @test length(unique(groups)) == length(rep_variables)
    X = rand(MvNormal(zeros(p), Σ), n)' |> Matrix
    nrep = 1
    groups, rep_variables = hc_partition_groups(X, cutoff=0.7, nrep=nrep)
    @test length(unique(groups)) == length(rep_variables)
    @test countmap(groups[rep_variables]) |> values |> maximum ≤ nrep

    # id_partition_groups: based on X or Σ
    nrep = 2
    groups, rep_variables = id_partition_groups(Symmetric(Σ), nrep=nrep)
    @test countmap(groups[rep_variables]) |> values |> maximum ≤ nrep
    X = rand(MvNormal(zeros(p), Σ), n)' |> Matrix
    groups, rep_variables = id_partition_groups(X, nrep=nrep)
    @test countmap(groups[rep_variables]) |> values |> maximum ≤ nrep
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

    # group knockoffs (use sum)
    β = [1.0, 0.2, -0.3, 0.8, -0.1, 0.5, 0.5, -0.1]
    groups = [1, 1, 1, 1, 2, 2, 2, 2]
    original = [1, 4, 6, 7]
    knockoff = [2, 3, 5, 8]
    w = coefficient_diff(β, groups, original, knockoff, compute_avg=false)
    @test length(w) == 2
    @test w[1] ≈ 1.8 - 0.5
    @test w[2] ≈ 1.0 - 0.2

    # group knockoffs (use average)
    β = [1.0, 0.2, -0.3, 0.8, -0.1, 0.5, 0.5, -0.1]
    groups = [1, 1, 1, 1, 2, 2, 2, 2]
    original = [1, 4, 6, 7]
    knockoff = [2, 3, 5, 8]
    w = coefficient_diff(β, groups, original, knockoff, compute_avg=true)
    @test length(w) == 2
    @test w[1] ≈ (1 + 0.8 - 0.2 - 0.3) / 2
    @test w[2] ≈ (0.5 + 0.5 - 0.1 - 0.1) / 2
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
    X = rand(MvNormal(μ, Σ), n)' |> Matrix

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
    @time sdp_filter = fit_lasso(y, Xko_sdp, debias=nothing)
    @time mvr_filter = fit_lasso(y, Xko_mvr, debias=nothing)
    @time me_filter = fit_lasso(y, Xko_maxent, debias=nothing)

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
    Σ = Matrix(SymmetricToeplitz(ρ.^(0:(p-1)))) # true covariance matrix
    μ = zeros(p) # true mean parameters
    X = rand(MvNormal(μ, Σ), n)' |> Matrix

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
    @time nodebias = fit_lasso(y, Xko, debias=nothing)
    Random.seed!(seed)
    @time yesdebias = fit_lasso(y, Xko, debias=:ls)

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
    true_mu = zeros(p)
    X = rand(MvNormal(true_mu, Sigma), n)' |> Matrix

    # ASDP (fixed window ranges)
    @time asdp = approx_modelX_gaussian_knockoffs(X, :sdp, windowsize = 99)
    λmin = eigvals(2*asdp.Σ - Diagonal(asdp.s)) |> minimum
    @test λmin ≥ 0 || isapprox(λmin, 0, atol=1e-8)

    # AMVR (arbitrary window ranges)
    window_ranges = [1:99, 100:121, 122:444, 445:500]
    @time amvr = approx_modelX_gaussian_knockoffs(X, :mvr, window_ranges);
    λmin = eigvals(2*amvr.Σ - Diagonal(amvr.s)) |> minimum
    @test λmin ≥ 0 || isapprox(λmin, 0, atol=1e-8)

    # AMVR (arbitrary window ranges, m=5)
    m = 5
    window_ranges = [1:99, 100:121, 122:444, 445:500]
    @time amvr = approx_modelX_gaussian_knockoffs(X, :mvr, window_ranges, m=m);
    λmin = eigvals((m+1)/m*amvr.Σ - Diagonal(amvr.s)) |> minimum
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
    @test length(ko.βs) == length(ko.a0)
    for i in 1:length(ko.βs)
        @show norm(ko.βs[i] - b) # second best
    end
    # idx = findall(!iszero, b)
    # [ko.βs[5][idx] b[idx]]
    
    # debias with lasso
    ko = fit_lasso(y, X, debias=:lasso);
    @test length(ko.βs) == length(ko.a0)
    for i in 1:length(ko.βs)
        @show norm(ko.βs[i] - b) # best
    end

    # no debias
    ko = fit_lasso(y, X, debias=nothing);
    @test length(ko.βs) == length(ko.a0)
    for i in 1:length(ko.βs)
        @show norm(ko.βs[i] - b) # worst
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
        @show norm(ls_ko.βs[i] - b) # best
    end
    
    # debias with lasso
    lasso_ko = fit_lasso(y, X, d = Binomial(), debias=:lasso)
    @test length(lasso_ko.βs) == length(lasso_ko.a0)
    for i in 1:length(lasso_ko.βs)
        @show norm(lasso_ko.βs[i] - b) # second best
    end

    # no debias
    nodebias_ko = fit_lasso(y, X, d = Binomial(), debias=nothing)
    @test length(nodebias_ko.βs) == length(nodebias_ko.a0)
    for i in 1:length(nodebias_ko.βs)
        @show norm(nodebias_ko.βs[i] - b) # worst
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
    ko = fit_lasso(y, X, debias=:ls, filter_method=:knockoff)
    ŷs = Knockoffs.predict(ko, Xtest)
    for i in 1:length(ko.βs)
        # println("R2 = $(R2(ŷs[i], ytest))")
        @test R2(ŷs[i], ytest) > 0.5
    end

    ko = fit_lasso(y, X, debias=:lasso, filter_method=:knockoff)
    ŷs = Knockoffs.predict(ko, Xtest)
    for i in 1:length(ko.βs)
        # println("R2 = $(R2(ŷs[i], ytest))")
        @test R2(ŷs[i], ytest) > 0.5
    end

    ko = fit_lasso(y, X, debias=nothing, filter_method=:knockoff)
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
    groups = 100 # number of groups
    pi = 5  # features per group
    k = 10  # number of causal groups
    ρ = 0.4 # within group correlation
    γ = 0.2 # between group correlation
    p = groups * pi # number of features
    n = 1000 # sample size
    m = 5 # number of knockoffs per feature
    groups = repeat(1:groups, inner=5)
    Σ = simulate_block_covariance(groups, ρ, γ)
    true_mu = zeros(p)
    X = rand(MvNormal(true_mu, Σ), n)' |> Matrix
    zscore!(X, mean(X, dims=1), std(X, dims=1));

    # exact group knockoffs
    @time equi = modelX_gaussian_group_knockoffs(X, :equi, groups, true_mu, Σ, m=m)
    @test all(x -> x ≥ 0 || x ≈ 0, eigvals((m+1)/m*Σ - equi.S))
    @test all(x -> x ≥ 0 || x ≈ 0, eigvals(equi.S))

    @time sdp = modelX_gaussian_group_knockoffs(X, :sdp, groups, true_mu, Σ, m=m)
    @test all(x -> x ≥ 0 || x ≈ 0, eigvals((m+1)/m*Σ - sdp.S))
    @test all(x -> x ≥ 0 || x ≈ 0, eigvals(sdp.S))

    # @time sdp_subopt = modelX_gaussian_group_knockoffs(X, :sdp_subopt, groups, true_mu, Σ, m=m)
    # @test all(x -> x ≥ 0 || x ≈ 0, eigvals((m+1)/m*Σ - sdp_subopt.S))
    # @test all(x -> x ≥ 0 || x ≈ 0, eigvals(sdp_subopt.S))

    # @time sdp_subopt_correct = modelX_gaussian_group_knockoffs(X, :sdp_subopt_correct, groups, true_mu, Σ, m=m)
    # @test all(x -> x ≥ 0 || x ≈ 0, eigvals((m+1)/m*Σ - sdp_subopt_correct.S))
    # @test all(x -> x ≥ 0 || x ≈ 0, eigvals(sdp_subopt_correct.S))

    @time mvr = modelX_gaussian_group_knockoffs(X, :mvr, groups, true_mu, Σ, m=m, tol=0.001, verbose=true)
    @test all(x -> x ≥ 0 || x ≈ 0, eigvals((m+1)/m*Σ - mvr.S))
    @test all(x -> x ≥ 0 || x ≈ 0, eigvals(mvr.S))

    @time me = modelX_gaussian_group_knockoffs(X, :maxent, groups, true_mu, Σ, m=m, tol=0.001, verbose=true)
    @test all(x -> x ≥ 0 || x ≈ 0, eigvals((m+1)/m*Σ - me.S))
    @test all(x -> x ≥ 0 || x ≈ 0, eigvals(me.S))

    # second order knockoffs
    @time equi = modelX_gaussian_group_knockoffs(X, :equi, groups, m=m)
    @test all(x -> x ≥ 0 || x ≈ 0, eigvals((m+1)/m*Σ - equi.S))
    @test all(x -> x ≥ 0 || x ≈ 0, eigvals(equi.S))

    @time me = modelX_gaussian_group_knockoffs(X, :maxent, groups, m=m)
    @test all(x -> x ≥ 0 || x ≈ 0, eigvals((m+1)/m*Σ - me.S))
    @test all(x -> x ≥ 0 || x ≈ 0, eigvals(me.S))

    # test adjacency constrained hierachical clustering
    distmat = rand(4, 4)
    LinearAlgebra.copytri!(distmat, 'U')
    group1 = [1, 2]
    group2 = [3, 4]
    val, pos = findmin(distmat[group1, group2])
    @test val == Knockoffs.single_linkage_distance(distmat, group1, group2)

    # test all groups in adj_constrained_hclust are contiguous
    n = 100
    p = 500
    μ = zeros(p)
    Σ = simulate_AR1(p, a=3, b=1)
    X = rand(MvNormal(μ, Σ), n)' |> Matrix
    zscore!(X, mean(X, dims=1), std(X, dims=1))
    distmat = cor(X)
    @inbounds @simd for i in eachindex(distmat)
        distmat[i] = 1 - abs(distmat[i])
    end
    groups = Knockoffs.adj_constrained_hclust(distmat, h=0.3)
    for g in unique(groups)
        idx = findall(x -> x == g, groups)
        @test all(diff(idx) .== 1)
    end
end

@testset "representative group knockoffs" begin
    # simulate data
    p = 500
    k = 50
    n = 250
    Σ = simulate_AR1(p, a=3, b=1)
    true_mu = zeros(p)
    X = rand(MvNormal(true_mu, Σ), n)' |> Matrix
    zscore!(X, mean(X, dims=1), std(X, dims=1))

    #
    # Some tests for defining groups and choosing representatives within groups
    #
    #Interpolative decomposition, selecting group reps by ID
    nrep = 3
    rep_method = :id
    groups1, group_reps = id_partition_groups(X, rep_method=rep_method, nrep=nrep)
    @test countmap(groups1[group_reps]) |> values |> collect |> maximum == 3
    groups1, group_reps = id_partition_groups(Symmetric(cor(X)), rep_method=rep_method, nrep=nrep)
    @test countmap(groups1[group_reps]) |> values |> collect |> maximum == 3

    #Interpolative decomposition, selecting group reps by Trevor's method
    nrep = 2
    rep_method = :rss
    groups2, group_reps = id_partition_groups(X, rep_method=rep_method, nrep=nrep)
    @test countmap(groups2[group_reps]) |> values |> collect |> maximum == 2

    #hierarchical clustering, using ID to choose reps
    nrep = 2
    rep_method = :id
    groups, group_reps = hc_partition_groups(X, rep_method=rep_method, nrep=nrep)
    @test countmap(groups[group_reps]) |> values |> collect |> maximum == 2

    #hierarchical clustering, using Trevor's method to choose reps
    nrep = 1
    rep_method = :rss
    groups2, group_reps2 = hc_partition_groups(X, rep_method=rep_method, nrep=nrep)
    groups2, group_reps2 = hc_partition_groups(Symmetric(cor(X)), rep_method=rep_method, nrep=nrep)
    @test countmap(groups2[group_reps2]) |> values |> collect |> maximum == 1

    # 
    # single representative = running single variant knockoffs
    #
    nrep = 1
    groups, group_reps = hc_partition_groups(Σ, cutoff=0.7, nrep=nrep)
    rme = modelX_gaussian_rep_group_knockoffs(
        X, :maxent, true_mu, Σ, 
        groups, group_reps, nrep=nrep
    )
    me = modelX_gaussian_knockoffs(
        X[:,group_reps], :maxent, true_mu[group_reps], Σ[group_reps, group_reps], 
        verbose=false, # whether to print informative intermediate results
    )
    @test all(me.s .≈ rme.ko.s)

    nrep = 5
    groups, group_reps = id_partition_groups(X, rep_method=:id, nrep=nrep)
    rme = modelX_gaussian_rep_group_knockoffs(
        X, :maxent, true_mu, Σ, 
        groups, group_reps, nrep=nrep
    )
    @test typeof(rme.ko.S) <: Matrix
    offdiag_nz_idx = findall(!iszero, rme.ko.S - Diagonal(rme.ko.S))
end

@testset "group fit_lasso" begin
    # simulate some data
    m = 100 # number of groups
    pi = 5  # features per group
    k = 10  # number of causal groups
    ρ = 0.4 # within group correlation
    γ = 0.2 # between group correlation
    p = m * pi # number of features
    n = 1000 # sample size

    groups = repeat(1:m, inner=5)
    Σ = simulate_block_covariance(groups, ρ, γ)
    true_mu = zeros(p)
    X = rand(MvNormal(true_mu, Σ), n)' |> Matrix
    zscore!(X, mean(X, dims=1), std(X, dims=1));

    βtrue = zeros(m*pi)
    βtrue[1:k] .= rand(-1:2:1, k) .* 3.5
    shuffle!(βtrue)
    ϵ = randn(n)
    y = X * βtrue + ϵ;

    # no debias
    Random.seed!(2022)
    @time ko_filter1 = fit_lasso(y, X, method=:equi, groups=groups, debias=nothing, filter_method=:knockoff)
    # debias with least squares (stringent)
    Random.seed!(2022)
    @time ko_filter2 = fit_lasso(y, X, method=:equi, groups=groups, debias=:ls, stringent=true, filter_method=:knockoff)
    # debias with lasso (not stringent)
    Random.seed!(2022)
    @time ko_filter3 = fit_lasso(y, X, method=:equi, groups=groups, debias=:lasso, stringent=false, filter_method=:knockoff)

    nz_1 = count.(!iszero, ko_filter1.βs)
    nz_2 = count.(!iszero, ko_filter2.βs)
    nz_3 = count.(!iszero, ko_filter3.βs)
    @test all(nz_1 .== nz_2 .≤ nz_3)
end

@testset "multiple knockoffs" begin
    Random.seed!(2022)
    n = 100 # sample size
    p = 500 # number of covariates
    ρ = 0.4
    Σ = Matrix(SymmetricToeplitz(ρ.^(0:(p-1)))) # true covariance matrix
    μ = zeros(p) # true mean parameters
    X = rand(MvNormal(μ, Σ), n)' |> Matrix

    # routine for solving s and generating knockoffs satisfy PSD constraint
    mvr_multiple = modelX_gaussian_knockoffs(X, :mvr, μ, Σ, m=3)
    @test eigmin(4/3 * Σ - Diagonal(mvr_multiple.s)) ≥ 0
    me_multiple = modelX_gaussian_knockoffs(X, :maxent, μ, Σ, m=5)
    @test eigmin(6/5 * Σ - Diagonal(me_multiple.s)) ≥ 0
    sdp_multiple = modelX_gaussian_knockoffs(X, :sdp, μ, Σ, m=5)
    λmin = eigmin(6/5 * Σ - Diagonal(sdp_multiple.s))
    @test λmin ≥ 0 || isapprox(λmin, 0, atol=1e-8)
    sdp_fast_multiple = modelX_gaussian_knockoffs(X, :sdp_ccd, μ, Σ, m=5)
    λmin = eigmin(6/5 * Σ - Diagonal(sdp_fast_multiple.s))
    @test λmin ≥ 0 || isapprox(λmin, 0, atol=1e-8)

    # Check lasso runs with multiple knockoffs
    k = 15
    βtrue = zeros(p)
    βtrue[1:k] .= randn(k)
    shuffle!(βtrue)
    correct_position = findall(!iszero, βtrue)
    y = X * βtrue + randn(n)
    @time mvr_filter = fit_lasso(y, X, method=:mvr, m=3, filter_method=:knockoff_plus)
    @time me_filter = fit_lasso(y, X, method=:maxent, m=5, filter_method=:knockoff_plus)

    @test size(mvr_filter.X) == (n, p)
    @test size(mvr_filter.ko.X̃) == (n, 3p)
    @test size(me_filter.X) == (n, p)
    @test size(me_filter.ko.X̃) == (n, 5p)
end

@testset "block descent SDP group knockoff" begin
    p = 15
    group_sizes = [5 for i in 1:div(p, 5)] # each group has 5 variables
    groups = vcat([i*ones(g) for (i, g) in enumerate(group_sizes)]...) |> Vector{Int}
    Σ = Matrix(SymmetricToeplitz(0.4.^(0:(p-1)))) # true covariance matrix
    m = 1 # make just 1 knockoff per variable

    # initialize with equicorrelated solution
    Sequi, γ = solve_s_group(Σ, groups, :equi)
    
    # form constraints for block 1
    Σ11 = Σ[1:5, 1:5]
    A = (m+1)/m * Σ
    D = A - Sequi
    A11 = @view(A[1:5, 1:5])
    D12 = @view(D[1:5, 6:end])
    D22 = @view(D[6:end, 6:end])
    ub = A11 - D12 * inv(D22) * D12'
    
    # solve first block
    @time S1_new, success = Knockoffs.solve_group_SDP_single_block(Σ11, ub)
    λmin = eigmin(S1_new)
    @test λmin ≥ 0 || isapprox(λmin, 0, atol=1e-8)
    λmin = eigmin(ub - S1_new)
    @test λmin ≥ 0 || isapprox(λmin, 0, atol=1e-8)

    # eyeball result
    # @show S1_new
    # @show sum(abs.(Σ11 - S1_new))
end
