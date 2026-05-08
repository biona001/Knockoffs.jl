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

    # centering is keyword-controlled and preserves the fixed-X Gram conditions
    n = 31
    p = 10
    X = randn(n, p) .+ randn(1, p)
    centered = fixed_knockoffs(X, :equi; center=true)
    @test all(isapprox.(vec(sum(centered.X; dims=1)), 0, atol=1e-10))
    @test all(isapprox.(vec(sum(centered.Xko; dims=1)), 0, atol=1e-10))
    @test all(isapprox.(vec(sum(abs2, centered.X; dims=1)), 1, atol=1e-10))
    @test all(isapprox.(centered.X' * centered.X, centered.Sigma, atol=1e-8))
    @test all(isapprox.(centered.Xko' * centered.Xko, centered.Sigma, atol=1e-8))
    @test all(isapprox.(centered.X' * centered.Xko, centered.Sigma - Diagonal(centered.s), atol=1e-8))

    # When n = 2p, the centered nullspace has dimension p - 1, so the
    # knockoff basis cannot generally also be constrained to be centered.
    n = 20
    p = 10
    X = randn(n, p) .+ randn(1, p)
    centered_boundary = fixed_knockoffs(X, :equi; center=true)
    @test all(isapprox.(vec(sum(centered_boundary.X; dims=1)), 0, atol=1e-10))
    @test all(isapprox.(centered_boundary.Xko' * centered_boundary.Xko, centered_boundary.Sigma, atol=1e-8))
    @test all(isapprox.(centered_boundary.X' * centered_boundary.Xko, centered_boundary.Sigma - Diagonal(centered_boundary.s), atol=1e-8))
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

@testset "parallel max entropy solver" begin
    Random.seed!(2022)
    p = 60
    ρ = 0.4
    Sigma = Matrix(SymmetricToeplitz(ρ.^(0:(p-1))))
    s = solve_s(Symmetric(Sigma), :maxent_fast;
        niter=3, min_spacing=20, shuffle_offsets=false)

    @test all(s .≥ 0)
    @test all(1 .≥ s)
    λmin = eigmin(2Sigma - Diagonal(s))
    @test λmin ≥ 0 || isapprox(λmin, 0, atol=1e-8)

    s_string = solve_s(Symmetric(Sigma), "maxent_fast";
        niter=3, min_spacing=20, shuffle_offsets=false)
    @test s ≈ s_string
end

@testset "parallel MVR and SDP solvers" begin
    Random.seed!(2022)
    p = 60
    ρ = 0.4
    Sigma = Matrix(SymmetricToeplitz(ρ.^(0:(p-1))))

    for method in [:mvr_fast, :sdp_fast]
        s = solve_s(Symmetric(Sigma), method;
            niter=3, min_spacing=20, nworkers=4, shuffle_offsets=false)
        @test all(s .≥ 0)
        @test all(1 .≥ s)
        λmin = eigmin(2Sigma - Diagonal(s))
        @test λmin ≥ 0 || isapprox(λmin, 0, atol=1e-8)
    end

    @test solve_s(Symmetric(Sigma), :sdp; niter=3) ≈
        solve_s(Symmetric(Sigma), :sdp_ccd; niter=3)
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

@testset "multiple knockoff q-values" begin
    Random.seed!(2026)
    p = 60
    m = 3
    T0 = randn(p)
    Tk = [randn(p) for _ in 1:m]
    κ, τ, W = MK_statistics(T0, Tk)
    qvalues = get_knockoff_qvalue(κ, τ, m)

    @test length(qvalues) == p
    @test all((0 .<= qvalues) .& (qvalues .<= 1))

    for fdr in [0.01, 0.05, 0.1, 0.2, 0.4]
        tau_hat = mk_threshold(τ, κ, m, fdr)
        standard_selected = findall(x -> x ≥ tau_hat, W)
        qvalue_selected = findall(x -> x ≤ fdr, qvalues)
        @test standard_selected == qvalue_selected
    end
end

@testset "q-value selection matches standard selection on simulated data" begin
    Random.seed!(2026)
    n = 220
    p = 80
    m = 3
    ρ = 0.35
    Σ = Matrix(SymmetricToeplitz(ρ.^(0:(p-1))))
    μ = zeros(p)
    X = rand(MvNormal(μ, Σ), n)' |> Matrix

    β = zeros(p)
    signal_idx = sample(1:p, 12, replace=false)
    β[signal_idx] .= rand([-1.5, -1.0, 1.0, 1.5], 12)
    y = X * β + 0.8 .* randn(n)

    ko = modelX_gaussian_knockoffs(X, :equi, μ, Σ, m=m)
    fdrs = [0.02, 0.05, 0.1, 0.2, 0.3]
    marginal_filter = fit_marginal(y, ko)
    @test length(marginal_filter.qvalues) == p

    # Recompute threshold-based selections from (κ, τ, W)
    y_std = zscore(y, mean(y), std(y))
    X_std = zscore(X, mean(X, dims=1), std(X, dims=1))
    Xko_std = zscore(ko.Xko, mean(ko.Xko, dims=1), std(ko.Xko, dims=1))
    T0 = abs2.(X_std' * y_std) ./ n
    Tk = [abs2.(Transpose(@view(Xko_std[:, (k-1)*p+1:k*p])) * y_std) ./ n for k in 1:m]
    κ, τ, W = MK_statistics(T0, Tk)
    qvalues = get_knockoff_qvalue(κ, τ, m)

    for (i, fdr) in enumerate(fdrs)
        tau_hat = mk_threshold(τ, κ, m, fdr)
        standard_selected = findall(x -> x ≥ tau_hat, W)
        qvalue_selected = findall(x -> x ≤ fdr, qvalues)
        @test standard_selected == qvalue_selected
        @test select_variables(marginal_filter, fdr) == qvalue_selected
    end
end

@testset "select_groups returns group labels and member indices" begin
    Random.seed!(2027)
    n = 180
    p = 70
    Σ = Matrix(SymmetricToeplitz(0.3.^(0:(p-1))))
    μ = zeros(p)
    X = rand(MvNormal(μ, Σ), n)' |> Matrix
    zscore!(X, mean(X, dims=1), std(X, dims=1))
    groups = hc_partition_groups(X, cutoff=0.5)

    β = zeros(p)
    β[sample(1:p, 10, replace=false)] .= rand([-1.0, 1.0], 10)
    y = X * β + 0.7 .* randn(n)

    ko_filter = fit_lasso(y, X, μ, Σ, method=:equi, groups=groups, m=1)
    q = 0.2
    selected_stats = select_variables(ko_filter, q)
    selected_groups, selected_group_vars = select_groups(ko_filter, q)
    expected_groups = isnothing(ko_filter.stat_groups) ? selected_stats :
        ko_filter.stat_groups[selected_stats]
    @test selected_groups == expected_groups
    @test length(selected_groups) == length(selected_group_vars)
    for (g, vars) in zip(selected_groups, selected_group_vars)
        @test all(==(g), @view(ko_filter.ko.groups[vars]))
    end
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
    for q in [0.01, 0.05, 0.1, 0.25, 0.5]
        # extract beta for current q-value threshold
        betasdp, _ = selected_coefficients(sdp_filter, q; debias=nothing)
        betamvr, _ = selected_coefficients(mvr_filter, q; debias=nothing)
        betame, _ = selected_coefficients(me_filter, q; debias=nothing)
        
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

@testset "SDP CCD aliases" begin
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
    @time Xko_sdp_ccd = modelX_gaussian_knockoffs(X, :sdp_ccd, mu, Sigma)
    @time Xko_sdp_fast = modelX_gaussian_knockoffs(X, :sdp_fast, mu, Sigma,
        nworkers=4, min_spacing=100, shuffle_offsets=false)

    @test Xko_sdp.s ≈ Xko_sdp_ccd.s
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
    for q in [0.01, 0.05, 0.1, 0.25, 0.5]
        βn, _ = selected_coefficients(nodebias, q; debias=nothing)
        βd, _ = selected_coefficients(yesdebias, q; debias=:ls)
        @test issubset(findall(!iszero, βd), findall(!iszero, βn))
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
    for q in [0.01, 0.05, 0.1, 0.25, 0.5]
        β̂, _ = selected_coefficients(ko, q; debias=:ls)
        @show norm(β̂ - b) # second best
    end
    # idx = findall(!iszero, b)
    # [ko.betas[5][idx] b[idx]]
    
    # debias with lasso
    ko = fit_lasso(y, X, debias=:lasso);
    for q in [0.01, 0.05, 0.1, 0.25, 0.5]
        β̂, _ = selected_coefficients(ko, q; debias=:lasso)
        @show norm(β̂ - b) # best
    end

    # no debias
    ko = fit_lasso(y, X, debias=nothing);
    for q in [0.01, 0.05, 0.1, 0.25, 0.5]
        β̂, _ = selected_coefficients(ko, q; debias=nothing)
        @show norm(β̂ - b) # worst
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
    for q in [0.01, 0.05, 0.1, 0.25, 0.5]
        β̂, _ = selected_coefficients(ls_ko, q; debias=:ls)
        @show norm(β̂ - b) # best
    end
    
    # debias with lasso
    lasso_ko = fit_lasso(y, X, d = Binomial(), debias=:lasso)
    for q in [0.01, 0.05, 0.1, 0.25, 0.5]
        β̂, _ = selected_coefficients(lasso_ko, q; debias=:lasso)
        @show norm(β̂ - b) # second best
    end

    # no debias
    nodebias_ko = fit_lasso(y, X, d = Binomial(), debias=nothing)
    for q in [0.01, 0.05, 0.1, 0.25, 0.5]
        β̂, _ = selected_coefficients(nodebias_ko, q; debias=nothing)
        @show norm(β̂ - b) # worst
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
    ŷs = Knockoffs.predict(ko, Xtest, [0.01, 0.05, 0.1, 0.25, 0.5], debias=:ls)
    for i in eachindex(ŷs)
        # println("R2 = $(R2(ŷs[i], ytest))")
        @test R2(ŷs[i], ytest) > 0.5
    end

    ko = fit_lasso(y, X, debias=:lasso, filter_method=:knockoff)
    ŷs = Knockoffs.predict(ko, Xtest, [0.01, 0.05, 0.1, 0.25, 0.5], debias=:lasso)
    for i in eachindex(ŷs)
        # println("R2 = $(R2(ŷs[i], ytest))")
        @test R2(ŷs[i], ytest) > 0.5
    end

    ko = fit_lasso(y, X, debias=nothing, filter_method=:knockoff)
    ŷs = Knockoffs.predict(ko, Xtest, [0.01, 0.05, 0.1, 0.25, 0.5], debias=nothing)
    for i in eachindex(ŷs)
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

    # block updates (exact constructions with known mean/covariance)
    tol = 0.01
    for method in [:sdp_block, :mvr_block, :maxent_block]
        @time ko = modelX_gaussian_group_knockoffs(X, method, groups, 
            true_mu, Sigma, m=m, tol = tol, verbose=true)
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

    # data for hc_partition_groups
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

    # issue 71: https://github.com/biona001/Knockoffs.jl/issues/71
    Sigma = Symmetric([1.0 0.14138129031570185 -0.07530156753912527 0.1425092255960416 -0.03975762881225776 -0.03975762881225776; 
        0.14138129031570185 1.0 -0.06269073019504869 0.18796864772097638 -0.042632500750932216 -0.042632500750932216; 
        -0.07530156753912527 -0.06269073019504869 1.0 0.10393405423314547 0.8269051049537056 0.8269051049537056; 
        0.1425092255960416 0.18796864772097638 0.10393405423314547 1.0 0.1598138853324871 0.1598138853324871; 
        -0.03975762881225776 -0.042632500750932216 0.8269051049537056 0.1598138853324871 1.0 0.9999999999999998; 
        -0.03975762881225776 -0.042632500750932216 0.8269051049537056 0.1598138853324871 0.9999999999999998 1.0])
    groups = [1, 2, 3, 4, 3, 3]
    group_reps = choose_group_reps(Sigma, groups, threshold=0.5)
    @test group_reps == [1, 2, 3, 4]
    group_reps = choose_group_reps(Sigma, groups, threshold=0.999)
    @test group_reps == [1, 2, 3, 4, 5]

    # issue 72: 
    n = 10
    x = randn(n, n)
    x[:, 2] .= x[:, 1]
    x[:, 8] .= x[:, 10]
    x[:, 9] .= x[:, 10]
    Σ = Symmetric(cor(x))
    groups = hc_partition_groups(Σ)
    group_reps = choose_group_reps(Σ, groups, threshold=0.5)
    @test groups[1] == groups[2]
    @test groups[8] == groups[9] == groups[10]
    @test length(intersect(group_reps, 1:2)) ≤ 1 # only 1 variable from features 1-2 are selected
    @test length(intersect(group_reps, 8:10)) ≤ 1 # only 1 variable from features 8-10 are selected

    prioritize_idx = [2, 9] # prioritize features 2 and 9 to be representatives
    group_reps = choose_group_reps(Σ, groups, threshold=0.5, prioritize_idx=prioritize_idx)
    if groups[1] != groups[8]
        # if features [1, 2] belong to different group than [8, 9, 10]
        # then both 2 and 9 should be selected
        @test 2 ∈ group_reps
        @test 9 ∈ group_reps
    else
        # if [1, 2] and [8, 9, 10] are in the same group, then 9 may not get selected
        @test 2 ∈ group_reps
    end

    # same result with/without precomputed inverse covariance
    Σ = Symmetric(simulate_AR1(40, a=3, b=1))
    groups = repeat(1:8, inner=5)
    reps_noinv = choose_group_reps(Σ, groups, threshold=0.6)
    reps_inv = choose_group_reps(Σ, groups, threshold=0.6, Σinv=inv(Σ))
    @test reps_noinv == reps_inv

    # priority index should be retained among duplicate columns
    # (issue-72 style edge case, but forces a specific duplicate to survive)
    Σdup = Symmetric([1.0 0.14138129031570185 -0.07530156753912527 0.1425092255960416 -0.03975762881225776 -0.03975762881225776;
                      0.14138129031570185 1.0 -0.06269073019504869 0.18796864772097638 -0.042632500750932216 -0.042632500750932216;
                      -0.07530156753912527 -0.06269073019504869 1.0 0.10393405423314547 0.8269051049537056 0.8269051049537056;
                      0.1425092255960416 0.18796864772097638 0.10393405423314547 1.0 0.1598138853324871 0.1598138853324871;
                      -0.03975762881225776 -0.042632500750932216 0.8269051049537056 0.1598138853324871 1.0 0.9999999999999998;
                      -0.03975762881225776 -0.042632500750932216 0.8269051049537056 0.1598138853324871 0.9999999999999998 1.0])
    groupsdup = [1, 2, 3, 4, 3, 3]
    reps_priority_dup = choose_group_reps(Σdup, groupsdup, threshold=0.999, prioritize_idx=[6])
    @test 6 ∈ reps_priority_dup
    @test 5 ∉ reps_priority_dup

    # dimension mismatch check
    @test_throws Exception choose_group_reps(Symmetric(Matrix{Float64}(I, 3, 3)), [1, 1], threshold=0.5)
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
