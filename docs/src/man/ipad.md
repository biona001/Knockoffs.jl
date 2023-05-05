
# IPAD knockoffs

This tutorial generates knockoffs based on the [intertwined probabilistic factors decoupling (IPAD)](https://www.tandfonline.com/doi/full/10.1080/01621459.2019.1654878) method, described in the following paper

> Fan, Yingying, Jinchi Lv, Mahrad Sharifvaghefi, and Yoshimasa Uematsu. "IPAD: stable interpretable forecasting with knockoffs inference." Journal of the American Statistical Association 115, no. 532 (2020): 1822-1834.

As we will see shortly,

+ IPAD knockoffs are more powerful and much more efficient to generate than model-X MVR/ME/SDP knockoffs, if we can assume a low-dimensional factored model for the data matrix `X`. 
+ If `X` doesn't really assume a factored model, then IPAD knockoff's empirical FDR will be inflated, sometimes severely so.

Our comparison uses target FDR 10%,  Lasso coefficient difference statistic, and we explore 3 ways (ER, GR, VR) to choose the number of latent factors `r` for IPAD knockoffs.


```julia
# load packages
using Knockoffs
using LinearAlgebra
using Random
using StatsBase
using Statistics
using ToeplitzMatrices
using Distributions
using Random
using CSV, DataFrames

# some helper functions to compute power and empirical FDR
function TP(correct_groups, signif_groups)
    return length(signif_groups ∩ correct_groups) / max(1, length(correct_groups))
end
function TP(correct_groups, β̂, groups)
    signif_groups = get_signif_groups(β̂, groups)
    return TP(correct_groups, signif_groups)
end
function FDR(correct_groups, signif_groups)
    FP = length(signif_groups) - length(signif_groups ∩ correct_groups) # number of false positives
    return FP / max(1, length(signif_groups))
end
function FDR(correct_groups, β̂, groups)
    signif_groups = get_signif_groups(β̂, groups)
    return FDR(correct_groups, signif_groups)
end
function get_signif_groups(β, groups)
    correct_groups = Int[]
    for i in findall(!iszero, β)
        g = groups[i]
        g ∈ correct_groups || push!(correct_groups, g)
    end
    return correct_groups
end
```

    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mPrecompiling Knockoffs [878bf26d-0c49-448a-9df5-b057c815d613]





    get_signif_groups (generic function with 1 method)



## Test1: Simulate under factored model

This is design 1 of [IPAD: Stable Interpretable Forecasting with Knockoffs Inference](https://www.tandfonline.com/doi/abs/10.1080/01621459.2019.1654878?journalCode=uasa20). 

+ Here $n=500, p=1000$
+ ``X = F\Lambda' + \sqrt{r\theta}E`` where ``r = 3, \theta=1`` and ``F, \Lambda, E`` are ``n \times r, p \times r`, and ``n \times p`` iid gaussian matrices. 
+ ``y = X\beta + \sqrt{c}\epsilon`` where ``c = 0.2``
+ 50 causal variables with $\beta_i = A = 0.1$
+ For MVR/ME/SDP, we generate 2nd order knockoffs by estimating a shrinked covariance matrix


```julia
function compare_ipad(nsims)
    n = 500 # number of samples
    p = 1000 # number of covariates
    m = 1    # number of knockoffs per variable
    k = 50   # number of causal variables
    rtrue = 3 # true rank
    A = 0.1 # causal beta
    θ = 1
    c = 0.2 # some noise term for simulating y    
    
    sdp_powers, sdp_fdrs, sdp_times = 0.0, 0.0, 0.0
    me_powers, me_fdrs, me_times = 0.0, 0.0, 0.0
    mvr_powers, mvr_fdrs, mvr_times = 0.0, 0.0, 0.0
    ipad_er_powers, ipad_er_fdrs, ipad_er_times = 0.0, 0.0, 0.0
    ipad_gr_powers, ipad_gr_fdrs, ipad_gr_times = 0.0, 0.0, 0.0
    ipad_ve_powers, ipad_ve_fdrs, ipad_ve_times = 0.0, 0.0, 0.0

    for seed in 1:nsims
        # simulate X
        Random.seed!(seed)
        F = randn(n, rtrue)
        Λ = randn(p, rtrue)
        C = F * Λ'
        E = randn(n, p)
        X = C + sqrt(rtrue * θ) * E

        # simulate y
        Random.seed!(seed)
        βtrue = zeros(p)
        βtrue[1:k] .= rand(-1:2:1, k) .* A
        shuffle!(βtrue)
        ϵ = randn(n)
        y = X * βtrue + sqrt(c) .* ϵ
        μ = zeros(p)
        correct_snps = findall(!iszero, βtrue)

        # ipad with ER
        Random.seed!(seed)
        ipad_er_t = @elapsed ipad_er = ipad(X, r_method = :er, m = m)
        ipad_er_ko_filter = fit_lasso(y, ipad_er)
        discovered_snps = findall(!iszero, ipad_er_ko_filter.βs[3])
        ipad_er_power = round(TP(correct_snps, discovered_snps), digits=3)
        ipad_er_fdr = round(FDR(correct_snps, discovered_snps), digits=3)

        #ipad with GR
        Random.seed!(seed)
        ipad_gr_t = @elapsed ipad_gr = ipad(X, r_method = :gr, m = m)
        ipad_gr_ko_filter = fit_lasso(y, ipad_gr)
        discovered_snps = findall(!iszero, ipad_gr_ko_filter.βs[3])
        ipad_gr_power = round(TP(correct_snps, discovered_snps), digits=3)
        ipad_gr_fdr = round(FDR(correct_snps, discovered_snps), digits=3)

        # ipad with VE
        Random.seed!(seed)
        ipad_ve_t = @elapsed ipad_ve = ipad(X, r_method = :ve, m = m)
        ipad_ve_ko_filter = fit_lasso(y, ipad_ve)
        discovered_snps = findall(!iszero, ipad_ve_ko_filter.βs[3])
        ipad_ve_power = round(TP(correct_snps, discovered_snps), digits=3)
        ipad_ve_fdr = round(FDR(correct_snps, discovered_snps), digits=3)

        # ME knockoffs
        Random.seed!(seed)
        me_t = @elapsed me = modelX_gaussian_knockoffs(X, :maxent, m = m)
        me_ko_filter = fit_lasso(y, me)
        discovered_snps = findall(!iszero, me_ko_filter.βs[3])
        me_power = round(TP(correct_snps, discovered_snps), digits=3)
        me_fdr = round(FDR(correct_snps, discovered_snps), digits=3)

        # MVR knockoffs
        Random.seed!(seed)
        mvr_t = @elapsed mvr = modelX_gaussian_knockoffs(X, :mvr, m = m)
        mvr_ko_filter = fit_lasso(y, mvr)
        discovered_snps = findall(!iszero, mvr_ko_filter.βs[3])
        mvr_power = round(TP(correct_snps, discovered_snps), digits=3)
        mvr_fdr = round(FDR(correct_snps, discovered_snps), digits=3)

        # SDP (ccd) knockoffs
        Random.seed!(seed)
        sdp_t = @elapsed sdp = modelX_gaussian_knockoffs(X, :sdp_ccd, m = m)
        sdp_ko_filter = fit_lasso(y, sdp)
        discovered_snps = findall(!iszero, sdp_ko_filter.βs[3])
        sdp_power = round(TP(correct_snps, discovered_snps), digits=3)
        sdp_fdr = round(FDR(correct_snps, discovered_snps), digits=3)

        # record results
        println("ipad ER: power = $ipad_er_power, FDR = $ipad_er_fdr, rank = $(ipad_er.r), time = $(round(ipad_er_t, digits=3))")
        println("ipad GR: power = $ipad_gr_power, FDR = $ipad_gr_fdr, rank = $(ipad_gr.r), time = $(round(ipad_gr_t, digits=3))")
        println("ipad VE: power = $ipad_ve_power, FDR = $ipad_ve_fdr, rank = $(ipad_ve.r), time = $(round(ipad_ve_t, digits=3))")
        println("ME: power = $me_power, FDR = $me_fdr, time = $(round(me_t, digits=3))")
        println("MVR: power = $mvr_power, FDR = $mvr_fdr, time = $(round(mvr_t, digits=3))")
        println("SDP: power = $sdp_power, FDR = $sdp_fdr, time = $(round(sdp_t, digits=3))\n\n")
        sdp_powers += sdp_power
        sdp_fdrs += sdp_fdr
        sdp_times += sdp_t
        me_powers += me_power
        me_fdrs += me_fdr
        me_times += me_t
        mvr_powers += mvr_power
        mvr_fdrs += mvr_fdr
        mvr_times += mvr_t
        ipad_er_powers += ipad_er_power
        ipad_er_fdrs += ipad_er_fdr
        ipad_er_times += ipad_er_t
        ipad_gr_powers += ipad_gr_power
        ipad_gr_fdrs += ipad_gr_fdr
        ipad_gr_times += ipad_gr_t
        ipad_ve_powers += ipad_ve_power
        ipad_ve_fdrs += ipad_ve_fdr
        ipad_ve_times += ipad_ve_t
    end
    
    # compute average
    ipad_er_powers /= nsims
    ipad_er_fdrs /= nsims
    ipad_er_times /= nsims
    ipad_gr_powers /= nsims
    ipad_gr_fdrs /= nsims
    ipad_gr_times /= nsims
    ipad_ve_powers /= nsims
    ipad_ve_fdrs /= nsims
    ipad_ve_times /= nsims
    me_powers /= nsims
    me_fdrs /= nsims
    me_times /= nsims
    mvr_powers /= nsims
    mvr_fdrs /= nsims
    mvr_times /= nsims
    sdp_powers /= nsims
    sdp_fdrs /= nsims
    sdp_times /= nsims
    
    # save in dataframe
    result = DataFrame(
        method=["IPAD-er", "IPAD-gr", "IPAD-ve", "ME", "MVR", "SDP"],
        power=[ipad_er_powers,ipad_gr_powers,ipad_ve_powers,me_powers,mvr_powers,sdp_powers], 
        FDR=[ipad_er_fdrs,ipad_gr_fdrs,ipad_ve_fdrs,me_fdrs,mvr_fdrs,sdp_fdrs], 
        time=[ipad_er_times,ipad_gr_times,ipad_ve_times,me_times,mvr_times,sdp_times]
    )

    return result
end
```




    compare_ipad (generic function with 1 method)




```julia
nsims = 20
result = compare_ipad(nsims);
```

    ipad ER: power = 0.98, FDR = 0.075, rank = 3, time = 0.45
    ipad GR: power = 0.98, FDR = 0.075, rank = 3, time = 0.125
    ipad VE: power = 1.0, FDR = 0.167, rank = 257, time = 0.108
    ME: power = 0.0, FDR = 0.0, time = 2.748
    MVR: power = 0.0, FDR = 0.0, time = 2.742
    SDP: power = 0.48, FDR = 0.0, time = 11.437
    
    
    ipad ER: power = 1.0, FDR = 0.265, rank = 3, time = 0.082
    ipad GR: power = 1.0, FDR = 0.265, rank = 3, time = 0.041
    ipad VE: power = 1.0, FDR = 0.123, rank = 263, time = 0.045
    ME: power = 0.5, FDR = 0.0, time = 2.229
    MVR: power = 0.4, FDR = 0.0, time = 2.769
    SDP: power = 0.94, FDR = 0.078, time = 11.942
    
    
    ipad ER: power = 0.96, FDR = 0.02, rank = 3, time = 0.139
    ipad GR: power = 0.96, FDR = 0.02, rank = 3, time = 0.066
    ipad VE: power = 0.98, FDR = 0.093, rank = 257, time = 0.072
    ME: power = 0.66, FDR = 0.0, time = 2.306
    MVR: power = 0.66, FDR = 0.0, time = 2.759
    SDP: power = 0.8, FDR = 0.024, time = 11.682
    
    
    ipad ER: power = 0.94, FDR = 0.041, rank = 3, time = 0.13
    ipad GR: power = 0.94, FDR = 0.041, rank = 3, time = 0.122
    ipad VE: power = 0.92, FDR = 0.042, rank = 260, time = 0.048
    ME: power = 0.0, FDR = 0.0, time = 2.222
    MVR: power = 0.0, FDR = 0.0, time = 2.79
    SDP: power = 0.0, FDR = 0.0, time = 11.571
    
    
    ipad ER: power = 1.0, FDR = 0.206, rank = 3, time = 0.108
    ipad GR: power = 1.0, FDR = 0.206, rank = 3, time = 0.042
    ipad VE: power = 0.98, FDR = 0.075, rank = 256, time = 0.049
    ME: power = 0.78, FDR = 0.025, time = 2.247
    MVR: power = 0.64, FDR = 0.0, time = 2.769
    SDP: power = 0.9, FDR = 0.1, time = 11.45
    
    
    ipad ER: power = 1.0, FDR = 0.057, rank = 3, time = 0.108
    ipad GR: power = 1.0, FDR = 0.057, rank = 3, time = 0.129
    ipad VE: power = 0.9, FDR = 0.135, rank = 253, time = 0.132
    ME: power = 0.84, FDR = 0.023, time = 2.427
    MVR: power = 0.78, FDR = 0.025, time = 2.989
    SDP: power = 0.76, FDR = 0.05, time = 11.448
    
    
    ipad ER: power = 0.92, FDR = 0.098, rank = 3, time = 0.111
    ipad GR: power = 0.92, FDR = 0.098, rank = 3, time = 0.044
    ipad VE: power = 0.92, FDR = 0.042, rank = 256, time = 0.045
    ME: power = 0.54, FDR = 0.0, time = 2.357
    MVR: power = 0.62, FDR = 0.0, time = 2.816
    SDP: power = 0.8, FDR = 0.048, time = 11.439
    
    
    ipad ER: power = 1.0, FDR = 0.107, rank = 3, time = 0.111
    ipad GR: power = 1.0, FDR = 0.107, rank = 3, time = 0.135
    ipad VE: power = 1.0, FDR = 0.153, rank = 260, time = 0.049
    ME: power = 0.66, FDR = 0.0, time = 2.447
    MVR: power = 0.66, FDR = 0.0, time = 2.759
    SDP: power = 0.8, FDR = 0.0, time = 11.596
    
    
    ipad ER: power = 1.0, FDR = 0.091, rank = 3, time = 0.117
    ipad GR: power = 1.0, FDR = 0.091, rank = 3, time = 0.087
    ipad VE: power = 1.0, FDR = 0.074, rank = 257, time = 0.044
    ME: power = 0.0, FDR = 0.0, time = 2.313
    MVR: power = 0.0, FDR = 0.0, time = 2.756
    SDP: power = 0.86, FDR = 0.0, time = 11.64
    
    
    ipad ER: power = 0.94, FDR = 0.145, rank = 3, time = 0.099
    ipad GR: power = 0.94, FDR = 0.145, rank = 3, time = 0.041
    ipad VE: power = 0.94, FDR = 0.19, rank = 255, time = 0.049
    ME: power = 0.68, FDR = 0.0, time = 2.393
    MVR: power = 0.7, FDR = 0.0, time = 2.809
    SDP: power = 0.9, FDR = 0.022, time = 11.447
    
    
    ipad ER: power = 0.92, FDR = 0.0, rank = 3, time = 0.1
    ipad GR: power = 0.92, FDR = 0.0, rank = 3, time = 0.179
    ipad VE: power = 0.96, FDR = 0.077, rank = 257, time = 0.053
    ME: power = 0.0, FDR = 0.0, time = 2.207
    MVR: power = 0.0, FDR = 0.0, time = 2.802
    SDP: power = 0.48, FDR = 0.0, time = 11.579
    
    
    ipad ER: power = 1.0, FDR = 0.167, rank = 3, time = 0.099
    ipad GR: power = 1.0, FDR = 0.167, rank = 3, time = 0.083
    ipad VE: power = 1.0, FDR = 0.138, rank = 263, time = 0.045
    ME: power = 0.0, FDR = 0.0, time = 2.4
    MVR: power = 0.0, FDR = 0.0, time = 2.933
    SDP: power = 0.46, FDR = 0.0, time = 11.682
    
    
    ipad ER: power = 1.0, FDR = 0.038, rank = 3, time = 0.143
    ipad GR: power = 1.0, FDR = 0.038, rank = 3, time = 0.042
    ipad VE: power = 1.0, FDR = 0.038, rank = 258, time = 0.109
    ME: power = 0.6, FDR = 0.0, time = 2.24
    MVR: power = 0.4, FDR = 0.0, time = 2.771
    SDP: power = 0.0, FDR = 0.0, time = 11.649
    
    
    ipad ER: power = 0.98, FDR = 0.109, rank = 3, time = 0.104
    ipad GR: power = 0.98, FDR = 0.109, rank = 3, time = 0.125
    ipad VE: power = 1.0, FDR = 0.123, rank = 259, time = 0.177
    ME: power = 0.0, FDR = 0.0, time = 2.208
    MVR: power = 0.0, FDR = 0.0, time = 2.757
    SDP: power = 0.9, FDR = 0.0, time = 11.386
    
    
    ipad ER: power = 0.98, FDR = 0.058, rank = 3, time = 0.124
    ipad GR: power = 0.98, FDR = 0.058, rank = 3, time = 0.042
    ipad VE: power = 1.0, FDR = 0.091, rank = 258, time = 0.045
    ME: power = 0.6, FDR = 0.0, time = 2.203
    MVR: power = 0.0, FDR = 0.0, time = 3.052
    SDP: power = 0.8, FDR = 0.024, time = 11.453
    
    
    ipad ER: power = 0.98, FDR = 0.197, rank = 3, time = 0.109
    ipad GR: power = 0.98, FDR = 0.197, rank = 3, time = 0.042
    ipad VE: power = 1.0, FDR = 0.107, rank = 257, time = 0.044
    ME: power = 0.66, FDR = 0.0, time = 2.222
    MVR: power = 0.64, FDR = 0.0, time = 2.768
    SDP: power = 0.84, FDR = 0.023, time = 11.549
    
    
    ipad ER: power = 1.0, FDR = 0.153, rank = 3, time = 0.131
    ipad GR: power = 1.0, FDR = 0.153, rank = 3, time = 0.071
    ipad VE: power = 1.0, FDR = 0.091, rank = 263, time = 0.045
    ME: power = 0.0, FDR = 0.0, time = 2.297
    MVR: power = 0.0, FDR = 0.0, time = 3.06
    SDP: power = 0.0, FDR = 0.0, time = 11.443
    
    
    ipad ER: power = 1.0, FDR = 0.038, rank = 3, time = 0.186
    ipad GR: power = 1.0, FDR = 0.038, rank = 3, time = 0.042
    ipad VE: power = 0.98, FDR = 0.197, rank = 263, time = 0.104
    ME: power = 0.84, FDR = 0.0, time = 2.283
    MVR: power = 0.86, FDR = 0.0, time = 2.868
    SDP: power = 0.88, FDR = 0.0, time = 12.216
    
    
    ipad ER: power = 1.0, FDR = 0.02, rank = 3, time = 0.115
    ipad GR: power = 1.0, FDR = 0.02, rank = 3, time = 0.119
    ipad VE: power = 1.0, FDR = 0.02, rank = 255, time = 0.105
    ME: power = 0.84, FDR = 0.045, time = 2.234
    MVR: power = 0.86, FDR = 0.065, time = 3.067
    SDP: power = 0.84, FDR = 0.045, time = 11.421
    
    
    ipad ER: power = 0.96, FDR = 0.158, rank = 3, time = 0.095
    ipad GR: power = 0.96, FDR = 0.158, rank = 3, time = 0.047
    ipad VE: power = 0.86, FDR = 0.0, rank = 258, time = 0.046
    ME: power = 0.0, FDR = 0.0, time = 2.274
    MVR: power = 0.0, FDR = 0.0, time = 2.748
    SDP: power = 0.0, FDR = 0.0, time = 11.502
    
    



```julia
# check average
@show result;
```

    result = 6×4 DataFrame
     Row │ method   power    FDR      time
         │ String   Float64  Float64  Float64
    ─────┼───────────────────────────────────────
       1 │ IPAD-er    0.978  0.10215   0.133172
       2 │ IPAD-gr    0.978  0.10215   0.0811693
       3 │ IPAD-ve    0.972  0.0988    0.0707433
       4 │ ME         0.41   0.00465   2.3128
       5 │ MVR        0.361  0.0045    2.83926
       6 │ SDP        0.622  0.0207   11.5767


Summary
+ All methods control FDR (target = 10%) 
+ ER and GR method always find the correct rank (r = 3), while VE overestimates the rank
+ IPAD method has much better power compared to model-X knockoffs via ME/MVR/SDP construction
+ Surprisingly, ME/MVR has worse power than SDP
+ IPAD method is much more efficient to construct

## Test2: Try $X$ that's not a factored model

+ Here ``y \sim N(X\beta, 1)`` and ``X_i \sim N(0, \Sigma)`` where ``\Sigma`` is an AR(1) model.
+ 50 causal SNPs with ``\beta_i \sim \pm N(0, 0.5)``
+ For MVR/ME/SDP knockoffs, we assume the true ``\mu`` and ``\Sigma`` are available. 


```julia
function compare_ipad2(nsims)
    n = 500 # number of samples
    p = 1000 # number of covariates
    m = 1    # number of knockoffs per variable
    k = 50   # number of causal variables

    sdp_powers, sdp_fdrs, sdp_times = 0.0, 0.0, 0.0
    me_powers, me_fdrs, me_times = 0.0, 0.0, 0.0
    mvr_powers, mvr_fdrs, mvr_times = 0.0, 0.0, 0.0
    ipad_er_powers, ipad_er_fdrs, ipad_er_times = 0.0, 0.0, 0.0
    ipad_gr_powers, ipad_gr_fdrs, ipad_gr_times = 0.0, 0.0, 0.0
    ipad_ve_powers, ipad_ve_fdrs, ipad_ve_times = 0.0, 0.0, 0.0

    for seed in 1:nsims
        # simulate X
        Random.seed!(seed)
        Σ = simulate_AR1(p, a=3, b=1)
        μ = zeros(p)
        X = rand(MvNormal(μ, Σ), n)' |> Matrix
        zscore!(X, mean(X, dims=1), std(X, dims=1))

        # simulate y
        Random.seed!(seed)
        βtrue = zeros(p)
        βtrue[1:k] .= rand(-1:2:1, k) .* rand(Normal(0, 0.5), k)
        shuffle!(βtrue)
        ϵ = randn(n)
        y = X * βtrue + ϵ
        μ = zeros(p)
        correct_snps = findall(!iszero, βtrue)

        # ipad with ER
        Random.seed!(seed)
        ipad_er_t = @elapsed ipad_er = ipad(X, r_method = :er, m = m)
        ipad_er_ko_filter = fit_lasso(y, ipad_er)
        discovered_snps = findall(!iszero, ipad_er_ko_filter.βs[3])
        ipad_er_power = round(TP(correct_snps, discovered_snps), digits=3)
        ipad_er_fdr = round(FDR(correct_snps, discovered_snps), digits=3)

        #ipad with GR
        Random.seed!(seed)
        ipad_gr_t = @elapsed ipad_gr = ipad(X, r_method = :gr, m = m)
        ipad_gr_ko_filter = fit_lasso(y, ipad_gr)
        discovered_snps = findall(!iszero, ipad_gr_ko_filter.βs[3])
        ipad_gr_power = round(TP(correct_snps, discovered_snps), digits=3)
        ipad_gr_fdr = round(FDR(correct_snps, discovered_snps), digits=3)

        # ipad with VE
        Random.seed!(seed)
        ipad_ve_t = @elapsed ipad_ve = ipad(X, r_method = :ve, m = m)
        ipad_ve_ko_filter = fit_lasso(y, ipad_ve)
        discovered_snps = findall(!iszero, ipad_ve_ko_filter.βs[3])
        ipad_ve_power = round(TP(correct_snps, discovered_snps), digits=3)
        ipad_ve_fdr = round(FDR(correct_snps, discovered_snps), digits=3)

        # ME knockoffs
        Random.seed!(seed)
        me_t = @elapsed me = modelX_gaussian_knockoffs(X, :maxent, μ, Σ, m = m)
        me_ko_filter = fit_lasso(y, me)
        discovered_snps = findall(!iszero, me_ko_filter.βs[3])
        me_power = round(TP(correct_snps, discovered_snps), digits=3)
        me_fdr = round(FDR(correct_snps, discovered_snps), digits=3)

        # MVR knockoffs
        Random.seed!(seed)
        mvr_t = @elapsed mvr = modelX_gaussian_knockoffs(X, :mvr, μ, Σ,m = m)
        mvr_ko_filter = fit_lasso(y, mvr)
        discovered_snps = findall(!iszero, mvr_ko_filter.βs[3])
        mvr_power = round(TP(correct_snps, discovered_snps), digits=3)
        mvr_fdr = round(FDR(correct_snps, discovered_snps), digits=3)

        # SDP (ccd) knockoffs
        Random.seed!(seed)
        sdp_t = @elapsed sdp = modelX_gaussian_knockoffs(X, :sdp_ccd, μ, Σ, m = m)
        sdp_ko_filter = fit_lasso(y, sdp)
        discovered_snps = findall(!iszero, sdp_ko_filter.βs[3])
        sdp_power = round(TP(correct_snps, discovered_snps), digits=3)
        sdp_fdr = round(FDR(correct_snps, discovered_snps), digits=3)

        # record results
        println("ipad ER: power = $ipad_er_power, FDR = $ipad_er_fdr, rank = $(ipad_er.r), time = $(round(ipad_er_t, digits=3))")
        println("ipad GR: power = $ipad_gr_power, FDR = $ipad_gr_fdr, rank = $(ipad_gr.r), time = $(round(ipad_gr_t, digits=3))")
        println("ipad VE: power = $ipad_ve_power, FDR = $ipad_ve_fdr, rank = $(ipad_ve.r), time = $(round(ipad_ve_t, digits=3))")
        println("ME: power = $me_power, FDR = $me_fdr, time = $(round(me_t, digits=3))")
        println("MVR: power = $mvr_power, FDR = $mvr_fdr, time = $(round(mvr_t, digits=3))")
        println("SDP: power = $sdp_power, FDR = $sdp_fdr, time = $(round(sdp_t, digits=3))\n\n")
        sdp_powers += sdp_power
        sdp_fdrs += sdp_fdr
        sdp_times += sdp_t
        me_powers += me_power
        me_fdrs += me_fdr
        me_times += me_t
        mvr_powers += mvr_power
        mvr_fdrs += mvr_fdr
        mvr_times += mvr_t
        ipad_er_powers += ipad_er_power
        ipad_er_fdrs += ipad_er_fdr
        ipad_er_times += ipad_er_t
        ipad_gr_powers += ipad_gr_power
        ipad_gr_fdrs += ipad_gr_fdr
        ipad_gr_times += ipad_gr_t
        ipad_ve_powers += ipad_ve_power
        ipad_ve_fdrs += ipad_ve_fdr
        ipad_ve_times += ipad_ve_t
    end
    
    # compute average
    ipad_er_powers /= nsims
    ipad_er_fdrs /= nsims
    ipad_er_times /= nsims
    ipad_gr_powers /= nsims
    ipad_gr_fdrs /= nsims
    ipad_gr_times /= nsims
    ipad_ve_powers /= nsims
    ipad_ve_fdrs /= nsims
    ipad_ve_times /= nsims
    me_powers /= nsims
    me_fdrs /= nsims
    me_times /= nsims
    mvr_powers /= nsims
    mvr_fdrs /= nsims
    mvr_times /= nsims
    sdp_powers /= nsims
    sdp_fdrs /= nsims
    sdp_times /= nsims
    
    # save in dataframe
    result = DataFrame(
        method=["IPAD-er", "IPAD-gr", "IPAD-ve", "ME", "MVR", "SDP"],
        power=[ipad_er_powers,ipad_gr_powers,ipad_ve_powers,me_powers,mvr_powers,sdp_powers], 
        FDR=[ipad_er_fdrs,ipad_gr_fdrs,ipad_ve_fdrs,me_fdrs,mvr_fdrs,sdp_fdrs], 
        time=[ipad_er_times,ipad_gr_times,ipad_ve_times,me_times,mvr_times,sdp_times]
    )

    return result
end
```




    compare_ipad2 (generic function with 1 method)




```julia
nsims = 20
result = compare_ipad2(nsims);
```

    ipad ER: power = 0.0, FDR = 0.0, rank = 496, time = 0.139
    ipad GR: power = 0.64, FDR = 0.179, rank = 1, time = 0.116
    ipad VE: power = 0.62, FDR = 0.205, rank = 217, time = 0.079
    ME: power = 0.28, FDR = 0.0, time = 1.998
    MVR: power = 0.26, FDR = 0.0, time = 2.878
    SDP: power = 0.4, FDR = 0.0, time = 7.752
    
    
    ipad ER: power = 0.5, FDR = 0.242, rank = 1, time = 0.042
    ipad GR: power = 0.5, FDR = 0.242, rank = 1, time = 0.063
    ipad VE: power = 0.5, FDR = 0.242, rank = 219, time = 0.044
    ME: power = 0.0, FDR = 0.0, time = 1.889
    MVR: power = 0.2, FDR = 0.0, time = 2.765
    SDP: power = 0.38, FDR = 0.05, time = 7.801
    
    
    ipad ER: power = 0.58, FDR = 0.216, rank = 1, time = 0.046
    ipad GR: power = 0.58, FDR = 0.216, rank = 1, time = 0.116
    ipad VE: power = 0.62, FDR = 0.244, rank = 217, time = 0.15
    ME: power = 0.5, FDR = 0.107, time = 2.102
    MVR: power = 0.48, FDR = 0.077, time = 2.801
    SDP: power = 0.0, FDR = 0.0, time = 7.656
    
    
    ipad ER: power = 0.46, FDR = 0.148, rank = 480, time = 0.048
    ipad GR: power = 0.62, FDR = 0.295, rank = 10, time = 0.114
    ipad VE: power = 0.66, FDR = 0.283, rank = 213, time = 0.044
    ME: power = 0.54, FDR = 0.129, time = 2.168
    MVR: power = 0.46, FDR = 0.08, time = 2.804
    SDP: power = 0.4, FDR = 0.091, time = 7.814
    
    
    ipad ER: power = 0.56, FDR = 0.2, rank = 1, time = 0.065
    ipad GR: power = 0.56, FDR = 0.2, rank = 1, time = 0.07
    ipad VE: power = 0.56, FDR = 0.222, rank = 217, time = 0.044
    ME: power = 0.38, FDR = 0.136, time = 2.121
    MVR: power = 0.5, FDR = 0.167, time = 2.783
    SDP: power = 0.4, FDR = 0.091, time = 7.58
    
    
    ipad ER: power = 0.0, FDR = 0.0, rank = 497, time = 0.049
    ipad GR: power = 0.72, FDR = 0.25, rank = 3, time = 0.041
    ipad VE: power = 0.74, FDR = 0.351, rank = 218, time = 0.06
    ME: power = 0.64, FDR = 0.059, time = 2.054
    MVR: power = 0.64, FDR = 0.059, time = 2.736
    SDP: power = 0.62, FDR = 0.061, time = 7.83
    
    
    ipad ER: power = 0.76, FDR = 0.345, rank = 3, time = 0.041
    ipad GR: power = 0.76, FDR = 0.345, rank = 3, time = 0.042
    ipad VE: power = 0.76, FDR = 0.367, rank = 214, time = 0.045
    ME: power = 0.6, FDR = 0.091, time = 2.165
    MVR: power = 0.6, FDR = 0.118, time = 2.798
    SDP: power = 0.22, FDR = 0.0, time = 7.919
    
    
    ipad ER: power = 0.0, FDR = 0.0, rank = 495, time = 0.055
    ipad GR: power = 0.58, FDR = 0.237, rank = 2, time = 0.041
    ipad VE: power = 0.56, FDR = 0.2, rank = 213, time = 0.068
    ME: power = 0.0, FDR = 0.0, time = 1.998
    MVR: power = 0.0, FDR = 0.0, time = 2.782
    SDP: power = 0.4, FDR = 0.13, time = 7.664
    
    
    ipad ER: power = 0.68, FDR = 0.292, rank = 1, time = 0.092
    ipad GR: power = 0.68, FDR = 0.292, rank = 1, time = 0.041
    ipad VE: power = 0.7, FDR = 0.314, rank = 218, time = 0.071
    ME: power = 0.0, FDR = 0.0, time = 2.052
    MVR: power = 0.0, FDR = 0.0, time = 2.647
    SDP: power = 0.0, FDR = 0.0, time = 8.211
    
    
    ipad ER: power = 0.6, FDR = 0.375, rank = 1, time = 0.041
    ipad GR: power = 0.6, FDR = 0.375, rank = 1, time = 0.041
    ipad VE: power = 0.54, FDR = 0.386, rank = 213, time = 0.043
    ME: power = 0.44, FDR = 0.154, time = 2.175
    MVR: power = 0.36, FDR = 0.1, time = 2.596
    SDP: power = 0.0, FDR = 0.0, time = 8.262
    
    
    ipad ER: power = 0.0, FDR = 0.0, rank = 495, time = 0.051
    ipad GR: power = 0.46, FDR = 0.303, rank = 35, time = 0.155
    ipad VE: power = 0.46, FDR = 0.233, rank = 212, time = 0.174
    ME: power = 0.22, FDR = 0.0, time = 1.921
    MVR: power = 0.0, FDR = 0.0, time = 2.625
    SDP: power = 0.0, FDR = 0.0, time = 8.102
    
    
    ipad ER: power = 0.0, FDR = 0.0, rank = 496, time = 0.048
    ipad GR: power = 0.72, FDR = 0.321, rank = 2, time = 0.088
    ipad VE: power = 0.72, FDR = 0.333, rank = 213, time = 0.044
    ME: power = 0.48, FDR = 0.111, time = 1.949
    MVR: power = 0.46, FDR = 0.115, time = 2.722
    SDP: power = 0.54, FDR = 0.129, time = 8.258
    
    
    ipad ER: power = 0.54, FDR = 0.129, rank = 483, time = 0.048
    ipad GR: power = 0.72, FDR = 0.28, rank = 108, time = 0.042
    ipad VE: power = 0.74, FDR = 0.26, rank = 215, time = 0.044
    ME: power = 0.42, FDR = 0.045, time = 1.942
    MVR: power = 0.44, FDR = 0.043, time = 2.723
    SDP: power = 0.34, FDR = 0.056, time = 8.185
    
    
    ipad ER: power = 0.56, FDR = 0.3, rank = 2, time = 0.04
    ipad GR: power = 0.56, FDR = 0.3, rank = 2, time = 0.082
    ipad VE: power = 0.58, FDR = 0.341, rank = 212, time = 0.043
    ME: power = 0.36, FDR = 0.0, time = 1.998
    MVR: power = 0.42, FDR = 0.0, time = 2.72
    SDP: power = 0.46, FDR = 0.08, time = 7.732
    
    
    ipad ER: power = 0.24, FDR = 0.0, rank = 494, time = 0.08
    ipad GR: power = 0.62, FDR = 0.225, rank = 2, time = 0.041
    ipad VE: power = 0.54, FDR = 0.156, rank = 215, time = 0.044
    ME: power = 0.46, FDR = 0.08, time = 1.99
    MVR: power = 0.48, FDR = 0.04, time = 2.727
    SDP: power = 0.3, FDR = 0.0, time = 7.715
    
    
    ipad ER: power = 0.26, FDR = 0.0, rank = 495, time = 0.048
    ipad GR: power = 0.58, FDR = 0.256, rank = 7, time = 0.09
    ipad VE: power = 0.58, FDR = 0.31, rank = 218, time = 0.102
    ME: power = 0.52, FDR = 0.161, time = 2.058
    MVR: power = 0.5, FDR = 0.167, time = 2.862
    SDP: power = 0.44, FDR = 0.043, time = 7.542
    
    
    ipad ER: power = 0.4, FDR = 0.048, rank = 485, time = 0.048
    ipad GR: power = 0.74, FDR = 0.315, rank = 1, time = 0.041
    ipad VE: power = 0.76, FDR = 0.321, rank = 212, time = 0.043
    ME: power = 0.44, FDR = 0.083, time = 1.986
    MVR: power = 0.44, FDR = 0.083, time = 2.63
    SDP: power = 0.66, FDR = 0.214, time = 7.773
    
    
    ipad ER: power = 0.64, FDR = 0.179, rank = 1, time = 0.041
    ipad GR: power = 0.64, FDR = 0.179, rank = 1, time = 0.04
    ipad VE: power = 0.66, FDR = 0.233, rank = 216, time = 0.068
    ME: power = 0.5, FDR = 0.038, time = 1.997
    MVR: power = 0.44, FDR = 0.043, time = 2.716
    SDP: power = 0.44, FDR = 0.043, time = 7.753
    
    
    ipad ER: power = 0.4, FDR = 0.0, rank = 496, time = 0.048
    ipad GR: power = 0.6, FDR = 0.268, rank = 1, time = 0.041
    ipad VE: power = 0.62, FDR = 0.311, rank = 215, time = 0.044
    ME: power = 0.54, FDR = 0.182, time = 1.977
    MVR: power = 0.58, FDR = 0.171, time = 2.712
    SDP: power = 0.56, FDR = 0.067, time = 8.281
    
    
    ipad ER: power = 0.46, FDR = 0.233, rank = 3, time = 0.041
    ipad GR: power = 0.46, FDR = 0.233, rank = 3, time = 0.109
    ipad VE: power = 0.48, FDR = 0.314, rank = 213, time = 0.043
    ME: power = 0.32, FDR = 0.059, time = 1.908
    MVR: power = 0.4, FDR = 0.048, time = 2.67
    SDP: power = 0.44, FDR = 0.043, time = 7.994
    
    



```julia
@show result;
```

    result = 6×4 DataFrame
     Row │ method   power    FDR      time
         │ String   Float64  Float64  Float64
    ─────┼──────────────────────────────────────
       1 │ IPAD-er    0.382  0.13535  0.0554262
       2 │ IPAD-gr    0.617  0.26555  0.0707459
       3 │ IPAD-ve    0.62   0.2813   0.0648921
       4 │ ME         0.382  0.07175  2.02233
       5 │ MVR        0.383  0.06555  2.73484
       6 │ SDP        0.35   0.0549   7.8911


Summary

+ IPAD methods have slightly~pretty inflated FDR (target = 10%) 
+ When $X$ does not follow the factored model, ER/GR/VE finds wildly differing ranks


## Test3: Simulate with gnomAD panel

+ Here we simulate ``X_i \sim N(0, \Sigma)`` where ``\Sigma`` is from the European [gnomAD LD panel](https://gnomad.broadinstitute.org/downloads#v2-linkage-disequilibrium). 
+ ``\Sigma`` can be downloaded and extract with the software [EasyLD.jl](https://github.com/biona001/EasyLD.jl). 


```julia
function compare_ipad3(nsims)
    # import panel
    datadir = "/Users/biona001/Benjamin_Folder/research/4th_project_PRS/group_knockoff_test_data"
    mapfile = CSV.read(joinpath(datadir, "Groups_2_127374341_128034347.txt"), DataFrame)
    groups = mapfile[!, :group]
    covfile = CSV.read(joinpath(datadir, "CorG_2_127374341_128034347.txt"), DataFrame)
    Σ = covfile |> Matrix{Float64}
    Σ = 0.99Σ + 0.01I #ensure PSD

    # test on smaller data
    idx = findlast(x -> x == 263, groups) # 263 is largest group with 192 members
    groups = groups[1:idx]
    Σ = Σ[1:idx, 1:idx]
    
    # simulation parameters
    n = 500 # number of samples
    p = size(Σ, 1) # number of covariates
    m = 1    # number of knockoffs per variable
    k = 50   # number of causal variables
    
    sdp_powers, sdp_fdrs, sdp_times = 0.0, 0.0, 0.0
    me_powers, me_fdrs, me_times = 0.0, 0.0, 0.0
    mvr_powers, mvr_fdrs, mvr_times = 0.0, 0.0, 0.0
    ipad_er_powers, ipad_er_fdrs, ipad_er_times = 0.0, 0.0, 0.0
    ipad_gr_powers, ipad_gr_fdrs, ipad_gr_times = 0.0, 0.0, 0.0
    ipad_ve_powers, ipad_ve_fdrs, ipad_ve_times = 0.0, 0.0, 0.0

    for seed in 1:nsims
        # simulate X
        Random.seed!(seed)
        X = rand(MvNormal(Σ), n)' |> Matrix
        zscore!(X, mean(X, dims=1), std(X, dims=1));

        # simulate y
        Random.seed!(seed)
        βtrue = zeros(p)
        βtrue[1:k] .= rand(-1:2:1, k) .* randn(k)
        shuffle!(βtrue)
        ϵ = randn(n)
        y = X * βtrue + ϵ
        μ = zeros(p)
        correct_snps = findall(!iszero, βtrue)

        # ipad with ER
        Random.seed!(seed)
        ipad_er_t = @elapsed ipad_er = ipad(X, r_method = :er, m = m)
        ipad_er_ko_filter = fit_lasso(y, ipad_er)
        discovered_snps = findall(!iszero, ipad_er_ko_filter.βs[3])
        ipad_er_power = round(TP(correct_snps, discovered_snps), digits=3)
        ipad_er_fdr = round(FDR(correct_snps, discovered_snps), digits=3)

        #ipad with GR
        Random.seed!(seed)
        ipad_gr_t = @elapsed ipad_gr = ipad(X, r_method = :gr, m = m)
        ipad_gr_ko_filter = fit_lasso(y, ipad_gr)
        discovered_snps = findall(!iszero, ipad_gr_ko_filter.βs[3])
        ipad_gr_power = round(TP(correct_snps, discovered_snps), digits=3)
        ipad_gr_fdr = round(FDR(correct_snps, discovered_snps), digits=3)

        # ipad with VE
        Random.seed!(seed)
        ipad_ve_t = @elapsed ipad_ve = ipad(X, r_method = :ve, m = m)
        ipad_ve_ko_filter = fit_lasso(y, ipad_ve)
        discovered_snps = findall(!iszero, ipad_ve_ko_filter.βs[3])
        ipad_ve_power = round(TP(correct_snps, discovered_snps), digits=3)
        ipad_ve_fdr = round(FDR(correct_snps, discovered_snps), digits=3)

        # ME knockoffs
        Random.seed!(seed)
        me_t = @elapsed me = modelX_gaussian_knockoffs(X, :maxent, m = m)
        me_ko_filter = fit_lasso(y, me)
        discovered_snps = findall(!iszero, me_ko_filter.βs[3])
        me_power = round(TP(correct_snps, discovered_snps), digits=3)
        me_fdr = round(FDR(correct_snps, discovered_snps), digits=3)

        # MVR knockoffs
        Random.seed!(seed)
        mvr_t = @elapsed mvr = modelX_gaussian_knockoffs(X, :mvr, m = m)
        mvr_ko_filter = fit_lasso(y, mvr)
        discovered_snps = findall(!iszero, mvr_ko_filter.βs[3])
        mvr_power = round(TP(correct_snps, discovered_snps), digits=3)
        mvr_fdr = round(FDR(correct_snps, discovered_snps), digits=3)

        # SDP (ccd) knockoffs
        Random.seed!(seed)
        sdp_t = @elapsed sdp = modelX_gaussian_knockoffs(X, :sdp_ccd, m = m)
        sdp_ko_filter = fit_lasso(y, sdp)
        discovered_snps = findall(!iszero, sdp_ko_filter.βs[3])
        sdp_power = round(TP(correct_snps, discovered_snps), digits=3)
        sdp_fdr = round(FDR(correct_snps, discovered_snps), digits=3)

        # record results
        println("ipad ER: power = $ipad_er_power, FDR = $ipad_er_fdr, rank = $(ipad_er.r), time = $(round(ipad_er_t, digits=3))")
        println("ipad GR: power = $ipad_gr_power, FDR = $ipad_gr_fdr, rank = $(ipad_gr.r), time = $(round(ipad_gr_t, digits=3))")
        println("ipad VE: power = $ipad_ve_power, FDR = $ipad_ve_fdr, rank = $(ipad_ve.r), time = $(round(ipad_ve_t, digits=3))")
        println("ME: power = $me_power, FDR = $me_fdr, time = $(round(me_t, digits=3))")
        println("MVR: power = $mvr_power, FDR = $mvr_fdr, time = $(round(mvr_t, digits=3))")
        println("SDP: power = $sdp_power, FDR = $sdp_fdr, time = $(round(sdp_t, digits=3))\n\n")
        sdp_powers += sdp_power
        sdp_fdrs += sdp_fdr
        sdp_times += sdp_t
        me_powers += me_power
        me_fdrs += me_fdr
        me_times += me_t
        mvr_powers += mvr_power
        mvr_fdrs += mvr_fdr
        mvr_times += mvr_t
        ipad_er_powers += ipad_er_power
        ipad_er_fdrs += ipad_er_fdr
        ipad_er_times += ipad_er_t
        ipad_gr_powers += ipad_gr_power
        ipad_gr_fdrs += ipad_gr_fdr
        ipad_gr_times += ipad_gr_t
        ipad_ve_powers += ipad_ve_power
        ipad_ve_fdrs += ipad_ve_fdr
        ipad_ve_times += ipad_ve_t
    end
    
    # compute average
    ipad_er_powers /= nsims
    ipad_er_fdrs /= nsims
    ipad_er_times /= nsims
    ipad_gr_powers /= nsims
    ipad_gr_fdrs /= nsims
    ipad_gr_times /= nsims
    ipad_ve_powers /= nsims
    ipad_ve_fdrs /= nsims
    ipad_ve_times /= nsims
    me_powers /= nsims
    me_fdrs /= nsims
    me_times /= nsims
    mvr_powers /= nsims
    mvr_fdrs /= nsims
    mvr_times /= nsims
    sdp_powers /= nsims
    sdp_fdrs /= nsims
    sdp_times /= nsims
    
    # save in dataframe
    result = DataFrame(
        method=["IPAD-er", "IPAD-gr", "IPAD-ve", "ME", "MVR", "SDP"],
        power=[ipad_er_powers,ipad_gr_powers,ipad_ve_powers,me_powers,mvr_powers,sdp_powers], 
        FDR=[ipad_er_fdrs,ipad_gr_fdrs,ipad_ve_fdrs,me_fdrs,mvr_fdrs,sdp_fdrs], 
        time=[ipad_er_times,ipad_gr_times,ipad_ve_times,me_times,mvr_times,sdp_times]
    )

    return result
end
```




    compare_ipad3 (generic function with 1 method)




```julia
nsims = 20
result = compare_ipad3(nsims);
```

    ipad ER: power = 0.4, FDR = 0.31, rank = 1, time = 0.083
    ipad GR: power = 0.4, FDR = 0.31, rank = 1, time = 0.085
    ipad VE: power = 0.38, FDR = 0.321, rank = 90, time = 0.115
    ME: power = 0.34, FDR = 0.056, time = 13.224
    MVR: power = 0.36, FDR = 0.053, time = 12.361
    SDP: power = 0.32, FDR = 0.0, time = 58.465
    
    
    ipad ER: power = 0.18, FDR = 0.25, rank = 1, time = 0.089
    ipad GR: power = 0.18, FDR = 0.25, rank = 1, time = 0.089
    ipad VE: power = 0.18, FDR = 0.308, rank = 91, time = 0.116
    ME: power = 0.46, FDR = 0.233, time = 13.859
    MVR: power = 0.44, FDR = 0.241, time = 13.163
    SDP: power = 0.44, FDR = 0.043, time = 62.731
    
    
    ipad ER: power = 0.32, FDR = 0.385, rank = 1, time = 0.095
    ipad GR: power = 0.32, FDR = 0.385, rank = 1, time = 0.094
    ipad VE: power = 0.36, FDR = 0.357, rank = 91, time = 0.099
    ME: power = 0.38, FDR = 0.05, time = 13.979
    MVR: power = 0.34, FDR = 0.0, time = 15.163
    SDP: power = 0.36, FDR = 0.0, time = 62.336
    
    
    ipad ER: power = 0.42, FDR = 0.4, rank = 1, time = 0.098
    ipad GR: power = 0.42, FDR = 0.4, rank = 1, time = 0.093
    ipad VE: power = 0.42, FDR = 0.364, rank = 91, time = 0.117
    ME: power = 0.28, FDR = 0.067, time = 13.812
    MVR: power = 0.28, FDR = 0.067, time = 13.708
    SDP: power = 0.24, FDR = 0.077, time = 62.871
    
    
    ipad ER: power = 0.4, FDR = 0.167, rank = 1, time = 0.091
    ipad GR: power = 0.4, FDR = 0.167, rank = 1, time = 0.093
    ipad VE: power = 0.4, FDR = 0.259, rank = 91, time = 0.095
    ME: power = 0.4, FDR = 0.13, time = 13.751
    MVR: power = 0.4, FDR = 0.13, time = 14.272
    SDP: power = 0.44, FDR = 0.0, time = 60.371
    
    
    ipad ER: power = 0.36, FDR = 0.438, rank = 1, time = 0.105
    ipad GR: power = 0.36, FDR = 0.438, rank = 1, time = 0.095
    ipad VE: power = 0.38, FDR = 0.424, rank = 91, time = 0.101
    ME: power = 0.46, FDR = 0.115, time = 13.838
    MVR: power = 0.46, FDR = 0.115, time = 12.917
    SDP: power = 0.42, FDR = 0.045, time = 60.665
    
    
    ipad ER: power = 0.0, FDR = 0.0, rank = 1, time = 0.092
    ipad GR: power = 0.0, FDR = 0.0, rank = 1, time = 0.093
    ipad VE: power = 0.0, FDR = 0.0, rank = 90, time = 0.098
    ME: power = 0.32, FDR = 0.0, time = 15.112
    MVR: power = 0.32, FDR = 0.0, time = 12.944
    SDP: power = 0.3, FDR = 0.0, time = 60.531
    
    
    ipad ER: power = 0.44, FDR = 0.569, rank = 1, time = 0.085
    ipad GR: power = 0.44, FDR = 0.569, rank = 1, time = 0.088
    ipad VE: power = 0.44, FDR = 0.569, rank = 91, time = 0.094
    ME: power = 0.38, FDR = 0.095, time = 13.635
    MVR: power = 0.34, FDR = 0.105, time = 12.778
    SDP: power = 0.38, FDR = 0.174, time = 60.466
    
    
    ipad ER: power = 0.34, FDR = 0.575, rank = 1, time = 0.093
    ipad GR: power = 0.34, FDR = 0.575, rank = 1, time = 0.1
    ipad VE: power = 0.34, FDR = 0.585, rank = 91, time = 0.101
    ME: power = 0.24, FDR = 0.077, time = 14.06
    MVR: power = 0.24, FDR = 0.077, time = 13.026
    SDP: power = 0.26, FDR = 0.235, time = 59.403
    
    
    ipad ER: power = 0.38, FDR = 0.424, rank = 1, time = 0.087
    ipad GR: power = 0.38, FDR = 0.424, rank = 1, time = 0.084
    ipad VE: power = 0.38, FDR = 0.441, rank = 91, time = 0.09
    ME: power = 0.24, FDR = 0.0, time = 13.807
    MVR: power = 0.24, FDR = 0.077, time = 12.41
    SDP: power = 0.4, FDR = 0.048, time = 57.882
    
    
    ipad ER: power = 0.28, FDR = 0.391, rank = 1, time = 0.091
    ipad GR: power = 0.28, FDR = 0.391, rank = 1, time = 0.085
    ipad VE: power = 0.28, FDR = 0.417, rank = 91, time = 0.111
    ME: power = 0.32, FDR = 0.2, time = 13.484
    MVR: power = 0.34, FDR = 0.15, time = 12.471
    SDP: power = 0.36, FDR = 0.217, time = 57.782
    
    
    ipad ER: power = 0.4, FDR = 0.5, rank = 1, time = 0.084
    ipad GR: power = 0.4, FDR = 0.5, rank = 1, time = 0.082
    ipad VE: power = 0.42, FDR = 0.5, rank = 91, time = 0.11
    ME: power = 0.34, FDR = 0.0, time = 13.35
    MVR: power = 0.34, FDR = 0.0, time = 12.35
    SDP: power = 0.34, FDR = 0.0, time = 57.646
    
    
    ipad ER: power = 0.46, FDR = 0.41, rank = 1, time = 0.085
    ipad GR: power = 0.46, FDR = 0.41, rank = 1, time = 0.097
    ipad VE: power = 0.46, FDR = 0.425, rank = 90, time = 0.106
    ME: power = 0.44, FDR = 0.154, time = 13.544
    MVR: power = 0.44, FDR = 0.154, time = 12.512
    SDP: power = 0.28, FDR = 0.067, time = 77.268
    
    
    ipad ER: power = 0.48, FDR = 0.478, rank = 1, time = 0.105
    ipad GR: power = 0.48, FDR = 0.478, rank = 1, time = 0.18
    ipad VE: power = 0.48, FDR = 0.538, rank = 90, time = 0.092
    ME: power = 0.28, FDR = 0.067, time = 13.719
    MVR: power = 0.26, FDR = 0.071, time = 12.733
    SDP: power = 0.36, FDR = 0.143, time = 59.276
    
    
    ipad ER: power = 0.36, FDR = 0.471, rank = 1, time = 0.095
    ipad GR: power = 0.36, FDR = 0.471, rank = 1, time = 0.09
    ipad VE: power = 0.36, FDR = 0.419, rank = 90, time = 0.095
    ME: power = 0.18, FDR = 0.1, time = 23.572
    MVR: power = 0.18, FDR = 0.1, time = 26.353
    SDP: power = 0.26, FDR = 0.133, time = 68.618
    
    
    ipad ER: power = 0.48, FDR = 0.4, rank = 1, time = 0.094
    ipad GR: power = 0.48, FDR = 0.4, rank = 1, time = 0.089
    ipad VE: power = 0.46, FDR = 0.439, rank = 92, time = 0.1
    ME: power = 0.48, FDR = 0.172, time = 13.986
    MVR: power = 0.48, FDR = 0.143, time = 13.283
    SDP: power = 0.46, FDR = 0.179, time = 61.874
    
    
    ipad ER: power = 0.4, FDR = 0.574, rank = 1, time = 0.09
    ipad GR: power = 0.4, FDR = 0.574, rank = 1, time = 0.095
    ipad VE: power = 0.4, FDR = 0.583, rank = 91, time = 0.098
    ME: power = 0.22, FDR = 0.083, time = 13.932
    MVR: power = 0.22, FDR = 0.154, time = 13.347
    SDP: power = 0.24, FDR = 0.077, time = 59.452
    
    
    ipad ER: power = 0.34, FDR = 0.585, rank = 1, time = 0.084
    ipad GR: power = 0.34, FDR = 0.585, rank = 1, time = 0.096
    ipad VE: power = 0.34, FDR = 0.553, rank = 91, time = 0.094
    ME: power = 0.3, FDR = 0.118, time = 14.271
    MVR: power = 0.3, FDR = 0.0, time = 12.45
    SDP: power = 0.38, FDR = 0.174, time = 56.762
    
    
    ipad ER: power = 0.46, FDR = 0.465, rank = 1, time = 0.082
    ipad GR: power = 0.46, FDR = 0.465, rank = 1, time = 0.085
    ipad VE: power = 0.46, FDR = 0.425, rank = 91, time = 0.091
    ME: power = 0.4, FDR = 0.048, time = 12.84
    MVR: power = 0.4, FDR = 0.048, time = 12.205
    SDP: power = 0.44, FDR = 0.083, time = 56.775
    
    
    ipad ER: power = 0.44, FDR = 0.353, rank = 1, time = 0.081
    ipad GR: power = 0.44, FDR = 0.353, rank = 1, time = 0.081
    ipad VE: power = 0.46, FDR = 0.378, rank = 91, time = 0.089
    ME: power = 0.5, FDR = 0.107, time = 12.742
    MVR: power = 0.5, FDR = 0.107, time = 12.327
    SDP: power = 0.56, FDR = 0.152, time = 56.161
    
    



```julia
@show result;
```

    result = 6×4 DataFrame
     Row │ method   power    FDR      time
         │ String   Float64  Float64  Float64
    ─────┼───────────────────────────────────────
       1 │ IPAD-er    0.367  0.40725   0.0903964
       2 │ IPAD-gr    0.367  0.40725   0.0946635
       3 │ IPAD-ve    0.37   0.41525   0.100552
       4 │ ME         0.348  0.0936   14.2259
       5 │ MVR        0.344  0.0896   13.6386
       6 │ SDP        0.362  0.09235  60.8669


Summary
+ IPAD methods have extremely inflated FDR (target = 10%)
