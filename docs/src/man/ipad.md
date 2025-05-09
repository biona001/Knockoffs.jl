
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
using StatsKit
using ToeplitzMatrices
using Distributions
using Random
using CSV, DataFrames
```

## Test1: Simulate under factored model

This is design 1 of [IPAD: Stable Interpretable Forecasting with Knockoffs Inference](https://www.tandfonline.com/doi/abs/10.1080/01621459.2019.1654878?journalCode=uasa20). 

+ Here $n=500, p=1000$
+ ``X = F\Lambda' + \sqrt{r\theta}E`` where ``r = 3, \theta=1`` and ``F, \Lambda, E`` are ``n \times r, p \times r``, and ``n \times p`` iid gaussian matrices. 
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
        correct_position = findall(!iszero, βtrue)

        # ipad with ER
        Random.seed!(seed)
        ipad_er_t = @elapsed ipad_er = ipad(X, r_method = :er, m = m)
        ipad_er_ko_filter = fit_lasso(y, ipad_er)
        selected = ipad_er_ko_filter.selected[3]
        ipad_er_power = length(selected ∩ correct_position) / k
        ipad_er_fdr = length(setdiff(selected, correct_position)) / max(length(selected), 1)

        #ipad with GR
        Random.seed!(seed)
        ipad_gr_t = @elapsed ipad_gr = ipad(X, r_method = :gr, m = m)
        ipad_gr_ko_filter = fit_lasso(y, ipad_gr)
        selected = ipad_gr_ko_filter.selected[3]
        ipad_gr_power = length(selected ∩ correct_position) / k
        ipad_gr_fdr = length(setdiff(selected, correct_position)) / max(length(selected), 1)

        # ipad with VE
        Random.seed!(seed)
        ipad_ve_t = @elapsed ipad_ve = ipad(X, r_method = :ve, m = m)
        ipad_ve_ko_filter = fit_lasso(y, ipad_ve)
        selected = ipad_ve_ko_filter.selected[3]
        ipad_ve_power = length(selected ∩ correct_position) / k
        ipad_ve_fdr = length(setdiff(selected, correct_position)) / max(length(selected), 1)

        # ME knockoffs
        Random.seed!(seed)
        me_t = @elapsed me = modelX_gaussian_knockoffs(X, :maxent, m = m)
        me_ko_filter = fit_lasso(y, me)
        selected = me_ko_filter.selected[3]
        me_power = length(selected ∩ correct_position) / k
        me_fdr = length(setdiff(selected, correct_position)) / max(length(selected), 1)

        # MVR knockoffs
        Random.seed!(seed)
        mvr_t = @elapsed mvr = modelX_gaussian_knockoffs(X, :mvr, m = m)
        mvr_ko_filter = fit_lasso(y, mvr)
        selected = mvr_ko_filter.selected[3]
        mvr_power = length(selected ∩ correct_position) / k
        mvr_fdr = length(setdiff(selected, correct_position)) / max(length(selected), 1)

        # SDP (ccd) knockoffs
        Random.seed!(seed)
        sdp_t = @elapsed sdp = modelX_gaussian_knockoffs(X, :sdp_ccd, m = m)
        sdp_ko_filter = fit_lasso(y, sdp)
        selected = sdp_ko_filter.selected[3]
        sdp_power = length(selected ∩ correct_position) / k
        sdp_fdr = length(setdiff(selected, correct_position)) / max(length(selected), 1)

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

nsims = 10
result = compare_ipad(nsims);
```

    ipad ER: power = 0.98, FDR = 0.07547169811320754, rank = 3, time = 0.232
    ipad GR: power = 0.98, FDR = 0.07547169811320754, rank = 3, time = 0.068
    ipad VE: power = 1.0, FDR = 0.16666666666666666, rank = 257, time = 0.052
    ME: power = 0.0, FDR = 0.0, time = 2.312
    MVR: power = 0.0, FDR = 0.0, time = 2.953
    SDP: power = 0.48, FDR = 0.0, time = 11.525
    
    
    ipad ER: power = 1.0, FDR = 0.2647058823529412, rank = 3, time = 0.029
    ipad GR: power = 1.0, FDR = 0.2647058823529412, rank = 3, time = 0.028
    ipad VE: power = 1.0, FDR = 0.12280701754385964, rank = 263, time = 0.031
    ME: power = 0.5, FDR = 0.0, time = 2.256
    MVR: power = 0.4, FDR = 0.0, time = 2.944
    SDP: power = 0.94, FDR = 0.0784313725490196, time = 11.536
    
    
    ipad ER: power = 0.96, FDR = 0.02040816326530612, rank = 3, time = 0.043
    ipad GR: power = 0.96, FDR = 0.02040816326530612, rank = 3, time = 0.027
    ipad VE: power = 0.98, FDR = 0.09259259259259259, rank = 257, time = 0.031
    ME: power = 0.66, FDR = 0.0, time = 2.268
    MVR: power = 0.66, FDR = 0.0, time = 3.047
    SDP: power = 0.8, FDR = 0.024390243902439025, time = 11.565
    
    
    ipad ER: power = 0.94, FDR = 0.04081632653061224, rank = 3, time = 0.049
    ipad GR: power = 0.94, FDR = 0.04081632653061224, rank = 3, time = 0.027
    ipad VE: power = 0.92, FDR = 0.041666666666666664, rank = 260, time = 0.225
    ME: power = 0.0, FDR = 0.0, time = 2.61
    MVR: power = 0.0, FDR = 0.0, time = 2.943
    SDP: power = 0.0, FDR = 0.0, time = 11.484
    
    
    ipad ER: power = 1.0, FDR = 0.20634920634920634, rank = 3, time = 0.029
    ipad GR: power = 1.0, FDR = 0.20634920634920634, rank = 3, time = 0.03
    ipad VE: power = 0.98, FDR = 0.07547169811320754, rank = 256, time = 0.03
    ME: power = 0.78, FDR = 0.025, time = 2.255
    MVR: power = 0.64, FDR = 0.0, time = 3.009
    SDP: power = 0.9, FDR = 0.1, time = 11.51
    
    
    ipad ER: power = 1.0, FDR = 0.05660377358490566, rank = 3, time = 0.03
    ipad GR: power = 1.0, FDR = 0.05660377358490566, rank = 3, time = 0.029
    ipad VE: power = 0.9, FDR = 0.1346153846153846, rank = 253, time = 0.03
    ME: power = 0.84, FDR = 0.023255813953488372, time = 2.294
    MVR: power = 0.78, FDR = 0.025, time = 2.946
    SDP: power = 0.76, FDR = 0.05, time = 11.468
    
    
    ipad ER: power = 0.92, FDR = 0.09803921568627451, rank = 3, time = 0.033
    ipad GR: power = 0.92, FDR = 0.09803921568627451, rank = 3, time = 0.036
    ipad VE: power = 0.92, FDR = 0.041666666666666664, rank = 256, time = 0.029
    ME: power = 0.54, FDR = 0.0, time = 2.257
    MVR: power = 0.62, FDR = 0.0, time = 2.95
    SDP: power = 0.8, FDR = 0.047619047619047616, time = 11.574
    
    
    ipad ER: power = 1.0, FDR = 0.10714285714285714, rank = 3, time = 0.029
    ipad GR: power = 1.0, FDR = 0.10714285714285714, rank = 3, time = 0.03
    ipad VE: power = 1.0, FDR = 0.15254237288135594, rank = 260, time = 0.028
    ME: power = 0.66, FDR = 0.0, time = 2.254
    MVR: power = 0.66, FDR = 0.0, time = 3.003
    SDP: power = 0.8, FDR = 0.0, time = 11.761
    
    
    ipad ER: power = 1.0, FDR = 0.09090909090909091, rank = 3, time = 0.031
    ipad GR: power = 1.0, FDR = 0.09090909090909091, rank = 3, time = 0.027
    ipad VE: power = 1.0, FDR = 0.07407407407407407, rank = 257, time = 0.032
    ME: power = 0.0, FDR = 0.0, time = 2.245
    MVR: power = 0.0, FDR = 0.0, time = 2.952
    SDP: power = 0.86, FDR = 0.0, time = 11.547
    
    
    ipad ER: power = 0.94, FDR = 0.14545454545454545, rank = 3, time = 0.048
    ipad GR: power = 0.94, FDR = 0.14545454545454545, rank = 3, time = 0.026
    ipad VE: power = 0.94, FDR = 0.1896551724137931, rank = 255, time = 0.04
    ME: power = 0.68, FDR = 0.0, time = 2.269
    MVR: power = 0.7, FDR = 0.0, time = 2.962
    SDP: power = 0.9, FDR = 0.021739130434782608, time = 11.542
    
    



```julia
# check average
@show result;
```

    result = 6×4 DataFrame
     Row │ method   power    FDR         time
         │ String   Float64  Float64     Float64
    ─────┼──────────────────────────────────────────
       1 │ IPAD-er    0.974  0.11059      0.055441
       2 │ IPAD-gr    0.974  0.11059      0.032905
       3 │ IPAD-ve    0.964  0.109176     0.0528907
       4 │ ME         0.466  0.00482558   2.30208
       5 │ MVR        0.446  0.0025       2.97096
       6 │ SDP        0.724  0.032218    11.5513


Summary (when $X$ follows the IPAD model assumption)

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
        correct_position = findall(!iszero, βtrue)

        # ipad with ER
        Random.seed!(seed)
        ipad_er_t = @elapsed ipad_er = ipad(X, r_method = :er, m = m)
        ipad_er_ko_filter = fit_lasso(y, ipad_er)
        selected = ipad_er_ko_filter.selected[3]
        ipad_er_power = length(selected ∩ correct_position) / k
        ipad_er_fdr = length(setdiff(selected, correct_position)) / max(length(selected), 1)

        #ipad with GR
        Random.seed!(seed)
        ipad_gr_t = @elapsed ipad_gr = ipad(X, r_method = :gr, m = m)
        ipad_gr_ko_filter = fit_lasso(y, ipad_gr)
        selected = ipad_gr_ko_filter.selected[3]
        ipad_gr_power = length(selected ∩ correct_position) / k
        ipad_gr_fdr = length(setdiff(selected, correct_position)) / max(length(selected), 1)

        # ipad with VE
        Random.seed!(seed)
        ipad_ve_t = @elapsed ipad_ve = ipad(X, r_method = :ve, m = m)
        ipad_ve_ko_filter = fit_lasso(y, ipad_ve)
        selected = ipad_ve_ko_filter.selected[3]
        ipad_ve_power = length(selected ∩ correct_position) / k
        ipad_ve_fdr = length(setdiff(selected, correct_position)) / max(length(selected), 1)

        # ME knockoffs
        Random.seed!(seed)
        me_t = @elapsed me = modelX_gaussian_knockoffs(X, :maxent, μ, Σ, m = m)
        me_ko_filter = fit_lasso(y, me)
        selected = me_ko_filter.selected[3]
        me_power = length(selected ∩ correct_position) / k
        me_fdr = length(setdiff(selected, correct_position)) / max(length(selected), 1)

        # MVR knockoffs
        Random.seed!(seed)
        mvr_t = @elapsed mvr = modelX_gaussian_knockoffs(X, :mvr, μ, Σ,m = m)
        mvr_ko_filter = fit_lasso(y, mvr)
        selected = mvr_ko_filter.selected[3]
        mvr_power = length(selected ∩ correct_position) / k
        mvr_fdr = length(setdiff(selected, correct_position)) / max(length(selected), 1)

        # SDP (ccd) knockoffs
        Random.seed!(seed)
        sdp_t = @elapsed sdp = modelX_gaussian_knockoffs(X, :sdp_ccd, μ, Σ, m = m)
        sdp_ko_filter = fit_lasso(y, sdp)
        selected = sdp_ko_filter.selected[3]
        sdp_power = length(selected ∩ correct_position) / k
        sdp_fdr = length(setdiff(selected, correct_position)) / max(length(selected), 1)

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

nsims = 10
result = compare_ipad2(nsims);
```

    ipad ER: power = 0.0, FDR = 0.0, rank = 496, time = 0.052
    ipad GR: power = 0.64, FDR = 0.1794871794871795, rank = 1, time = 0.039
    ipad VE: power = 0.62, FDR = 0.20512820512820512, rank = 217, time = 0.044
    ME: power = 0.28, FDR = 0.0, time = 2.299
    MVR: power = 0.26, FDR = 0.0, time = 3.286
    SDP: power = 0.4, FDR = 0.0, time = 8.727
    
    
    ipad ER: power = 0.5, FDR = 0.24242424242424243, rank = 1, time = 0.043
    ipad GR: power = 0.5, FDR = 0.24242424242424243, rank = 1, time = 0.227
    ipad VE: power = 0.5, FDR = 0.24242424242424243, rank = 219, time = 0.041
    ME: power = 0.0, FDR = 0.0, time = 2.627
    MVR: power = 0.2, FDR = 0.0, time = 3.515
    SDP: power = 0.38, FDR = 0.05, time = 8.974
    
    
    ipad ER: power = 0.58, FDR = 0.21621621621621623, rank = 1, time = 0.039
    ipad GR: power = 0.58, FDR = 0.21621621621621623, rank = 1, time = 0.031
    ipad VE: power = 0.62, FDR = 0.24390243902439024, rank = 217, time = 0.078
    ME: power = 0.5, FDR = 0.10714285714285714, time = 2.364
    MVR: power = 0.48, FDR = 0.07692307692307693, time = 3.309
    SDP: power = 0.0, FDR = 0.0, time = 8.754
    
    
    ipad ER: power = 0.46, FDR = 0.14814814814814814, rank = 480, time = 0.049
    ipad GR: power = 0.62, FDR = 0.29545454545454547, rank = 10, time = 0.035
    ipad VE: power = 0.66, FDR = 0.2826086956521739, rank = 213, time = 0.049
    ME: power = 0.54, FDR = 0.12903225806451613, time = 2.489
    MVR: power = 0.46, FDR = 0.08, time = 3.403
    SDP: power = 0.4, FDR = 0.09090909090909091, time = 9.418
    
    
    ipad ER: power = 0.56, FDR = 0.2, rank = 1, time = 0.058
    ipad GR: power = 0.56, FDR = 0.2, rank = 1, time = 0.077
    ipad VE: power = 0.56, FDR = 0.2222222222222222, rank = 217, time = 0.073
    ME: power = 0.38, FDR = 0.13636363636363635, time = 2.52
    MVR: power = 0.5, FDR = 0.16666666666666666, time = 3.628
    SDP: power = 0.4, FDR = 0.09090909090909091, time = 8.948
    
    
    ipad ER: power = 0.0, FDR = 0.0, rank = 493, time = 0.039
    ipad GR: power = 0.72, FDR = 0.25, rank = 3, time = 0.048
    ipad VE: power = 0.74, FDR = 0.3508771929824561, rank = 218, time = 0.039
    ME: power = 0.64, FDR = 0.058823529411764705, time = 2.555
    MVR: power = 0.64, FDR = 0.058823529411764705, time = 3.265
    SDP: power = 0.62, FDR = 0.06060606060606061, time = 8.543
    
    
    ipad ER: power = 0.76, FDR = 0.3448275862068966, rank = 3, time = 0.029
    ipad GR: power = 0.76, FDR = 0.3448275862068966, rank = 3, time = 0.04
    ipad VE: power = 0.76, FDR = 0.36666666666666664, rank = 214, time = 0.064
    ME: power = 0.6, FDR = 0.09090909090909091, time = 2.302
    MVR: power = 0.6, FDR = 0.11764705882352941, time = 3.329
    SDP: power = 0.22, FDR = 0.0, time = 8.788
    
    
    ipad ER: power = 0.0, FDR = 0.0, rank = 495, time = 0.064
    ipad GR: power = 0.58, FDR = 0.23684210526315788, rank = 2, time = 0.036
    ipad VE: power = 0.56, FDR = 0.2, rank = 213, time = 0.034
    ME: power = 0.0, FDR = 0.0, time = 2.239
    MVR: power = 0.0, FDR = 0.0, time = 3.351
    SDP: power = 0.4, FDR = 0.13043478260869565, time = 8.523
    
    
    ipad ER: power = 0.68, FDR = 0.2916666666666667, rank = 1, time = 0.047
    ipad GR: power = 0.68, FDR = 0.2916666666666667, rank = 1, time = 0.036
    ipad VE: power = 0.7, FDR = 0.3137254901960784, rank = 218, time = 0.034
    ME: power = 0.0, FDR = 0.0, time = 2.29
    MVR: power = 0.0, FDR = 0.0, time = 3.332
    SDP: power = 0.0, FDR = 0.0, time = 8.96
    
    
    ipad ER: power = 0.6, FDR = 0.375, rank = 1, time = 0.105
    ipad GR: power = 0.6, FDR = 0.375, rank = 1, time = 0.025
    ipad VE: power = 0.54, FDR = 0.38636363636363635, rank = 213, time = 0.173
    ME: power = 0.44, FDR = 0.15384615384615385, time = 2.265
    MVR: power = 0.36, FDR = 0.1, time = 3.223
    SDP: power = 0.0, FDR = 0.0, time = 8.907
    
    



```julia
@show result;
```

    result = 6×4 DataFrame
     Row │ method   power    FDR        time
         │ String   Float64  Float64    Float64
    ─────┼────────────────────────────────────────
       1 │ IPAD-er    0.414  0.181828   0.0524697
       2 │ IPAD-gr    0.624  0.263192   0.059193
       3 │ IPAD-ve    0.626  0.281392   0.0629533
       4 │ ME         0.338  0.0676118  2.39515
       5 │ MVR        0.35   0.060006   3.36422
       6 │ SDP        0.282  0.0422859  8.85414


Summary (when $X$ does not follow the factored model)

+ IPAD methods have slightly~pretty inflated FDR (target = 10%) 
+ ER/GR/VE finds wildly differing ranks


## Test3: Simulate with gnomAD panel

+ Here we simulate ``X_i \sim N(0, \Sigma)`` where ``\Sigma`` is from the European [gnomAD LD panel](https://gnomad.broadinstitute.org/downloads#v2-linkage-disequilibrium). 
+ ``\Sigma`` can be downloaded and extract with the software [EasyLD.jl](https://github.com/biona001/EasyLD.jl). 


```julia
function compare_ipad3(nsims)
    # import panel
    datadir = "/Users/biona001/Benjamin_Folder/research/4th_project_groupKO/group_knockoff_test_data"
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
        correct_position = findall(!iszero, βtrue)

        # ipad with ER
        Random.seed!(seed)
        ipad_er_t = @elapsed ipad_er = ipad(X, r_method = :er, m = m)
        ipad_er_ko_filter = fit_lasso(y, ipad_er)
        selected = ipad_er_ko_filter.selected[3]
        ipad_er_power = length(selected ∩ correct_position) / k
        ipad_er_fdr = length(setdiff(selected, correct_position)) / max(length(selected), 1)

        #ipad with GR
        Random.seed!(seed)
        ipad_gr_t = @elapsed ipad_gr = ipad(X, r_method = :gr, m = m)
        ipad_gr_ko_filter = fit_lasso(y, ipad_gr)
        selected = ipad_gr_ko_filter.selected[3]
        ipad_gr_power = length(selected ∩ correct_position) / k
        ipad_gr_fdr = length(setdiff(selected, correct_position)) / max(length(selected), 1)

        # ipad with VE
        Random.seed!(seed)
        ipad_ve_t = @elapsed ipad_ve = ipad(X, r_method = :ve, m = m)
        ipad_ve_ko_filter = fit_lasso(y, ipad_ve)
        selected = ipad_ve_ko_filter.selected[3]
        ipad_ve_power = length(selected ∩ correct_position) / k
        ipad_ve_fdr = length(setdiff(selected, correct_position)) / max(length(selected), 1)

        # ME knockoffs
        Random.seed!(seed)
        me_t = @elapsed me = modelX_gaussian_knockoffs(X, :maxent, m = m)
        me_ko_filter = fit_lasso(y, me)
        selected = me_ko_filter.selected[3]
        me_power = length(selected ∩ correct_position) / k
        me_fdr = length(setdiff(selected, correct_position)) / max(length(selected), 1)

        # MVR knockoffs
        Random.seed!(seed)
        mvr_t = @elapsed mvr = modelX_gaussian_knockoffs(X, :mvr, m = m)
        mvr_ko_filter = fit_lasso(y, mvr)
        selected = mvr_ko_filter.selected[3]
        mvr_power = length(selected ∩ correct_position) / k
        mvr_fdr = length(setdiff(selected, correct_position)) / max(length(selected), 1)

        # SDP (ccd) knockoffs
        Random.seed!(seed)
        sdp_t = @elapsed sdp = modelX_gaussian_knockoffs(X, :sdp_ccd, m = m)
        sdp_ko_filter = fit_lasso(y, sdp)
        selected = sdp_ko_filter.selected[3]
        sdp_power = length(selected ∩ correct_position) / k
        sdp_fdr = length(setdiff(selected, correct_position)) / max(length(selected), 1)

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

nsims = 10
result = compare_ipad3(nsims);
```

    ipad ER: power = 0.54, FDR = 0.5645161290322581, rank = 1, time = 0.036
    ipad GR: power = 0.54, FDR = 0.5645161290322581, rank = 1, time = 0.03
    ipad VE: power = 0.54, FDR = 0.578125, rank = 90, time = 0.033
    ME: power = 0.32, FDR = 0.1111111111111111, time = 6.315
    MVR: power = 0.32, FDR = 0.1111111111111111, time = 6.39
    SDP: power = 0.34, FDR = 0.05555555555555555, time = 22.637
    
    
    ipad ER: power = 0.28, FDR = 0.6818181818181818, rank = 1, time = 0.036
    ipad GR: power = 0.28, FDR = 0.6818181818181818, rank = 1, time = 0.035
    ipad VE: power = 0.28, FDR = 0.6888888888888889, rank = 91, time = 0.029
    ME: power = 0.2, FDR = 0.0, time = 7.001
    MVR: power = 0.2, FDR = 0.0, time = 6.373
    SDP: power = 0.24, FDR = 0.0, time = 22.658
    
    
    ipad ER: power = 0.46, FDR = 0.43902439024390244, rank = 1, time = 0.029
    ipad GR: power = 0.46, FDR = 0.43902439024390244, rank = 1, time = 0.029
    ipad VE: power = 0.46, FDR = 0.4772727272727273, rank = 90, time = 0.029
    ME: power = 0.44, FDR = 0.12, time = 6.205
    MVR: power = 0.46, FDR = 0.14814814814814814, time = 6.395
    SDP: power = 0.5, FDR = 0.24242424242424243, time = 22.824
    
    
    ipad ER: power = 0.32, FDR = 0.5151515151515151, rank = 1, time = 0.039
    ipad GR: power = 0.32, FDR = 0.5151515151515151, rank = 1, time = 0.039
    ipad VE: power = 0.32, FDR = 0.5294117647058824, rank = 91, time = 0.042
    ME: power = 0.28, FDR = 0.0, time = 6.303
    MVR: power = 0.32, FDR = 0.058823529411764705, time = 6.45
    SDP: power = 0.44, FDR = 0.2903225806451613, time = 22.526
    
    
    ipad ER: power = 0.46, FDR = 0.41025641025641024, rank = 1, time = 0.028
    ipad GR: power = 0.46, FDR = 0.41025641025641024, rank = 1, time = 0.027
    ipad VE: power = 0.46, FDR = 0.425, rank = 91, time = 0.034
    ME: power = 0.42, FDR = 0.16, time = 6.166
    MVR: power = 0.42, FDR = 0.16, time = 6.561
    SDP: power = 0.44, FDR = 0.12, time = 23.091
    
    
    ipad ER: power = 0.56, FDR = 0.4716981132075472, rank = 1, time = 0.028
    ipad GR: power = 0.56, FDR = 0.4716981132075472, rank = 1, time = 0.044
    ipad VE: power = 0.56, FDR = 0.4716981132075472, rank = 90, time = 0.04
    ME: power = 0.46, FDR = 0.11538461538461539, time = 6.062
    MVR: power = 0.42, FDR = 0.08695652173913043, time = 6.837
    SDP: power = 0.48, FDR = 0.14285714285714285, time = 23.626
    
    
    ipad ER: power = 0.32, FDR = 0.627906976744186, rank = 1, time = 0.062
    ipad GR: power = 0.32, FDR = 0.627906976744186, rank = 1, time = 0.046
    ipad VE: power = 0.34, FDR = 0.6304347826086957, rank = 89, time = 0.05
    ME: power = 0.0, FDR = 0.0, time = 6.593
    MVR: power = 0.0, FDR = 0.0, time = 7.597
    SDP: power = 0.0, FDR = 0.0, time = 22.777
    
    
    ipad ER: power = 0.42, FDR = 0.6181818181818182, rank = 1, time = 0.029
    ipad GR: power = 0.42, FDR = 0.6181818181818182, rank = 1, time = 0.044
    ipad VE: power = 0.42, FDR = 0.6666666666666666, rank = 90, time = 0.069
    ME: power = 0.2, FDR = 0.23076923076923078, time = 6.604
    MVR: power = 0.2, FDR = 0.23076923076923078, time = 6.423
    SDP: power = 0.2, FDR = 0.16666666666666666, time = 23.46
    
    
    ipad ER: power = 0.58, FDR = 0.4727272727272727, rank = 1, time = 0.067
    ipad GR: power = 0.58, FDR = 0.4727272727272727, rank = 1, time = 0.031
    ipad VE: power = 0.58, FDR = 0.48214285714285715, rank = 90, time = 0.096
    ME: power = 0.32, FDR = 0.0, time = 6.368
    MVR: power = 0.3, FDR = 0.0, time = 6.641
    SDP: power = 0.42, FDR = 0.0, time = 22.832
    
    
    ipad ER: power = 0.36, FDR = 0.4857142857142857, rank = 1, time = 0.044
    ipad GR: power = 0.36, FDR = 0.4857142857142857, rank = 1, time = 0.045
    ipad VE: power = 0.36, FDR = 0.4857142857142857, rank = 91, time = 0.04
    ME: power = 0.28, FDR = 0.0, time = 6.269
    MVR: power = 0.28, FDR = 0.0, time = 6.526
    SDP: power = 0.4, FDR = 0.09090909090909091, time = 23.831
    
    



```julia
@show result;
```

    result = 6×4 DataFrame
     Row │ method   power    FDR        time
         │ String   Float64  Float64    Float64
    ─────┼─────────────────────────────────────────
       1 │ IPAD-er    0.43   0.5287      0.0398029
       2 │ IPAD-gr    0.43   0.5287      0.0370067
       3 │ IPAD-ve    0.432  0.543536    0.0462691
       4 │ ME         0.292  0.0737265   6.38856
       5 │ MVR        0.292  0.0795809   6.61936
       6 │ SDP        0.346  0.110874   23.0262


Summary (when $X$ is simulated based on real data)
+ IPAD methods have extremely inflated FDR (target = 10%)
+ model-X Knockoffs control FDR
