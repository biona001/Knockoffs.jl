using Revise
using SnpArrays
using DelimitedFiles
using Random
using MendelIHT
using Knockoffs
using LinearAlgebra
using GLMNet
using Distributions
using DataFrames
using CSV
using Printf
BLAS.set_num_threads(1)

# predict with estimated β̂ (R2 = 1 - RSS/TSS)
function R2(X::AbstractMatrix, y::AbstractVector, β̂::AbstractVector)
    μ = y - X * β̂
    tss = y .- mean(y)
    return 1 - dot(μ, μ) / dot(tss, tss)
end

# predict with a low dimensional fit
function R2(Xtrain::AbstractMatrix, Xtest::AbstractMatrix,
    ytrain::AbstractVector, ytest::AbstractVector, β̂::AbstractVector)
    # fit low dimensional model on original data
    idx = findall(!iszero, β̂)
    β_new = zeros(length(β̂))
    β_new[idx] .= Xtrain[:, idx] \ ytrain
    # predict with low diemensional model on new data
    μ = ytest - Xtest * β_new
    t = ytest .- mean(ytest)
    return 1 - dot(μ, μ) / dot(t, t)
end

function TP(correct_groups, signif_groups)
    return length(signif_groups ∩ correct_groups) / length(correct_groups)
end

function FDR(correct_groups, signif_groups)
    FP = length(signif_groups) - length(signif_groups ∩ correct_groups) # number of false positives
    # FPR = FP / (FP + TN) # https://en.wikipedia.org/wiki/False_positive_rate#Definition
    FDR = FP / length(signif_groups)
    return length(signif_groups) == 0 ? 0 : FDR
end

function tune_k(y::AbstractVector, xko_la::AbstractMatrix, covar::AbstractVecOrMat,
    original::Vector{Int}, knockoff::Vector{Int}, fdr::Float64, best_k::Int
    )
    # do a grid search for best sparsity level
    best_result = nothing
    best_err = Inf
    for cur_k in best_k:5:round(Int, 2best_k)
        result = fit_iht(y, xko_la, covar, k=cur_k, init_beta=true, max_iter=500)
        W = coefficient_diff(result.beta, original, knockoff)
        τ = threshold(W, fdr, :knockoff)
        detected = count(x -> x ≥ τ, W)
        if abs(detected - best_k) < best_err
            best_result = deepcopy(result)
            best_err = abs(detected - best_k)
        end
        println("wrapped CV says best_k = $best_k; using k = $cur_k detected $detected")
        GC.gc()
    end
    return best_result
end
# function tune_k(y::AbstractVector, xko_la::AbstractMatrix, original::Vector{Int},
#     knockoff::Vector{Int}, fdr::Float64, best_k::Int
#     )
#     cur_k = best_k
#     detected = 0
#     for iter in 1:5
#         result = fit_iht(y, xko_la, k=cur_k, init_beta=true, max_iter=500)
#         W = coefficient_diff(result.beta, original, knockoff)
#         τ = threshold(W, fdr, :knockoff)
#         detected = count(x -> x ≥ τ, W)
#         if detected < best_k
#             cur_k += best_k - detected
#         else
#             break
#         end
#         GC.gc()
#     end
#     @show result
#     println("wrapped CV says best_k = $best_k; after tuning detected = $detected")
#     return result.beta
# end

# todo: bolt
function run_sims(seed::Int; 
    k = 10, # number of causal SNPs
    use_PCA = false, 
    combine_beta=false,
    extra_k = 0,
    confounders = 0, # number of confounders
    causal_snp_upper_r2 = 1, # causal SNPs and their knockoffs must have correlation less than causal_snp_upper_r2
    causal_snp_lower_r2 = 0  # causal SNPs and their knockoffs must have correlation larger than causal_snp_lower_r2
    )
    #
    # import fastphase knockoffs
    #
    # chr = 10
    # plinkname = "/scratch/users/bbchu/ukb/subset/ukb.10k.chr$chr"
    # original = vec(readdlm("/scratch/users/bbchu/ukb/subset/ukb.chr$chr.original.snp.index", Int))
    # knockoff = vec(readdlm("/scratch/users/bbchu/ukb/subset/ukb.chr$chr.knockoff.snp.index", Int))
    # x = SnpArray(plinkname)
    # xko = SnpArray(knockoffname)
    # xla = convert(Matrix{Float64}, x, center=true, scale=true, impute=true)
    # xko_la = convert(Matrix{Float64}, xko, center=true, scale=true, impute=true)
    # cur_dir = pwd()

    #
    # import shapeit knockoffs
    #
    # chr = 10
    # plinkname = "/scratch/users/bbchu/ukb_SHAPEIT/knockoffs/ukb_gen_chr$(chr)_ibd1_res0" #shapeit knockoffs
    # # plinkname = "/scratch/users/bbchu/ukb_SHAPEIT/decorrelated_knockoffs/ukb_gen_chr$(chr)_ibd1_res0_decorrelated" #shapeit knockoffs (decorrelated)
    # xdata = SnpData(plinkname)
    # isknockoff = endswith.(xdata.snp_info[!, :snpid], ".k")
    # original, knockoff = Int[], Int[]
    # for i in 1:size(xdata)[2]
    #     isknockoff[i] ? push!(knockoff, i) : push!(original, i)
    # end
    # x = xdata.snparray
    # xla = convert(Matrix{Float64}, @view(x[1:10000, original]), center=true, scale=true, impute=true)
    # xko_la = convert(Matrix{Float64}, @view(x[1:10000, :]), center=true, scale=true, impute=true)
    # cur_dir = pwd()

    #
    # import Julia fastphase knockoffs
    #
    chr = 10
    # plinkname = "/scratch/users/bbchu/fastphase/1k/ukb.10k.chr10.bed"
    # knockoffname = "/scratch/users/bbchu/fastphase/1k/ukb.10k.chr10.merged.bed"
    # original = collect(29482:58962)
    # knockoff = collect(1:29481)
    plinkname = "/scratch/users/bbchu/fastphase/1k/ukb.10k.chr10.bed"
    knockoffname = "/scratch/users/bbchu/fastphase/1k/knockoff.bed"
    bimfile = CSV.read("/scratch/users/bbchu/fastphase/1k/knockoff.bim", DataFrame, header=false)
    original = findall(!endswith(".k"), bimfile[!, 2])
    knockoff = findall(endswith(".k"), bimfile[!, 2])
    x = SnpArray(plinkname)
    xko = SnpArray(knockoffname)
    xla = convert(Matrix{Float64}, x, center=true, scale=true, impute=true)
    xko_la = convert(Matrix{Float64}, xko, center=true, scale=true, impute=true)
    cur_dir = pwd()

    #
    # Import PCs
    # if use_PCA, make augmented design matrix for lasso and covarirate matrix for IHT
    # 
    z = readdlm("/scratch/users/bbchu/ukb_SHAPEIT/subset_pca/ukb.10k.chr$chr.projections.txt")
    standardize!(z)
    xla_full = use_PCA ? [xla z] : xla
    xko_la_full = use_PCA ? [xko_la z] : xko_la
    covar = use_PCA ? [ones(size(xla, 1)) z] : ones(size(xla, 1))

    #
    # simulate phenotypes using UKB chr10 subset
    #
    n, p = size(xla)
    # simulate β
    Random.seed!(seed)
    h2 = 0.5 # heritability
    d = Normal(0, sqrt(h2 / (2k))) # from paper: Efficient Implementation of Penalized Regression for Genetic Risk Prediction
    β = zeros(p)
    # β[1:k] .= rand(d, k)
    # shuffle!(β)
    # causal SNPs can only be those that aren't very correlated with their knockoffs
    possible_causal_snp_idx = Int[]
    for snp in 1:size(xla, 2)
        r2 = cor(@view(xko_la[:, 2snp]), @view(xko_la[:, 2snp - 1]))
        causal_snp_lower_r2 ≤ r2 ≤ causal_snp_upper_r2 && push!(possible_causal_snp_idx, snp)
    end
    shuffle!(possible_causal_snp_idx)
    β[possible_causal_snp_idx[1:k]] .= rand(d, k)
    # simulate y
    ϵ = Normal(0, 1 - h2)
    y = xla * β + rand(ϵ, n)
    #
    # confounders are PCs and have effect size ±0.2
    #
    if confounders > 0
        PCs = z[:, 1:confounders]
        γ = [rand(-1:2:1) * 0.2 for i in 1:confounders]
        y += PCs * γ
    end

    #
    # Run standard IHT
    #
    Random.seed!(seed)
    path = 10:10:200
    mses = cv_iht(y, xla, covar, path=path, init_beta=true)
    GC.gc()
    Random.seed!(seed)
    k_rough_guess = path[argmin(mses)]
    dense_path = (k_rough_guess - 9):(k_rough_guess + 9)
    mses_new = cv_iht(y, xla, covar, path=dense_path, init_beta=true)
    GC.gc()
    Random.seed!(seed)
    iht_result = fit_iht(y, xla, covar, k=dense_path[argmin(mses_new)], init_beta=true, max_iter=500)
    @show iht_result
    GC.gc()

    #
    # Run standard lasso
    #
    Random.seed!(seed)
    @time lasso_cv = glmnetcv(xla_full, y, nfolds=5, parallel=true) 

    #
    # run knockoff IHT 
    #
    Random.seed!(seed)
    path = 10:10:200
    mses = cv_iht(y, xko_la, covar, path=path, init_beta=true)
    GC.gc()
    Random.seed!(seed)
    k_rough_guess = path[argmin(mses)]
    dense_path = (k_rough_guess - 9):(k_rough_guess + 9)
    mses_new = cv_iht(y, xko_la, covar, path=dense_path, init_beta=true)
    GC.gc()
    Random.seed!(seed)
    iht_ko_result = fit_iht(y, xko_la, covar, k=dense_path[argmin(mses_new)]+extra_k,
        init_beta=true, max_iter=500)
    @show iht_ko_result

    #
    # Run knockoff lasso
    #
    Random.seed!(seed)
    @time lasso_ko_cv = glmnetcv(xko_la_full, y, nfolds=5, parallel=true)

    #
    # Run bolt lmm (need to make covariate and phenotype file first)
    #
    # bedfile = data_dir * "train"
    # covfile = "cov_bolt.txt"
    # phefile = "y_bolt.txt"
    # outfile = "bolt_output.txt"
    # open(phefile, "w") do io
    #     println(io, "FID IID trait1") 
    #     for i in 1:size(xla, 1)
    #         println(io, "$i 1 $(ytrain[i])") 
    #     end
    # end
    # if use_PCA
    #     open(covfile, "w") do io
    #         println(io, "FID IID PC1 PC2 PC3 PC4 PC5") 
    #         for i in 1:size(xla, 1)
    #             println(io, "$i 1 ", z[i, 1], ' ', z[i, 2], ' ',  z[i, 3], ' ', z[i, 4], ' ', z[i, 5]) 
    #         end
    #     end
    #     run(`$bolt_exe --bfile=$bedfile --covarFile=$covfile 
    #         --phenoFile=$phefile --phenoCol=trait1 
    #         --qCovarCol=PC\{1:5\} --lmmInfOnly --numLeaveOutChunks=2 --statsFile $outfile`)
    # else
    #     run(`$bolt_exe --bfile=$bedfile --phenoFile=$phefile --phenoCol=trait1 
    #         --lmmInfOnly --numLeaveOutChunks=2 --statsFile $outfile`)
    # end

    for fdr in [0.05, 0.1, 0.25, 0.5]
        top_dir = cur_dir * "/fdr$fdr/"
        new_dir = cur_dir * "/fdr$fdr/sim$seed"
        isdir(top_dir) || mkdir(top_dir)
        isdir(new_dir) || mkdir(new_dir)
        cd(new_dir)
        writedlm("y_true.txt", y)
        writedlm("beta_true.txt", β)

        # save IHT, lasso, iht+knockoff, lasso+knockoff results
        writedlm("iht.beta", iht_result.beta)
        writedlm("iht.covariates", iht_result.c)
        writedlm("lasso.beta", coef(lasso_cv))
        writedlm("iht.knockoff.beta", iht_ko_result.beta)
        writedlm("iht.knockoff.covariates", iht_ko_result.c)
        writedlm("lasso.knockoff.beta", coef(lasso_ko_cv))

        #
        # run knockoff IHT with wrapped cross validation
        #
        # Random.seed!(seed)
        # chr = 10
        # path = 10:10:200
        # mses = cv_iht_knockoff(y, xko_la, covar, original, knockoff, fdr, path=path,
        #     init_beta=true, combine_beta = combine_beta)
        # Random.seed!(seed)
        # GC.gc()
        # k_rough_guess = path[argmin(mses)]
        # dense_path = (k_rough_guess - 9):(k_rough_guess + 9)
        # mses_new = cv_iht_knockoff(y, xko_la, covar, original, knockoff, fdr,
        #     path=dense_path, init_beta=true, combine_beta = combine_beta)
        # GC.gc()
        # # adjust sparsity level so it best matches sparsity chosen by ko filter
        # Random.seed!(seed)
        # best_k = dense_path[argmin(mses_new)]
        # best_result = tune_k(y, xko_la, covar, original, knockoff, fdr, best_k)
        # writedlm("iht.knockoff.cv.beta", best_result.beta)
        # writedlm("iht.knockoff.cv.covariates", best_result.c)

        #
        # compare R2 across populations, save result in a dataframe
        #
        # combine_beta = false
        p = length(β)
        β_iht = β_iht_knockoff = β_iht_knockoff_cv = zeros(p)
        # β_iht = iht_result.beta
        β_lasso = coef(lasso_cv)[1:p]
        # β_iht_knockoff = extract_beta(iht_ko_result.beta, fdr,
        #     original, knockoff, :knockoff, combine_beta)
        β_lasso_knockoff = extract_beta(coef(lasso_ko_cv), fdr,
            original, knockoff, :knockoff, combine_beta)[1:p]

        writedlm("iht.knockoff.beta.postfilter", β_iht_knockoff)
        writedlm("lasso.knockoff.beta.postfilter", β_lasso_knockoff)

        populations = ["african", "asian", "bangladeshi", "british", "caribbean", "chinese",
            "indian", "irish", "pakistani", "white_asian", "white_black", "white"]

        df = DataFrame(pop = String[], IHT_R2 = Float64[], IHT_ko_R2 = Float64[],
            LASSO_R2 = Float64[], LASSO_ko_R2 = Float64[])

        for pop in populations
            xtest = SnpArray("/scratch/users/bbchu/ukb_SHAPEIT/populations/chr10/ukb.chr$chr.$pop.bed") # 10k samples with all snps
            Xtest = SnpLinAlg{Float64}(xtest, center=true, scale=true, impute=true)
            # simulate "true" phenotypes for these populations
            Random.seed!(seed)
            ytest = Xtest * β + rand(ϵ, size(Xtest, 1))
            # # IHT
            # iht_r2 = R2(Xtest, ytest, β_iht)
            # # IHT knockoff (low dimensional fit)
            # iht_ko_r2 = R2(xla, Xtest, y, ytest, β_iht_knockoff)
            # lasso β
            lasso_r2 = R2(Xtest, ytest, β_lasso)
            # knockoff lasso β (low dimensional)
            lasso_ko_r2 = R2(xla, Xtest, y, ytest, β_lasso_knockoff)
            # save to dataframe
            # push!(df, hcat(pop, iht_r2, iht_ko_r2, lasso_r2, lasso_ko_r2))
            push!(df, hcat(pop, 0, 0, lasso_r2, lasso_ko_r2))
            GC.gc()
        end

        # count non-zero entries of β returned from cross validation
        push!(df, hcat("beta_non_zero_count",
            # count(!iszero, β_iht), 
            # count(!iszero, iht_ko_result.beta),
            0, 0,
            count(!iszero, β_lasso),
            count(!iszero, coef(lasso_ko_cv))
            ))
        # count non-zero entries after knockoff filter
        push!(df, hcat("beta_selected", 
            count(!iszero, β_iht), 
            count(!iszero, β_iht_knockoff),
            count(!iszero, β_lasso),
            count(!iszero, β_lasso_knockoff)
            ))
        # count TP proportion
        correct_snps = findall(!iszero, β)
        push!(df, hcat("TPP", 
            TP(correct_snps,findall(!iszero, β_iht)),
            TP(correct_snps, findall(!iszero, β_iht_knockoff)),
            TP(correct_snps, findall(!iszero, β_lasso)),
            TP(correct_snps, findall(!iszero, β_lasso_knockoff))
            ))
        # count FDR
        push!(df, hcat("FDR", 
            FDR(correct_snps, findall(!iszero, β_iht)),
            FDR(correct_snps, findall(!iszero, β_iht_knockoff)),
            FDR(correct_snps, findall(!iszero, β_lasso)),
            FDR(correct_snps, findall(!iszero, β_lasso_knockoff))
            ))

        @show df
        CSV.write("summary.txt", df)
    end
end

#
# Run simulation (via `julia prs.jl n`)
# where n is a seed
#
# seed = parse(Int, ARGS[1])
# # causal_snp_upper_r2 = parse(Float64, ARGS[2]) # 0.1, 0.2, ..., 1.0
# # causal_snp_lower_r2 = causal_snp_upper_r2 - 0.1 # 0, 0.1, ..., 0.9
# k = 100
# confounders = 0 # 1 PC
# run_sims(seed, k=k, use_PCA=false, confounders=confounders)
