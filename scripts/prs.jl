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
function R2(Xtrain, Xtest, ytrain, ytest, β̂)
    # fit low dimensional model on original data
    idx = findall(!iszero, β̂)
    β_new = zeros(length(β̂))
    β_new[idx] .= Xtrain[:, idx] \ ytrain
    # predict with low diemensional model on new data
    μ = ytest - Xtest * β_new
    t = ytest .- mean(ytest)
    return 1 - dot(μ, μ) / dot(t, t)
end

function TP(correct_snps, signif_snps)
    return length(signif_snps ∩ correct_snps) / length(correct_snps)
end

function FDR(correct_groups, signif_groups)
    FP = length(signif_groups) - length(signif_groups ∩ correct_groups) # number of false positives
    # FPR = FP / (FP + TN) # https://en.wikipedia.org/wiki/False_positive_rate#Definition
    FDR = FP / length(signif_groups)
    return length(signif_groups) == 0 ? 0 : FDR
end

function tune_k(y::AbstractVector, xko_la::AbstractMatrix, original::Vector{Int},
    knockoff::Vector{Int}, fdr::Float64, best_k::Int
    )
    # do a grid search for best sparsity level
    best_β = Float64[]
    best_err = Inf
    for cur_k in best_k:5:round(Int, 1.5best_k)
        result = fit_iht(y, xko_la, k=cur_k, init_beta=true, max_iter=500)
        W = coefficient_diff(result.beta, original, knockoff)
        τ = threshold(W, fdr, :knockoff)
        detected = count(x -> x ≥ τ, W)
        if abs(detected - best_k) < best_err
            best_β = copy(result.beta)
            best_err = abs(detected - best_k)
        end
        println("wrapped CV says best_k = $best_k; using k = $cur_k detected $detected")
        GC.gc()
    end
    return best_β
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

# todo: use_PCA and bolt
function run_sims(seed::Int; combine_beta=false, extra_k = 0)
    #
    # import data
    #
    chr = 10
    plinkname = "/scratch/users/bbchu/ukb/subset/ukb.10k.chr$chr"
    knockoffname = "/scratch/users/bbchu/ukb/subset/ukb.10k.merged.chr$chr"
    x = SnpArray(plinkname * ".bed")
    xko = SnpArray(knockoffname * ".bed")
    xla = convert(Matrix{Float64}, x, center=true, scale=true, impute=true)
    xko_la = convert(Matrix{Float64}, xko, center=true, scale=true, impute=true)
    original = vec(readdlm("/scratch/users/bbchu/ukb/subset/ukb.chr$chr.original.snp.index", Int))
    knockoff = vec(readdlm("/scratch/users/bbchu/ukb/subset/ukb.chr$chr.knockoff.snp.index", Int))
    cur_dir = pwd()

    #
    # simulate phenotypes using UKB chr10 subset
    #
    n, p = size(x)
    # simulate β
    Random.seed!(seed)
    k = 100 # number of causal SNPs
    h2 = 0.5 # heritability
    d = Normal(0, sqrt(h2 / (2k))) # from paper: Efficient Implementation of Penalized Regression for Genetic Risk Prediction
    β = zeros(p)
    β[1:k] .= rand(d, k)
    shuffle!(β)
    # simulate y
    ϵ = Normal(0, 1 - h2)
    y = xla * β + rand(ϵ, n)

    #
    # Run standard IHT
    #
    Random.seed!(seed)
    path = 10:10:200
    mses = cv_iht(y, xla, path=path, init_beta=true)
    GC.gc()
    Random.seed!(seed)
    k_rough_guess = path[argmin(mses)]
    dense_path = (k_rough_guess - 9):(k_rough_guess + 9)
    mses_new = cv_iht(y, xla, path=dense_path, init_beta=true)
    GC.gc()
    Random.seed!(seed)
    iht_result = fit_iht(y, xla, k=dense_path[argmin(mses_new)], init_beta=true, max_iter=500)
    @show iht_result
    GC.gc()

    #
    # Run standard lasso
    #
    Random.seed!(seed)
    lasso_cv = glmnetcv(xla, y, nfolds=5, parallel=true) 

    #
    # run knockoff IHT 
    #
    Random.seed!(seed)
    path = 10:10:200
    mses = cv_iht(y, xko_la, path=path, init_beta=true)
    GC.gc()
    Random.seed!(seed)
    k_rough_guess = path[argmin(mses)]
    dense_path = (k_rough_guess - 9):(k_rough_guess + 9)
    mses_new = cv_iht(y, xko_la, path=dense_path, init_beta=true)
    GC.gc()
    Random.seed!(seed)
    iht_ko_result = fit_iht(y, xko_la, k=dense_path[argmin(mses_new)]+extra_k,
        init_beta=true, max_iter=500)
    @show iht_ko_result

    #
    # Run knockoff lasso
    #
    Random.seed!(seed)
    lasso_ko_cv = glmnetcv(xko_la, y, nfolds=5, parallel=true)

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
        writedlm("lasso.beta", coef(lasso_cv))
        writedlm("iht.knockoff.beta", iht_ko_result.beta)
        writedlm("lasso.knockoff.beta", coef(lasso_ko_cv))

        #
        # run knockoff IHT with wrapped cross validation
        #
        Random.seed!(seed)
        chr = 10
        path = 10:10:200
        z = ones(Float64, 10000)
        mses = cv_iht_knockoff(y, xko_la, z, original, knockoff, fdr, path=path,
            init_beta=true, combine_beta = combine_beta)
        Random.seed!(seed)
        GC.gc()
        k_rough_guess = path[argmin(mses)]
        dense_path = (k_rough_guess - 9):(k_rough_guess + 9)
        mses_new = cv_iht_knockoff(y, xko_la, z, original, knockoff, fdr,
            path=dense_path, init_beta=true, combine_beta = combine_beta)
        GC.gc()
        # adjust sparsity level so it best matches sparsity chosen by ko filter
        Random.seed!(seed)
        best_k = dense_path[argmin(mses_new)]
        best_β = tune_k(y, xko_la, original, knockoff, fdr, best_k)
        writedlm("iht.knockoff.cv.beta", best_β)

        #
        # compare R2 across populations, save result in a dataframe
        #
        β_iht = vec(readdlm("iht.beta"))
        β_lasso = vec(readdlm("lasso.beta"))
        β_iht_knockoff = extract_beta(vec(readdlm("iht.knockoff.beta")), fdr,
            original, knockoff, :knockoff, combine_beta)
        β_iht_knockoff_cv = extract_beta(vec(readdlm("iht.knockoff.cv.beta")),
            fdr, original, knockoff, :knockoff, combine_beta)
        β_lasso_knockoff = extract_beta(vec(readdlm("lasso.knockoff.beta")), fdr,
            original, knockoff, :knockoff, combine_beta)

        populations = ["african", "asian", "bangladeshi", "british", "caribbean", "chinese",
            "indian", "irish", "pakistani", "white_asian", "white_black", "white"]

        df = DataFrame(pop = String[], IHT_R2 = Float64[], IHT_ko_R2 = Float64[],
            IHT_ko_cv_R2 = Float64[], LASSO_R2 = Float64[], LASSO_ko_R2 = Float64[])

        for pop in populations
            xtest = SnpArray("/scratch/users/bbchu/ukb/populations/chr10/ukb.chr$chr.$pop.bed")
            Xtest = SnpLinAlg{Float64}(xtest, center=true, scale=true, impute=true)
            # simulate "true" phenotypes for these populations
            Random.seed!(seed)
            ytest = Xtest * β + rand(ϵ, size(Xtest, 1))
            # IHT
            iht_r2 = R2(Xtest, ytest, β_iht)
            # IHT knockoff (low dimensional fit)
            iht_ko_r2 = R2(xla, Xtest, y, ytest, β_iht_knockoff)
            # knockoff IHT cv (low dimensional)
            iht_ko_cv_r2 = R2(xla, Xtest, y, ytest, β_iht_knockoff_cv)
            # lasso β
            lasso_r2 = R2(Xtest, ytest, β_lasso)
            # knockoff lasso β (low dimensional)
            lasso_ko_r2 = R2(xla, Xtest, y, ytest, β_lasso_knockoff)
            # save to dataframe
            push!(df, hcat(pop, iht_r2, iht_ko_r2, iht_ko_cv_r2,
                lasso_r2, lasso_ko_r2))
            GC.gc()
        end

        # count non-zero entries of β returned from cross validation
        push!(df, hcat("beta_non_zero_count", count(!iszero, β_iht), 
            count(!iszero, vec(readdlm("iht.knockoff.beta"))),
            count(!iszero, vec(readdlm("iht.knockoff.cv.beta"))),
            count(!iszero, β_lasso),
            count(!iszero, vec(readdlm("lasso.knockoff.beta")))
            ))
        # count non-zero entries after knockoff filter
        push!(df, hcat("beta_selected", count(!iszero, β_iht), 
            count(!iszero, β_iht_knockoff),
            count(!iszero, β_iht_knockoff_cv), count(!iszero, β_lasso),
            count(!iszero, β_lasso_knockoff)
            ))
        # count TP proportion
        correct_snps = findall(!iszero, vec(readdlm("beta_true.txt")))
        push!(df, hcat("TPP", TP(correct_snps, findall(!iszero, β_iht)),
            TP(correct_snps, findall(!iszero, β_iht_knockoff)),
            TP(correct_snps, findall(!iszero, β_iht_knockoff_cv)),
            TP(correct_snps, findall(!iszero, β_lasso)),
            TP(correct_snps, findall(!iszero, β_lasso_knockoff))
            ))
        # count FDR
        push!(df, hcat("FDR", FDR(correct_snps, findall(!iszero, β_iht)),
            FDR(correct_snps, findall(!iszero, β_iht_knockoff)),
            FDR(correct_snps, findall(!iszero, β_iht_knockoff_cv)),
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
seed = parse(Int, ARGS[1])
run_sims(seed, combine_beta=true)
