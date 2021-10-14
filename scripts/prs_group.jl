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

function R2(X::AbstractMatrix, y::AbstractVector, β̂::AbstractVector)
    μ = y - X * β̂
    tss = y .- mean(y)
    return 1 - dot(μ, μ) / dot(tss, tss)
end
function TP(correct_groups, signif_groups)
    return length(signif_groups ∩ correct_groups) / length(correct_groups)
end
function FDR(correct_groups, signif_groups)
    FP = length(signif_groups) - length(signif_groups ∩ correct_groups) # number of false positives
    # FPR = FP / (FP + TN) # https://en.wikipedia.org/wiki/False_positive_rate#Definition
    FDR = FP / length(signif_groups)
    return FDR
end
function tune_k(y::AbstractVector, xko_la::AbstractMatrix, original::Vector{Int},
    knockoff::Vector{Int}, groups::Vector{Int}, fdr::Float64, best_k::Int
    )
    # do a grid search for best sparsity level
    best_β = Float64[]
    best_err = Inf
    for cur_k in best_k:5:round(Int, 1.5best_k)
        result = fit_iht(y, xko_la, k=cur_k, init_beta=true, max_iter=500)
        W = coefficient_diff(result.beta, groups, original, knockoff)
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
function get_signif_groups(β, groups)
    correct_groups = Int[]
    for i in findall(!iszero, β)
        g = groups[i]
        g ∈ correct_groups || push!(correct_groups, g)
    end
    return correct_groups
end

function run_sims(x::SnpArray, knockoff_idx::BitVector, groups::Vector{Int}, seed::Int)
    #
    # import data (first 10000 samples of chr 10)
    #
    chr = 10
    original = findall(knockoff_idx .== false)
    knockoff = findall(knockoff_idx)
    xla = convert(Matrix{Float64}, @view(x[1:10000, original]), center=true, scale=true, impute=true)
    xko_la = convert(Matrix{Float64}, @view(x[1:10000, :]), center=true, scale=true, impute=true)
    cur_dir = pwd()

    for fdr in [0.05, 0.1, 0.25, 0.5]
        top_dir = cur_dir * "/fdr$fdr/"
        new_dir = cur_dir * "/fdr$fdr/sim$seed"
        isdir(top_dir) || mkdir(top_dir)
        isdir(new_dir) || mkdir(new_dir)
        cd(new_dir)
        #
        # simulate phenotypes using UKB chr10 subset
        #
        n, p = size(xla)
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
        writedlm("y_true.txt", y)
        writedlm("beta_true.txt", β)

        #
        # Run standard IHT
        #
        path = 10:10:200
        mses = cv_iht(y, xla, path=path, init_beta=true)
        GC.gc()
        k_rough_guess = path[argmin(mses)]
        dense_path = (k_rough_guess - 9):(k_rough_guess + 9)
        mses_new = cv_iht(y, xla, path=path, init_beta=true)
        GC.gc()
        result = fit_iht(y, xla, k=dense_path[argmin(mses_new)], init_beta=true, max_iter=500)
        @show result
        writedlm("iht.beta", result.beta)
        GC.gc()

        #
        # Run standard lasso
        #
        cv = glmnetcv(xla, y, nfolds=5, parallel=true) 
        writedlm("lasso.beta", coef(cv))

        #
        # run knockoff IHT 
        #
        path = 10:10:200
        mses = cv_iht(y, xko_la, path=path, init_beta=true)
        GC.gc()
        k_rough_guess = path[argmin(mses)]
        dense_path = (k_rough_guess - 9):(k_rough_guess + 9)
        mses_new = cv_iht(y, xko_la, path=path, init_beta=true)
        GC.gc()
        result = fit_iht(y, xko_la, k=dense_path[argmin(mses_new)],
            init_beta=true, max_iter=500)
        @show result
        writedlm("iht.knockoff.beta", result.beta)

        #
        # run knockoff IHT with wrapped cross validation
        #
        path = 10:10:200
        z = ones(Float64, 10000)
        mses = cv_iht_knockoff(y, xko_la, z, original, knockoff, fdr, path=path,
            init_beta=true, group_ko=groups)
        GC.gc()
        k_rough_guess = path[argmin(mses)]
        dense_path = (k_rough_guess - 9):(k_rough_guess + 9)
        mses_new = cv_iht_knockoff(y, xko_la, z, original, knockoff, fdr,
            path=dense_path, init_beta=true, group_ko=groups)
        GC.gc()
        best_k = dense_path[argmin(mses_new)]
        best_β = tune_k(y, xko_la, original, knockoff, groups, fdr, best_k)
        writedlm("iht.knockoff.cv.beta", best_β)

        #
        # Run knockoff lasso
        #
        cv = glmnetcv(xko_la, y, nfolds=5, parallel=true)
        writedlm("lasso.knockoff.beta", coef(cv))

        #
        # compare R2 across populations, save result in a dataframe
        #
        β_iht = vec(readdlm("iht.beta"))
        β_iht_knockoff = extract_beta(vec(readdlm("iht.knockoff.beta")), fdr, groups,
            original, knockoff)
        β_iht_knockoff_cv = extract_beta(vec(readdlm("iht.knockoff.cv.beta")),
            fdr, groups, original, knockoff)
        β_lasso = vec(readdlm("lasso.beta"))
        β_lasso_knockoff = extract_beta(vec(readdlm("lasso.knockoff.beta")),
            fdr, groups, original, knockoff)

        populations = ["african", "asian", "bangladeshi", "british", "caribbean", "chinese",
            "indian", "irish", "pakistani", "white_asian", "white_black", "white"]

        df = DataFrame(pop = String[], IHT_R2 = Float64[], IHT_ko_R2 = Float64[],
            IHT_ko_cv_R2 = Float64[], LASSO_R2 = Float64[], LASSO_ko_R2 = Float64[])

        for pop in populations
            xpop = SnpArray("/scratch/users/bbchu/ukb/populations/chr10/ukb.chr$chr.$pop.bed")
            Xpop = SnpLinAlg{Float64}(xpop, center=true, scale=true, impute=true)
            # simulate "true" phenotypes for these populations
            ytrue = Xpop * β + rand(ϵ, size(Xpop, 1))
            # IHT
            iht_r2 = R2(Xpop, ytrue, β_iht)
            # knockoff IHT
            iht_ko_r2 = R2(Xpop, ytrue, β_iht_knockoff)
            # knockoff IHT cv
            iht_ko_cv_r2 = R2(Xpop, ytrue, β_iht_knockoff_cv)
            # lasso β
            lasso_r2 = R2(Xpop, ytrue, β_lasso)
            # knockoff lasso β
            lasso_ko_r2 = R2(Xpop, ytrue, β_lasso_knockoff)
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
        correct_groups = get_signif_groups(β, groups)
        push!(df, hcat("TPP", TP(correct_groups, findall(!iszero, β_iht)),
            TP(correct_groups, get_signif_groups(β_iht_knockoff, groups)),
            TP(correct_groups, get_signif_groups(β_iht_knockoff_cv, groups)),
            TP(correct_groups, get_signif_groups(β_lasso, groups)),
            TP(correct_groups, get_signif_groups(β_lasso_knockoff, groups))
            ))
        # count FDR
        push!(df, hcat("FDR", FDR(correct_groups, findall(!iszero, β_iht)),
            FDR(correct_groups, get_signif_groups(β_iht_knockoff, groups)),
            FDR(correct_groups, get_signif_groups(β_iht_knockoff_cv, groups)),
            FDR(correct_groups, get_signif_groups(β_lasso, groups)),
            FDR(correct_groups, get_signif_groups(β_lasso_knockoff, groups))
            ))

        @show df
        CSV.write("summary.txt", df)
    end
end

#
# import key and data
#
keyfile = "/scratch/users/bbchu/ukb/groups/Radj20_K50_s0/ukb_gen_chr10.key"
df = CSV.read(keyfile, DataFrame)
groups = convert(Vector{Int}, df[!, :Group])
knockoff_idx = convert(BitVector, df[!, :Knockoff])
x = SnpArray("/scratch/users/bbchu/ukb/groups/Radj20_K50_s0/ukb_gen_chr10.bed")

#
# Run simulation (via `julia --threads 16 5`)
# each seed is a different run
#
seed = parse(Int, ARGS[1])
run_sims(x, knockoff_idx, groups, seed)
