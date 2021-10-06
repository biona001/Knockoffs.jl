#
# simulate phenotypes from a LMM
#
using Revise
using MendelIHT
using SnpArrays
using Random
using GLM
using DelimitedFiles
using Distributions
using LinearAlgebra
using CSV
using DataFrames
using StatsBase
using TraitSimulation
using Knockoffs
BLAS.set_num_threads(1)

"""
    βi ~ Uniform([-0.5, -0.45, ..., -0.05, 0.05, ..., 0.5]) chosen uniformly 
on the odd indices (i.e. real genotypes) across genome

k = Number of causal SNPs
p = Total number of SNPs
traits = Number of traits (phenotypes)
overlap = number of pleiotropic SNPs (affects each trait with effect size βi / r)
"""
function simulate_fixed_beta(k::Int, p::Int, traits::Int; overlap::Int=0)
    true_b = zeros(p, traits)
    effect_sizes = collect(0.05:0.05:0.5)
    k_indep = k - 2overlap # pleiotropic SNPs affect 2 phenotypes
    num_causal_snps = k_indep + overlap
    @assert num_causal_snps > 0 "number of causal SNPs should be positive but was $num_causal_snps"
    idx_causal_snps = sample(1:p, num_causal_snps, replace=false)
    @assert length(idx_causal_snps) == num_causal_snps "length(idx_causal_snps) = $(length(idx_causal_snps)) != num_causal_snps"
    shuffle!(idx_causal_snps)

    # pleiotropic SNPs affect 2 phenotypes
    for i in 1:overlap
        j = idx_causal_snps[i]
        rs = sample(1:traits, 2, replace=false)
        for r in rs
            true_b[j, r] = rand(-1:2:1) * effect_sizes[rand(1:10)]
        end
    end
    # non pleiotropic SNPs affect only 1 phenotype
    for i in (overlap+1):length(idx_causal_snps)
        idx = idx_causal_snps[i]
        true_b[idx, rand(1:traits)] = rand(-1:2:1) * effect_sizes[rand(1:10)]
    end

    @assert count(!iszero, true_b) == k "count(!iszero, true_b) = $(count(!iszero, true_b)) != k = $k"

    return true_b
end

"""
Trait covariance matrix is σg * Φ + σe * I where Φ is the GRM. 
"""
function simulate_polygenic(
    plinkname::String, k::Int, r::Int;
    seed::Int=2021, σg=0.1, σe=0.9, βoverlap=2, 
    )
    # set seed
    Random.seed!(seed)

    # simulate `.bed` file with no missing data
    x = SnpArray(plinkname * ".bed")
    xla = SnpLinAlg{Float64}(x, model=ADDITIVE_MODEL, impute=true, center=true, scale=true)
    n, p = size(x)

    # intercept is the only nongenetic covariate
    Z = ones(n, 1)
    intercepts = zeros(r)' # each trait have 0 intercept

    # simulate β
    B = simulate_fixed_beta(k, p, r, overlap=βoverlap)
    writedlm("sim$(seed)/trueb.txt", B)

    # between trait covariance matrix
    Σ = random_covariance_matrix(r)
    writedlm("sim$(seed)/true_cov.txt", Σ)

    # between sample covariance is identity + GRM
    Φ = readdlm(plinkname * ".grm")
    V = σg * Φ + σe * I

    # simulate y using TraitSimulations.jl (https://github.com/OpenMendel/TraitSimulation.jl/blob/master/src/modelframework.jl#L137)
    vc = @vc Σ ⊗ V
    μ = zeros(n, r)
    μ_null = zeros(n, r)
    LinearAlgebra.mul!(μ_null, Z, intercepts)
    mul!(μ, xla, B)
    BLAS.axpby!(1.0, μ_null, 1.0, μ)
    VCM_model = VCMTrait(Z, intercepts, xla, B, vc, μ)
    Y = Matrix(Transpose(simulate(VCM_model)))

    # simulate using Distributions.jl
    # μ = z * intercepts + xla * B
    # Y = rand(MatrixNormal(μ', Σ, V))
    
    return xla, Matrix(Z'), B, Σ, Y
end

"""
Computes power and false positive rates
- p: total number of SNPs
- pleiotropic_snps: Indices (or ID) of the true causal SNPs that affect >1 phenotype
- independent_snps: Indices (or ID) of the true causal SNPs that affect exactly 1 phenotype
- signif_snps: Indices (or ID) of SNPs that are significant after testing

returns: pleiotropic SNP's power, independent SNP's power, number of false positives, and false discovery rate
"""
function power_and_fdr(p::Int, pleiotropic_snps, independent_snps, signif_snps)
    pleiotropic_power = length(signif_snps ∩ pleiotropic_snps) / length(pleiotropic_snps)
    independent_power = length(signif_snps ∩ independent_snps) / length(independent_snps)
    correct_snps = pleiotropic_snps ∪ independent_snps
    FP = length(signif_snps) - length(signif_snps ∩ correct_snps) # number of false positives
    TN = p - length(signif_snps) # number of true negatives
    # FPR = FP / (FP + TN)
    FDR = FP / length(signif_snps)
    return pleiotropic_power, independent_power, FP, FDR
end

function make_grm(chr::Int)
    dir = "/scratch/users/bbchu/ukb/subset/"
    cd(dir)
    isfile(dir * "ukb.10k.chr$chr.bed") || error("PLINK file not present!")
    if !isfile(dir * "ukb.10k.chr$chr.grm")
        println("GRM file not present, generating robust GRM")
        Φ = SnpArrays.grm(SnpArray(dir * "ukb.10k.chr$chr.bed"), method=:Robust)
        writedlm("ukb.10k.chr$chr.grm", Φ)
    end
end

function one_simulation(
    k::Int, r::Int;
    seed::Int=2021, σg=0.1, σe=0.9, βoverlap=2,
    path=5:5:50, init_beta=false, model=:polygenic, debias=100, fdr=0.1
    )
    isdir("sim$seed") ? (return nothing) : mkdir("sim$seed")
    plinkname = "/scratch/users/bbchu/ukb/subset/ukb.10k.chr10"
    knockoff_file = "/scratch/users/bbchu/ukb/subset/ukb.10k.merged.chr10"
    grid = path[2] - path[1] - 1

    # simulate data
    Random.seed!(seed)
    if model == :polygenic
        xla, Z, B, Σ, Y = simulate_polygenic(plinkname, k, r,
            seed=seed, σg=σg, σe=σe, βoverlap=βoverlap)
    elseif model == :sparse
        xla, Z, B, Σ, Y = simulate_sparse(plinkname, k, r,
            seed=seed, σg=σg, σe=σe, βoverlap=βoverlap)
    else
        error("model misspecified!")
    end
    cd("sim$seed")
    writedlm("simulated_phenotypes.phen", Y', ',')

    correct_snps = unique([x[1] for x in findall(!iszero, B)])
    pleiotropic_snps, independent_snps = Int[], Int[]
    for snp in correct_snps
        count(x -> abs(x) > 0, @view(B[snp, :])) > 1 ? 
            push!(pleiotropic_snps, snp) : push!(independent_snps, snp)
    end
    # snpdata = SnpData("../" * plinkname)
    # pleiotropic_snp_rsid = snpdata.snp_info[pleiotropic_snps, :snpid]
    # independent_snp_rsid = snpdata.snp_info[independent_snps, :snpid]

    # run GEMMA (GRM is precomputed already)
    # run(`cp ../../$(plinkname).bed .`)
    # run(`cp ../../$(plinkname).bim .`)
    # run(`cp ../../$(plinkname).cXX.txt .`)
    # make_GEMMA_fam_file(xla, Y, plinkname)
    # pheno_columns = [string(ri) for ri in 1:r]
    # gemma_time = @elapsed begin
    #     run(`../../gemma -bfile $plinkname -k $(plinkname).cXX.txt -notsnp -lmm 1 -n $pheno_columns -o gemma.sim$seed`)
    # end
    # gemma_pleiotropic_power, gemma_independent_power, gemma_FP, gemma_FPR, gemma_λ = 
    #     process_gemma_result("output/gemma.sim$seed.assoc.txt", pleiotropic_snp_rsid, independent_snp_rsid)
    # println("GEMMA time = $gemma_time, pleiotropic power = $gemma_pleiotropic_power, independent power = $gemma_independent_power, FP = $gemma_FP, FDR = $gemma_FDR, gemma_λ=$gemma_λ")
    # mv("output/gemma.sim$seed.assoc.txt", "gemma.sim$seed.assoc.txt")
    # mv("output/gemma.sim$seed.log.txt", "gemma.sim$seed.log.txt")

    # run multivariate IHT
    mIHT_time = @elapsed begin
        mses = cross_validate(plinkname, MvNormal, path=path, phenotypes="simulated_phenotypes.phen";
            init_beta=init_beta, debias=debias)
        k_rough_guess = path[argmin(mses)]
        dense_path = (k_rough_guess - grid):(k_rough_guess + grid)
        mses_new = cross_validate(plinkname, MvNormal, path=dense_path, phenotypes="simulated_phenotypes.phen";
            init_beta=init_beta, debias=debias, cv_summaryfile="miht.cviht.summary.txt")
        iht_result = iht(plinkname, dense_path[argmin(mses_new)], MvNormal, phenotypes="simulated_phenotypes.phen";
            init_beta=init_beta, debias=debias, summaryfile="miht.summary.txt", 
            betafile="miht.beta.txt")
    end
    detected_snps = Int[]
    for i in 1:r
        β = iht_result.beta[i, :]
        detected_snps = detected_snps ∪ findall(!iszero, β)
    end
    mIHT_pleiotropic_power, mIHT_independent_power, mIHT_FP, mIHT_FDR = power_and_fdr(size(B, 1), pleiotropic_snps, independent_snps, detected_snps)
    println("multivariate IHT time = $mIHT_time, pleiotropic power = $mIHT_pleiotropic_power, independent power = $mIHT_independent_power, FP = $mIHT_FP, FDR = $mIHT_FDR")

    # run knockoff + multivariate IHT
    mIHT_ko_time = @elapsed begin
        mses = cross_validate(knockoff_file, MvNormal, path=path, phenotypes="simulated_phenotypes.phen";
            init_beta=init_beta, debias=debias)
        k_rough_guess = path[argmin(mses)]
        dense_path = (k_rough_guess - grid):(k_rough_guess + grid)
        mses_new = cross_validate(knockoff_file, MvNormal, path=dense_path, phenotypes="simulated_phenotypes.phen";
            init_beta=init_beta, debias=debias, cv_summaryfile="miht.ko.cviht.summary.txt")
        iht_result = iht(knockoff_file, dense_path[argmin(mses_new)], MvNormal, phenotypes="simulated_phenotypes.phen";
            init_beta=init_beta, debias=debias, summaryfile="miht.ko.summary.txt",
            betafile="miht.knockoff.beta.txt")
    end
    W = coefficient_diff(iht_result.beta', :interleaved)
    τ = threshold(W, fdr)
    detected_snps = findall(W .> τ)
    mIHT_ko_pleiotropic_power, mIHT_ko_independent_power, mIHT_ko_FP, mIHT_ko_FDR = power_and_fdr(size(B, 1), pleiotropic_snps, independent_snps, detected_snps)
    println("multivariate IHT with knockoffs time = $mIHT_ko_time, pleiotropic power = $mIHT_ko_pleiotropic_power, independent power = $mIHT_ko_independent_power, FP = $mIHT_ko_FP, FDR = $mIHT_ko_FDR")

    # run multiple univariate IHT
    # detected_snps = Int[]
    # uIHT_time = @elapsed begin
    #     for trait in 1:r
    #         mses = cross_validate(plinkname, Normal, path=path, phenotypes=trait+5;
    #             init_beta=init_beta, debias=debias)
    #         k_rough_guess = path[argmin(mses)]
    #         dense_path = (k_rough_guess == 5) ? (0:5) : ((k_rough_guess - 4):(k_rough_guess + 4))
    #         mses_new = cross_validate(plinkname, Normal, path=dense_path, phenotypes=trait+5;
    #             init_beta=init_beta, debias=debias, cv_summaryfile="uiht.cviht.summary$trait.txt")
    #         best_k = dense_path[argmin(mses_new)]
    #         if best_k > 0
    #             iht_result = iht(plinkname, best_k, Normal, phenotypes=trait+5;
    #                 init_beta=init_beta, debias=debias, summaryfile="uiht.summary$trait.txt")
    #             β = iht_result.beta
    #         else
    #             β = zeros(size(B, 2))
    #         end

    #         # save results
    #         detected_snps = detected_snps ∪ findall(!iszero, β)
    #         writedlm("univariate_iht_beta$trait.txt", β)
    #     end
    # end
    # uIHT_pleiotropic_power, uIHT_independent_power, uIHT_FP, uIHT_FPR = power_and_fdr(size(B, 1), pleiotropic_snps, independent_snps, detected_snps)
    # println("univariate IHT time = $uIHT_time, pleiotropic power = $uIHT_pleiotropic_power, independent power = $uIHT_independent_power, FP = $uIHT_FP, FPR = $uIHT_FPR")    

    # run MVPLINK
    # phenofile = plinkname * ".phen"
    # make_MVPLINK_fam_and_phen_file(xla, Y, plinkname)
    # mvplink_time = @elapsed run(`../../plink.multivariate --bfile $plinkname --noweb --mult-pheno $phenofile --mqfam`)
    # mvPLINK_pleitropic_power, mvPLINK_independent_power, mvPLINK_FP, mvPLINK_FPR, mvPLINK_λ = 
    #     process_mvPLINK("plink.mqfam.total", pleiotropic_snps, independent_snps)
    # println("mvPLINK time = $mvplink_time, pleiotropic power = $mvPLINK_pleitropic_power, independent power = $mvPLINK_independent_power, FP = $mvPLINK_FP, FPR = $mvPLINK_FPR, mvPLINK_λ=$mvPLINK_λ")

    # clean up
    # rm("plink.hh", force=true)
    # rm("$(plinkname).fam", force=true)
    # rm("$(plinkname).bed", force=true)
    # rm("$(plinkname).bim", force=true)
    # rm("$(plinkname).cXX.txt", force=true)

    # save summary stats
    n, p = size(xla)
    open("summary.txt", "w") do io
        println(io, "Simulation $seed summary")
        println(io, "n = $n, p = $p, k = $k, r = $r, βoverlap=$βoverlap")
        println(io, "debias=$debias, init_beta=$init_beta")
        model == :polygenic ? println(io, "model = $model, σg=$σg, σe=$σe") : println(io, "model = $model")
        println(io, "")
        println(io, "mIHT time = $mIHT_time seconds, pleiotropic power = $mIHT_pleiotropic_power, independent power = $mIHT_independent_power, FP = $mIHT_FP, FDR = $mIHT_FDR, λ = NaN")
        println(io, "mIHT knockoff time = $mIHT_ko_time seconds, pleiotropic power = $mIHT_ko_pleiotropic_power, independent power = $mIHT_ko_independent_power, FP = $mIHT_ko_FP, FDR = $mIHT_ko_FDR, λ = NaN")
        # println(io, "uIHT time = $uIHT_time seconds, pleiotropic power = $uIHT_pleiotropic_power, independent power = $uIHT_independent_power, FP = $uIHT_FP, FDR = $uIHT_FDR, λ = NaN")
        # println(io, "mvPLINK time = $mvplink_time seconds, pleiotropic power = $mvPLINK_pleitropic_power, independent power = $mvPLINK_independent_power, FP = $mvPLINK_FP, FDR = $mvPLINK_FDR, λ = $mvPLINK_λ")
        # println(io, "GEMMA time = $gemma_time seconds, pleiotropic power = $gemma_pleiotropic_power, independent power = $gemma_independent_power, FP = $gemma_FP, FDR = $gemma_FDR, λ = $gemma_λ")
    end
    cd("../")

    return nothing
end

function run_simulation(set::Int, model::Symbol)
    σg = 0.1
    σe = 0.9
    fdr = 0.1
    init_beta = true
    debias = false
    βoverlap = [3, 5, 7]
    k = [10, 20, 100]
    r = [2, 3, 3]
    path = set ≥ 3 ? (10:10:200) : (5:5:50)

    println("Simulation model = $model, set $set has k = $(k[set]), r = $(r[set]), βoverlap = $(βoverlap[set])")

    cur_dir = pwd() * "/set$set"
    isdir(cur_dir) || mkdir(cur_dir)
    k_cur = k[set]
    r_cur = r[set]
    βoverlap_cur = βoverlap[set]

    for seed in 1:100
        try
            cd(cur_dir)
            one_simulation(k_cur, r_cur, seed = seed, path = path, βoverlap=βoverlap_cur, 
                σg=σg, σe=σe, init_beta=init_beta, model=model, debias=debias)
        catch e
            bt = catch_backtrace()
            msg = sprint(showerror, e, bt)
            println("set $set sim $seed threw an error!")
            println(msg)
            continue
        end
    end
end

# set = parse(Int, ARGS[1])
# model = :polygenic
# run_simulation(set, model)





# set = 1
# model = :polygenic
# σg = 0.1
# σe = 0.9
# fdr=0.1
# init_beta = true
# debias = false
# path = 5:5:50
# βoverlap = 3
# k = 10
# r = 2

# cur_dir = pwd() * "/set$set"
# isdir(cur_dir) || mkdir(cur_dir)
# seed = 1
# cd(cur_dir)

