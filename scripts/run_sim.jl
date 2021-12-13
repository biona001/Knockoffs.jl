using Revise
using SnpArrays
using Knockoffs
using DelimitedFiles
using Random
using LinearAlgebra
using Distributions
using ProgressMeter
using MendelIHT
using VCFTools
using StatsBase
using CSV
using DataFrames
using Printf
using GLMNet
using GLM
BLAS.set_num_threads(1)

global qctools_exe = "/scratch/users/bbchu/qctool/build/release/qctool_v2.0.7"
global snpknock2_exe = "/scratch/users/bbchu/knockoffgwas/snpknock2/bin/snpknock2"
global rapid_exe = "/scratch/users/bbchu/RaPID/RaPID_v.1.7"
global partition_exe = "/scratch/users/bbchu/knockoffgwas/knockoffgwas/utils/partition.R";
global propca_exe = "/scratch/users/bbchu/ProPCA/build/propca"
global bolt_exe = "/scratch/users/bbchu/BOLT-LMM_v2.2/bolt"

"""
    simulate_pop_structure(n, p, diff_markers, populations)

Simulate genotypes with K = 2 populations. 1% of SNPs will have different allele 
frequencies between the populations.

# Inputs
- `n`: Number of samples
- `p`: Number of SNPs
- `diff_markers`: Indices of the differentially expressed alleles.
- `populations`: Vector of length `n` indicating population membership for each sample. 
    Each entry should be 1 or 2 (total = 2 populations)

# Output
- `x1`: n×p matrix of the 1st haplotype for each sample. Each row is a haplotype
- `x2`: n×p matrix of the 2nd haplotype for each sample. `x = x1 + x2`

# Reference
https://www.nature.com/articles/nrg2813
"""
function simulate_pop_structure(n::Int, p::Int, diff_markers, populations)
    # first simulate genotypes treating all samples equally
    x1 = BitMatrix(undef, n, p)
    x2 = BitMatrix(undef, n, p)
    pmeter = Progress(p, 0.1, "Simulating genotypes...")
    @inbounds for j in 1:p
        d = Bernoulli(rand())
        for i in 1:n
            x1[i, j] = rand(d)
            x2[i, j] = rand(d)
        end
        next!(pmeter)
    end
    # simulate unually differentiated markers for differernt populations
    @inbounds for j in diff_markers
        allele_freq_diff = 0.25rand()
        allele_freq_a = allele_freq_diff * rand()
        allele_freq_b = allele_freq_a + allele_freq_diff
        pop1_dist = Bernoulli(allele_freq_a)
        pop2_dist = Bernoulli(allele_freq_b)
        for i in 1:n
            d = isone(populations[i]) ? pop1_dist : pop2_dist
            x1[i, j] = rand(d)
            x2[i, j] = rand(d)
        end
    end
    return x1, x2, populations
end

"""
    simulate_IBD(h1::BitMatrix, h2::BitMatrix, populations::Vector{Int}, k::Int)

Simulate recombination events. Parent haplotypes `h1` and `h2` will be used to generate 
`k` children, then both parent and children haplotypes will be returned. 

In offspring simulation, we first randomly sample 2 parents from the same population. 
Then generate offspring individuals by copying segments of the parents haplotype
directly to the offspring to represent IBD segments. The number of segments (i.e. places of
recombination) is 1 to 5 per sample chosen uniformly across all SNPs. 

# Inputs
- `h1`: `n × p` matrix of the 1st haplotype for each parent. Each row is a haplotype
- `h2`: `n × p` matrix of the 2nd haplotype for each parent. `H = h1 + h2`
- `populations`: `populations[i]` is the population (represented as integer) of sample `i`. 
- `k`: Total number of offsprings

# Output
- `H1`: `n+k × p` matrix of the 1st haplotype. The first `n` haplotypes are from parents
    and the next `k` haplotypes are the offsprings. Each row is a haplotype
- `H2`: `n+k × p` matrix of the 2nd haplotype. `x = x1 + x2`

# References
https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1003520
"""
function simulate_IBD(h1::BitMatrix, h2::BitMatrix, populations::Vector{Int}, k::Int)
    n, p = size(h1)
    unique_populations = unique(populations)
    # randomly designate gender for parents
    sex = bitrand(n)
    male_idx = findall(x -> x == true, sex)
    female_idx = findall(x -> x == false, sex)
    # simulate new samples
    x1 = falses(k, p)
    x2 = falses(k, p)
    fathers = Int[]
    mothers = Int[]
    pmeter = Progress(k, 0.1, "Simulating IBD segments...")
    for i in 1:k
        # assign parents (mom has to be from same population as dad)
        dad = rand(male_idx)
        mom = 0
        while true
            mom = rand(female_idx)
            populations[mom] == populations[dad] && break
        end
        push!(fathers, dad)
        push!(mothers, mom)
        # recombination
        recombine!(@view(x1[i, :]), @view(x2[i, :]), @view(h1[dad, :]),
                   @view(h2[dad, :]), @view(h1[mom, :]), @view(h2[mom, :]))
        # update progress
        next!(pmeter)
    end
    return x1, x2, fathers, mothers
end

function recombination_segments(breakpoints::Vector{Int}, snps::Int)
    start = 1
    result = UnitRange{Int}[]
    for bkpt in breakpoints
        push!(result, start:bkpt)
        start = bkpt + 1
    end
    push!(result, breakpoints[end]+1:snps)
    return result
end

function recombine!(child_h1, child_h2, dad_h1, dad_h2, mom_h1, mom_h2)
    p = length(child_h1)
    recombinations = rand(1:5)
    breakpoints = sort!(sample(1:p, recombinations, replace=false))
    segments = recombination_segments(breakpoints, p)
    for segment in segments
        dad_hap = rand() < 0.5 ? dad_h1 : dad_h2
        mom_hap = rand() < 0.5 ? mom_h1 : mom_h2
        copyto!(@view(child_h1[segment]), @view(dad_hap[segment]))
        copyto!(@view(child_h2[segment]), @view(mom_hap[segment]))
    end
end

function write_plink(outfile::AbstractString, x1::AbstractMatrix, x2::AbstractMatrix)
    n, p = size(x1)
    x = SnpArray(outfile * ".bed", n, p)
    for j in 1:p, i in 1:n
        c = x1[i, j] + x2[i, j]
        if c == 0
            x[i, j] = 0x00
        elseif c == 1
            x[i, j] = 0x02
        elseif c == 2
            x[i, j] = 0x03
        else
            error("matrix entries should be 0, 1, or 2 but was $c!")
        end
    end
    # create .bim file structure: https://www.cog-genomics.org/plink2/formats#bim
    open(outfile * ".bim", "w") do f
        for i in 1:p
            println(f, "1\tsnp$i\t0\t$(100i)\t1\t2")
        end
    end
    # create .fam file structure: https://www.cog-genomics.org/plink2/formats#fam
    open(outfile * ".fam", "w") do f
        for i in 1:n
            println(f, "$i\t1\t0\t0\t1\t-9")
        end
    end
    return nothing
end

function make_partition_mapfile(filename, p::Int)
    map_cM = LinRange(0.0, Int(p / 10000), p)
    open(filename, "w") do io
        println(io, "Chromosome\tPosition(bp)\tRate(cM/Mb)\tMap(cM)")
        for i in 1:p
            println(io, "chr1\t", 100i, '\t', 0.01rand(), '\t', map_cM[i])
        end
    end
end

function make_rapid_mapfile(filename, p::Int)
    map_cM = LinRange(0.0, Int(p / 10000), p)
    open(filename, "w") do io
        for i in 1:p
            println(io, i, '\t', map_cM[i])
        end
    end
end

function process_rapid_output(inputfile, outputfile)
    writer = open(outputfile, "w")
    println(writer, "CHR ID1 HID1 ID2 HID2 BP.start BP.end site.start site.end cM FAM1 FAM2")
    try
        df = readdlm(inputfile)
        for r in eachrow(df)
            chr, id1, id2, hap1, hap2, start_pos, end_pos, genetic_len, start_site, end_site = 
                Int(r[1]), Int(r[2]), Int(r[3]), Int(r[4]), Int(r[5]), Int(r[6]), Int(r[7]),
                r[8], Int(r[9]), Int(r[10])
            println(writer, chr, ' ', id1, ' ', hap1, ' ', id2, ' ', hap2, ' ', 
                start_pos, ' ', end_pos, ' ', start_site, ' ', end_site, ' ', 
                genetic_len, ' ', 1, ' ', 1)
        end
    catch
        println("0 IBD segments detected!")
    end
    close(writer)
end

function make_bgen_samplefile(filename, n)
    open(filename, "w") do io
        println(io, "ID_1 ID_2 missing sex")
        println(io, "0 0 0 D")
        for i in 1:n
            println(io, "$i 1 0 1")
        end
    end 
end

# will simulate ntrain training samples and ntest testing samples
function simulate_genotypes(ntrain, ntest, p, offsprings, ndiff_markers, seed)
    # simulate parent haplotypes
    n_total = ntrain + ntest
    Random.seed!(seed)
    diff_markers = sample(1:p, ndiff_markers, replace=false)
    populations = rand(1:2, n_total)
    h1, h2 = simulate_pop_structure(n_total, p, diff_markers, populations)

    # simulate random mating
    x1, x2 = simulate_IBD(h1, h2, populations, offsprings)
    
    # write training data (with cryptic relatedness)
    H1_train = [h1[1:ntrain, :]; x1]
    H2_train = [h2[1:ntrain, :]; x2]
    write_vcf("train.phased.vcf.gz", H1_train, H2_train)
    write_plink("train", H1_train, H2_train)

    # write testing data (no cryptic relatedness)
    H1_test = h1[ntrain+1:end, :]
    H2_test = h2[ntrain+1:end, :]
    write_plink("test", H1_test, H2_test)

    # save pop1/pop2 index and unually differentiated marker indices
    writedlm("populations.txt", populations)
    writedlm("diff_markers.txt", diff_markers)
    
    # generate fake map file
    make_partition_mapfile("sim.partition.map", p)

    # also generate QC file that contains all SNPs and all samples
    snpdata = SnpData("train")
    snpIDs = snpdata.snp_info[!, :snpid]
    sampleIDs = Matrix(snpdata.person_info[!, 1:2])
    writedlm("variants_qc.txt", snpIDs)
    writedlm("samples_qc.txt", sampleIDs)
end

function make_knockoffs(n, p, offsprings, seed)
    # fake map file for rapid
    make_rapid_mapfile("sim.rapid.map", p)

    # run rapid
    Random.seed!(seed)
    vcffile = "train.phased.vcf.gz"
    mapfile = "sim.rapid.map"
    outfolder = "rapid"
    d = 3    # minimum IBD length in cM
    w = 3    # number of SNPs per window
    r = 10   # number of runs
    s = 2    # Minimum number of successes to consider a hit
    @time rapid(rapid_exe, vcffile, mapfile, d, outfolder, w, r, s)

    # unzip and postprocess rapid output to suit snpknock2
    run(pipeline(`gunzip -c ./rapid/results.max.gz`, stdout="./rapid/results.max"))
    process_rapid_output("./rapid/results.max", "sim.snpknock.ibdmap")
    
    # convert VCF to BGEN format
    outfile = "train.bgen"
    run(`$qctools_exe -g $vcffile -og $outfile`)
    make_bgen_samplefile("train.sample", n + offsprings)
    
    # snpknock2 arguments
    bgenfile = "train"
    sample_qc = "samples_qc.txt"
    variant_qc = "variants_qc.txt"
    mapfile = "sim.partition.map"
    partfile = "sim.partition.txt"
    ibdfile = "sim.snpknock.ibdmap"
    K = 10
    cluster_size_min = 1000 
    cluster_size_max = 10000 
    hmm_rho = 1
    hmm_lambda = 1e-3 
    windows = 0
    n_threads = Threads.nthreads()
    compute_references = true
    generate_knockoffs = true
    outfile = "train.knockoffs"
    @time snpknock2(snpknock2_exe, bgenfile, sample_qc, variant_qc, mapfile, partfile, ibdfile, 
        K, cluster_size_min, cluster_size_max, hmm_rho, hmm_lambda, windows, n_threads, 
        seed, compute_references, generate_knockoffs, outfile)
end

"""
    simulate_beta_and_y!(xtrain, xtest, diff_markers, ::Normal)

Simulates β and y given indices of differentiated markers and random seem.
There are 100 causal snps, 50 on normally differentiated SNPs and 50 on
abnormally differentiated snps. If `add_interaction=true`, there will be 15  
interaction terms as summarized above.
"""
function simulate_beta_and_y!(
    xtrain::AbstractMatrix,
    xtest::AbstractMatrix,
    diff_markers::Vector{Int},
    ::Normal
    )
    n, p = size(xtrain)
    ntest = size(xtest, 1)
    h2 = 0.5 # heritability
    K = 100 # number of causal snps
    d = Normal(0, sqrt(h2 / 2K)) # from paper: Efficient Implementation of Penalized Regression for Genetic Risk Prediction

    # simulate β with linear effect, half are on normally differentiated SNPs half on abnormal SNPs
    β = zeros(p)
    norm_markers = setdiff(1:p, diff_markers)
    norm_causal_snps = sample(norm_markers, K >> 1, replace=false)
    β[norm_causal_snps] .= rand(d, K >> 1)
    if length(diff_markers) > 0
        shuffle!(diff_markers)
        diff_causal_markers = diff_markers[1:K >> 1]
        β[diff_causal_markers] .= rand(d, K >> 1)
    end

    # simulate y (train and test)
    ϵ = Normal(0, 1 - h2)
    ytrain = xtrain * β + rand(ϵ, n)
    ytest = xtest * β + rand(ϵ, ntest)
    
    return ytrain, ytest, β, norm_markers, norm_causal_snps, diff_markers
end

function simulate_beta_and_y!(
    xtrain::AbstractMatrix,
    xtest::AbstractMatrix,
    diff_markers::Vector{Int},
    ::Bernoulli
    )
    n, p = size(xtrain)
    ntest = size(xtest, 1)
    h2 = 0.5 # heritability
    K = 100 # number of causal snps
    d = Normal(0, 0.3) 

    # simulate β with linear effect, half are on normally differentiated SNPs half on abnormal SNPs
    β = zeros(p)
    norm_markers = setdiff(1:p, diff_markers)
    norm_causal_snps = sample(norm_markers, K >> 1, replace=false)
    β[norm_causal_snps] .= rand(d, K >> 1)
    if length(diff_markers) > 0
        shuffle!(diff_markers)
        diff_causal_markers = diff_markers[1:K >> 1]
        β[diff_causal_markers] .= rand(d, K >> 1)
    end

    # simulate y (train and test)
    ytrain_mean = GLM.linkinv.(LogitLink(), xtrain * β)
    ytest_mean = GLM.linkinv.(LogitLink(), xtest * β)
    ytrain = Float64.([rand(Bernoulli(i)) for i in ytrain_mean])
    ytest = Float64.([rand(Bernoulli(i)) for i in ytest_mean])

    return ytrain, ytest, β, norm_markers, norm_causal_snps, diff_markers
end

# function simulate_beta_and_y_with_interaction(xtrain::AbstractMatrix, xtest::AbstractMatrix, diff_markers::Vector{Int})
#     n, p = size(xtrain)
#     ntest = size(xtest, 1)
#     h2 = 0.5 # heritability
#     K = 100 # number of causal snps
#     d = Normal(0, sqrt(h2 / (2K))) # from paper: Efficient Implementation of Penalized Regression for Genetic Risk Prediction
    
#     # simulate β with linear effect, half are on normally differentiated SNPs half on abnormal SNPs
#     β = zeros(p)
#     norm_markers = setdiff(1:p, diff_markers)
#     norm_causal_snps = sample(norm_markers, K >> 1, replace=false)
#     β[norm_causal_snps] .= rand(d, K >> 1)
#     shuffle!(diff_markers)
#     diff_causal_markers = diff_markers[1:K >> 1]
#     β[diff_causal_markers] .= rand(d, K >> 1)
    
#     # make β with interacting terms
#     diff_non_causal_markers = diff_markers[K+1:end]
#     norm_non_causal_markers = setdiff(norm_markers, norm_causal_snps)
#     shuffle!(norm_non_causal_markers)
#     interact_pairs = Tuple{Int, Int}[]
#     for i in 1:5
#         k1, l1 = pop!(norm_non_causal_markers), pop!(norm_non_causal_markers)
#         k2, l2 = pop!(norm_non_causal_markers), pop!(diff_non_causal_markers)
#         k3, l3 = pop!(diff_non_causal_markers), pop!(diff_non_causal_markers)
#         push!(interact_pairs, (k1, l1)) # norm-norm
#         push!(interact_pairs, (k2, l2)) # norm-diff
#         push!(interact_pairs, (k3, l3)) # diff-diff
#     end
        
#     # form design matrix with interaction terms
#     J = length(interact_pairs) # number of interacting SNP pairs
#     ztrain = zeros(n, J)
#     ztest = zeros(ntest, J)
#     for j in 1:J
#         k, l = interact_pairs[j]
#         for i in 1:n
#             ztrain[i, j] = xtrain[i, k] * xtrain[i, l]
#         end
#         for i in 1:ntest
#             ztest[i, j] = xtest[i, k] * xtest[i, l]
#         end
#     end
    
#     # simulate y (train and test)
#     ϵ = Normal(0, 1 - h2)
#     γ = rand(Normal(0, sqrt(h2 / (2J))), J)
#     ytrain = xtrain * β + ztrain * γ + rand(ϵ, n)
#     ytest = xtest * β + ztest * γ + rand(ϵ, ntest)
    
#     return ytrain, ytest, β, norm_markers, norm_causal_snps, diff_markers, diff_causal_markers, γ, interact_pairs
# end

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

function tune_k(y::AbstractVector, xko_la::AbstractMatrix, covariates::AbstractVecOrMat,
    original::Vector{Int}, knockoff::Vector{Int}, groups::Vector{Int}, fdr::Float64, best_k::Int,
    dist::Distribution)
    # do a grid search for best sparsity level
    best_β = Float64[]
    best_err = Inf
    init_beta = dist == Normal() ? true : false
    l = canonicallink(dist)
    for cur_k in best_k:5:round(Int, 1.5best_k)
        result = fit_iht(y, xko_la, covariates, k=cur_k, d=dist, l=l, 
            init_beta=init_beta, max_iter=500)
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

function read_bolt_result(outfile)
    df = CSV.read(outfile, DataFrame)
    pvals = df[!, :P_BOLT_LMM_INF]
    β = df[!, :BETA]
    return β, pvals
end

function get_diff_markers(data_dir)
    try
        return Vector{Int}(vec(readdlm(data_dir)))
    catch
        return Int[]
    end
end

function one_simulation(n, p, dist, offsprings, seed; 
    ndiff_markers = Int(0.01p), # number of differentially expressed snps
    use_PCA=true,               # whether to compute, store, and use PCA in model selection
    simulate_interactions=true  # whether to add interacting SNPs in βtrue
    ) 
    cur_dir = pwd() * "/"

    # save all simulated data under data/sim$seed
    isdir("data") || mkdir("data")
    isdir("data/sim$seed") || mkdir("data/sim$seed")
    cd("data/sim$seed")
    data_dir = cur_dir * "data/sim$seed/"
    
    # simulate phased genotypes with pop structure and cryptic relatedness
    #ntest = Int(0.2n)
    #simulate_genotypes(n, ntest, p, offsprings, ndiff_markers, seed)
    
    # partition genotypes to groups (for making group knockoffs)
    # partition(partition_exe, "train", "sim.partition.map", "variants_qc.txt", "sim.partition.txt")
    
    # make knockoffs
    # make_knockoffs(n, p, offsprings, seed)
    
    # compute PCA on original genotypes using ProPCA
    if use_PCA
        plinkfile = data_dir * "train"
        outfile = data_dir * "pca_"
        run(`$propca_exe -g $plinkfile -o $outfile -nt $(Threads.nthreads()) -seed $seed`)
        z = readdlm(data_dir * "pca_projections.txt")
        standardize!(z)
    end
    
    # loop over group resolution
    for group in 0:1
        cd(data_dir)

        # import genotypes, knockoff, and other info
        snpdata = SnpData(data_dir * "knockoffs/train.knockoffs_res$(group)")
        snpid = snpdata.snp_info.snpid
        knockoff_idx = endswith.(snpid, ".k")
        original = findall(knockoff_idx .== false)
        knockoff = findall(knockoff_idx)
        x = snpdata.snparray
        xla = convert(Matrix{Float64}, @view(x[:, original]), center=true, scale=true, impute=true)
        xla_test = convert(Matrix{Float64}, SnpArray(data_dir * "test.bed"), center=true, scale=true, impute=true)
        xko_la = convert(Matrix{Float64}, x, center=true, scale=true, impute=true)
        groups = repeat(Vector{Int}(readdlm(data_dir * "knockoffs/train.knockoffs_res$(group)_grp.txt", 
            header=true)[1][:, 2]), inner=2) .+ 1
        diff_markers = get_diff_markers(data_dir * "diff_markers.txt")

        # If requested, concatenate PCs to genotype matrix (only needed for lasso)
        xla_full = use_PCA ? [xla z] : xla
        xko_la_full = use_PCA ? [xko_la z] : xko_la
        if use_PCA # must also asign groups to PCs
            max_g = maximum(groups)
            PC_groups = collect(max_g+1:max_g+5)
        end
        groups_full = use_PCA ? [groups; PC_groups] : groups
        GC.gc()

        #
        # simulate beta and y, form non-genetic covariates (used in IHT only)
        #
        Random.seed!(seed)
        if simulate_interactions
            ytrain, ytest, β, norm_markers, norm_causal_snps, diff_markers,
                γ, interact_pairs =
                simulate_beta_and_y_with_interaction(xla, xla_test, diff_markers)
        else
            ytrain, ytest, β, norm_markers, norm_causal_snps, diff_markers = 
                simulate_beta_and_y!(xla, xla_test, diff_markers, dist)
        end
        covar = use_PCA ? [ones(size(xla, 1)) z] : ones(size(xla, 1))
        link = canonicallink(dist) 
        init_beta = dist == Normal() ? true : false

        #
        # Run standard IHT
        #
        Random.seed!(seed)
        path = 10:10:250
        mses = cv_iht(ytrain, xla, covar, d=dist, l=link, path=path, init_beta=init_beta, max_iter=500)
        GC.gc()
        Random.seed!(seed)
        k_rough_guess = path[argmin(mses)]
        dense_path = (k_rough_guess - 9):(k_rough_guess + 9)
        mses_new = cv_iht(ytrain, xla, covar, d=dist, l=link, path=dense_path, init_beta=init_beta, max_iter=500)
        GC.gc()
        Random.seed!(seed)
        result = fit_iht(ytrain, xla, covar, k=dense_path[argmin(mses_new)], d=dist, l=link, 
            init_beta=init_beta, max_iter=500)
        β_iht = result.beta
        GC.gc()

        #
        # run knockoff IHT 
        #
        Random.seed!(seed)
        path = 10:10:250
        mses = cv_iht(ytrain, xko_la, covar, d=dist, l=link, path=path, init_beta=init_beta, max_iter=500)
        GC.gc()
        Random.seed!(seed)
        k_rough_guess = path[argmin(mses)]
        dense_path = (k_rough_guess - 9):(k_rough_guess + 9)
        mses_new = cv_iht(ytrain, xko_la, covar, d=dist, l=link, path=dense_path, init_beta=init_beta, max_iter=500)
        GC.gc()
        Random.seed!(seed)
        ko_iht_result = fit_iht(ytrain, xko_la, covar, d=dist, l=link, 
            k=dense_path[argmin(mses_new)], init_beta=init_beta, max_iter=500)

        #
        # Run standard lasso
        #
        Random.seed!(seed)
        lasso_y = dist == Normal() ? ytrain : string.(ytrain)
        cv = glmnetcv(xla_full, lasso_y, nfolds=5, parallel=true) 
        β_lasso = GLMNet.coef(cv)
        GC.gc()

        #
        # Run knockoff lasso
        #
        Random.seed!(seed)
        ko_lasso_cv = glmnetcv(xko_la_full, lasso_y, nfolds=5, parallel=true)
        GC.gc()

        #
        # Run bolt lmm (need to make covariate and phenotype file first)
        #
        bedfile = data_dir * "train"
        covfile = "cov_bolt.txt"
        phefile = "y_bolt.txt"
        outfile = "bolt_output.txt"
        open(phefile, "w") do io
            println(io, "FID IID trait1") 
            for i in 1:n
                println(io, "$i 1 $(ytrain[i])") 
            end
        end
        if use_PCA
            open(covfile, "w") do io
                println(io, "FID IID PC1 PC2 PC3 PC4 PC5") 
                for i in 1:n
                    println(io, "$i 1 ", z[i, 1], ' ', z[i, 2], ' ',  z[i, 3], ' ', z[i, 4], ' ', z[i, 5]) 
                end
            end
            run(`$bolt_exe --bfile=$bedfile --covarFile=$covfile 
                --phenoFile=$phefile --phenoCol=trait1 
                --qCovarCol=PC\{1:5\} --lmmInfOnly --numLeaveOutChunks=2 --statsFile $outfile`)
        else
            run(`$bolt_exe --bfile=$bedfile --phenoFile=$phefile --phenoCol=trait1 
                --lmmInfOnly --numLeaveOutChunks=2 --statsFile $outfile`)
        end

        # current directory
        group_dir = cur_dir * "res$group/"
        isdir(group_dir) || mkdir(group_dir)
        cd(group_dir)
        GC.gc()

        for fdr in [0.05, 0.1, 0.25, 0.5]
            top_dir = group_dir * "fdr$fdr/"
            new_dir = group_dir * "fdr$fdr/sim$seed/"
            isdir(top_dir) || mkdir(top_dir)
            isdir(new_dir) || mkdir(new_dir)
            cd(new_dir)

            #
            # save true (simulated) parameters
            #
            writedlm("y_train.txt", ytrain)
            writedlm("y_test.txt", ytest)
            writedlm("beta_true.txt", β)
            writedlm("normal_markers.txt", norm_markers)
            writedlm("normal_causal_markers.txt", norm_causal_snps)
            writedlm("diff_markers.txt", diff_markers)
            simulate_interactions && writedlm("gamma_true.txt", γ)
            simulate_interactions && writedlm("interacting_markers.txt", interact_pairs)

            #
            # save standard IHT/lasso results
            #
            writedlm("iht.beta", β_iht)
            writedlm("lasso.beta", β_lasso)

            # 
            # knockoff IHT statistics
            #
            β_iht_knockoff = extract_beta(ko_iht_result.beta, fdr, groups, original, knockoff)
            writedlm("iht.knockoff.beta", ko_iht_result.beta)
            writedlm("iht.knockoff.postfilter.beta", β_iht_knockoff)
            GC.gc()

            #
            # run knockoff IHT with wrapped cross validation
            #
            # Random.seed!(seed)
            # path = 10:10:250
            # mses = cv_iht_knockoff(ytrain, xko_la, covar, original,
            #     knockoff, fdr, d=dist, l=link, path=path, init_beta=init_beta, group_ko=groups, max_iter=500)
            # GC.gc()
            # Random.seed!(seed)
            # k_rough_guess = path[argmin(mses)]
            # dense_path = (k_rough_guess - 9):(k_rough_guess + 9)
            # mses_new = cv_iht_knockoff(ytrain, xko_la, covar, original, knockoff, fdr,
            #     d=dist, l=link, path=dense_path, init_beta=init_beta, group_ko=groups, max_iter=500)
            # GC.gc()
            # # adjust sparsity level so it best matches sparsity chosen by ko filter
            # Random.seed!(seed)
            # best_k = dense_path[argmin(mses_new)]
            # best_β = tune_k(ytrain, xko_la, covar, original, knockoff, groups, fdr, best_k, dist)
            # β_iht_knockoff_cv = extract_beta(best_β, fdr, groups, original, knockoff)
            # writedlm("iht.knockoff.cv.beta", best_β)
            # writedlm("iht.knockoff.cv.postfilter.beta", β_iht_knockoff_cv)
            # GC.gc()

            #
            # knockoff LASSO statistics
            #
            β_lasso_knockoff = extract_beta(GLMNet.coef(ko_lasso_cv), fdr, groups_full, original, knockoff)
            writedlm("lasso.knockoff.beta", GLMNet.coef(ko_lasso_cv))
            writedlm("lasso.knockoff.postfilter.beta", β_lasso_knockoff)
            GC.gc()

            #
            # copy bolt output
            #
            use_PCA && cp(data_dir * covfile, new_dir * covfile, force=true)
            cp(data_dir * phefile, new_dir * phefile, force=true)
            cp(data_dir * outfile, new_dir * outfile, force=true)
        end
        use_PCA && rm(data_dir * covfile, force=true)
        rm(data_dir * phefile, force=true)
        rm(data_dir * outfile, force=true)
        GC.gc()
    end
end

n = 5000
p = 50000
offsprings = 0      # this gives cryptic relatedness
ndiff_markers = 0 # this gives population structure
d = Normal()
seed = parse(Int, ARGS[1])
one_simulation(n, p, d, offsprings, seed, use_PCA=false, 
    ndiff_markers=ndiff_markers, simulate_interactions=false)

# allele freq diff = uniform(0, 0.5)
# simulate_genotypes(5000, 500, p, offsprings, 2022)
# x = SnpArray("train.bed")
# diff_markers = Int.(sort!(vec(readdlm("diff_markers.txt"))))
# xdiff = convert(Matrix{Float64}, @view(x[:, diff_markers]), center=true, scale=true)
# cor(xdiff)
# minimum(cor(xdiff)) # -0.03744612543696474
# maximum(cor(xdiff) - Diagonal(ones(500))) # 0.4935536908842222


# allele freq diff = uniform(0, 0.25)
# ndiff_markers = 500
# simulate_genotypes(5000, 500, p, offsprings, ndiff_markers, 2022)
# x = SnpArray("train.bed")
# diff_markers = Int.(sort!(vec(readdlm("diff_markers.txt"))))
# xdiff = convert(Matrix{Float64}, @view(x[:, diff_markers]), center=true, scale=true)
# cor(xdiff)
# minimum(cor(xdiff)) # -0.04200044938284282
# maximum(cor(xdiff) - Diagonal(ones(500))) # 0.24972376563342386
