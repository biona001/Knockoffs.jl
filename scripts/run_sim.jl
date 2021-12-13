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
using CodecZlib
BLAS.set_num_threads(1)

global qctools_exe = "/scratch/users/bbchu/qctool/build/release/qctool_v2.0.7"
global snpknock2_exe = "/scratch/users/bbchu/knockoffgwas/snpknock2/bin/snpknock2"
global rapid_exe = "/scratch/users/bbchu/RaPID/RaPID_v.1.7"
global partition_exe = "/scratch/users/bbchu/knockoffgwas/knockoffgwas/utils/partition.R" # modify this script to generate knockoffs of different group resolution
global propca_exe = "/scratch/users/bbchu/ProPCA/build/propca"
global bolt_exe = "/scratch/users/bbchu/BOLT-LMM_v2.2/bolt"
global hdf_to_vcf_exe = "/scratch/users/bbchu/scripts/hdf_to_vcf.py"
global prs_genotypes_exe = "/scratch/users/bbchu/PRS_Admixture_Simulation/simulate_genotypes.py"
global rapid_filter_exe = "/scratch/users/bbchu/scripts/filter_mapping_file.py"
global rapid_interpolate_exe = "/scratch/users/bbchu/scripts/interpolate_loci.py"

function create_plink(
    outfile::AbstractString,
    x1::AbstractMatrix,
    x2::AbstractMatrix;
    chr::Vector{String} = ["1" for _ in 1:size(x1, 2)],
    pos::Vector{Int} = collect(100:100:size(x1, 2))
    )
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
            println(f, "$(chr[i])\tsnp$i\t0\t$(pos[i])\t1\t2")
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

# todo: How to systematically reduce connected components in related graph?
# see: https://github.com/msesia/knockoffgwas/issues/3
function process_rapid_output(inputfile, outputfile)
    writer = open(outputfile, "w")
    println(writer, "CHR ID1 HID1 ID2 HID2 BP.start BP.end site.start site.end cM FAM1 FAM2")
    try
        df = readdlm(inputfile)
        total_ibd = size(df, 1)
        keep_prob = 1000 / total_ibd # want to keep on average 1000 IBD segments
        for r in eachrow(df)
            chr, id1, id2, hap1, hap2, start_pos, end_pos, genetic_len, start_site, end_site = 
                Int(r[1]), Int(r[2]), Int(r[3]), Int(r[4]), Int(r[5]), Int(r[6]), Int(r[7]),
                r[8], Int(r[9]), Int(r[10])
            rand() < keep_prob && println(writer, chr, ' ', id1, ' ', hap1, ' ', id2, ' ', hap2, ' ', 
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

# compute most ancestry informative markers
function find_500_AIMs(HCEU, HYRI)
    # we don't know which samples in all.vcf belongs to which population,
    # so create a VCF file for which we do know
    H1 = [HCEU[1:2:end, :]; HYRI[1:2:end, :]]
    H2 = [HCEU[2:2:end, :]; HYRI[2:2:end, :]]
    sample_ids = [string(i) for i in 1:size(H1, 1)]
    write_vcf("nonadmixed.vcf", H1, H2, sampleID=sample_ids)

    # create dictionary mapping sampleID to population
    nCEU = size(HCEU, 1) >> 1
    nYRI = size(HYRI, 1) >> 1
    sampleID_to_population = Dict{String, String}()
    [sampleID_to_population[string(i)] = "CEU" for i in 1:nCEU]
    [sampleID_to_population[string(i)] = "YRI" for i in (nCEU+1):(nCEU + nYRI)]

    # select aims
    pvals = VCFTools.aim_select("nonadmixed.vcf", sampleID_to_population)
    if count(iszero, pvals) ≥ 500
        idx = findall(iszero, pvals)
        shuffle!(idx)
        top_aims = idx[1:500]
    else
        top_aims = sortperm(pvals)[1:500]
    end

    # clean up
    rm("nonadmixed.vcf", force=true)

    return top_aims
end

# the header row of rfmix output is #CHROM  POS     ID      REF     VAL
# it should be #CHROM  POS     ID      REF     ALT ...
function process_rfmix(samples, intputfile, outfile)
    reader = GzipDecompressorStream(open(intputfile))
    writer = GzipCompressorStream(open(outfile, "w"))
    # skip broken header lines
    for i in 1:5
        readline(reader)
    end
    # write minimum header into
    print(writer, "##fileformat=VCFv4.2\n")
    print(writer, "##source=rfmix\n")
    print(writer, "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")
    print(writer, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")
    for id in ["sample$i" for i in 1:samples]
        print(writer, "\t", id)
    end
    print(writer, "\n")
    # write line by line
    while !eof(reader)
        println(writer, readline(reader))
    end
    close(reader)
    close(writer)
end

function simulate_genotypes(nYRI, nCEU, nmate, nadmixed, p, seed, outdir)
    # change directory to PRS_Admixture_Simulation
    # will copy data back to outdir at the end of this function
    cd("PRS_Admixture_Simulation")

    # simulate CEU and YRI populations and their mating, then convert to VCF format
    @time run(`python $prs_genotypes_exe --sim $seed --nYRI $nYRI --nCEU $nCEU --nmate $nmate --nadmixed $nadmixed --threads $(Threads.nthreads())`)
    cd("output/sim$seed/trees")
    run(pipeline(`python $hdf_to_vcf_exe tree_all.hdf`, stdout="all.full.vcf")) # 10000 samples
    run(pipeline(`python $hdf_to_vcf_exe tree_CEU_GWAS_nofilt.hdf`, stdout="CEU.full.vcf"))
    run(pipeline(`python $hdf_to_vcf_exe tree_YRI_GWAS_nofilt.hdf`, stdout="YRI.full.vcf"))
    # run(pipeline(`python $hdf_to_vcf_exe tree_mate.hdf`, stdout="mate.full.vcf")) # mate.full.vcf are CEU/YRI samples used to generate admixed samples, and then excluded from CEU/YRI.full.vcf files

    # admixed samples are made by rfmix, but their vcf format is old, so we need to process it
    process_rfmix(nadmixed, "../admixed_data/output/admix_afr_amer.query.vcf.gz", "admixed.full.vcf.gz")

    # keep p SNPs with maf at least 0.01 (todo: keep how many snps??)
    _, _, _, _, _, maf_by_record, _ = gtstats("all.full.vcf")
    idx = findall(x -> x > 0.01, maf_by_record)
    shuffle!(idx)
    record_mask = sort(idx[1:p])
    # maf_perm = sortperm(maf_by_record, rev=true)
    # record_mask = maf_perm[1:p]
    VCFTools.filter("CEU.full.vcf", record_mask, 1:VCFTools.nsamples("CEU.full.vcf"), des="CEU.vcf")
    VCFTools.filter("YRI.full.vcf", record_mask, 1:VCFTools.nsamples("YRI.full.vcf"), des="YRI.vcf")
    VCFTools.filter("admixed.full.vcf.gz", record_mask, 1:VCFTools.nsamples("admixed.full.vcf.gz"), des="admixed.vcf")

    # import haplotypes
    HCEU = convert_ht(Bool, "CEU.vcf", msg="importing CEU.vcf")
    HYRI = convert_ht(Bool, "YRI.vcf", msg="importing YRI.vcf")
    Hadmix,_,_,pos,_,_,_ = convert_ht(Bool, "admixed.vcf", save_snp_info=true, msg="importing admixed.vcf")

    # write 90% samples to training data
    chr = ["20" for i in 1:size(Hadmix, 2)]
    nCEUtrain = Int(0.9 * size(HCEU, 1))
    nYRItrain = Int(0.9 * size(HYRI, 1))
    nadmixedtrain = Int(0.9 * size(Hadmix, 1))
    H1_train = [HCEU[1:2:nCEUtrain, :]; HYRI[1:2:nYRItrain, :]; Hadmix[1:2:nadmixedtrain, :]]
    H2_train = [HCEU[2:2:nCEUtrain, :]; HYRI[2:2:nYRItrain, :]; Hadmix[2:2:nadmixedtrain, :]]
    create_plink("train", H1_train, H2_train, chr=chr, pos=pos)
    write_vcf("train.phased.vcf.gz", H1_train, H2_train, chr=chr, pos=pos)
    open("train.populations.txt", "w") do io
        [println(io, "CEU") for i in 1:nCEUtrain>>1]
        [println(io, "YRI") for i in 1:nYRItrain>>1]
        [println(io, "admixed") for i in 1:nadmixedtrain>>1]
    end

    # write 10% samples to testing data
    H1_test = [HCEU[nCEUtrain+1:2:end, :]; HYRI[nYRItrain+1:2:end, :]; Hadmix[nadmixedtrain+1:2:end, :]]
    H2_test = [HCEU[nCEUtrain+2:2:end, :]; HYRI[nYRItrain+2:2:end, :]; Hadmix[nadmixedtrain+2:2:end, :]]
    create_plink("test", H1_test, H2_test, chr=chr, pos=pos)
    open("test.populations.txt", "w") do io
        [println(io, "CEU") for i in 1:(size(HCEU, 1) - nCEUtrain)>>1]
        [println(io, "YRI") for i in 1:(size(HYRI, 1) - nYRItrain)>>1]
        [println(io, "admixed") for i in 1:(size(Hadmix, 1) - nadmixedtrain)>>1]
    end

    # find top 500 differentially expressed SNPs via AIM select
    diff_markers = find_500_AIMs(HCEU, HYRI)
    writedlm("diff_markers.txt", diff_markers)

    # also generate QC file that contains all SNPs and all samples
    snpdata = SnpData("train")
    snpIDs = snpdata.snp_info[!, :snpid]
    sampleIDs = Matrix(snpdata.person_info[!, 1:2])
    writedlm("variants_qc.txt", snpIDs)
    writedlm("samples_qc.txt", sampleIDs)

    # move relevant files to outdir
    mv("train.bed", outdir * "train.bed", force=true)
    mv("train.bim", outdir * "train.bim", force=true)
    mv("train.fam", outdir * "train.fam", force=true)
    mv("test.bed", outdir * "test.bed", force=true)
    mv("test.bim", outdir * "test.bim", force=true)
    mv("test.fam", outdir * "test.fam", force=true)
    mv("variants_qc.txt", outdir * "variants_qc.txt", force=true)
    mv("samples_qc.txt", outdir * "samples_qc.txt", force=true)
    mv("train.phased.vcf.gz", outdir * "train.phased.vcf.gz", force=true)
    mv("train.populations.txt", outdir * "train.populations.txt", force=true)
    mv("test.populations.txt", outdir * "test.populations.txt", force=true)
    mv("diff_markers.txt", outdir * "diff_markers.txt", force=true)

    # clean up 
    rm("CEU.vcf", force=true)
    rm("YRI.vcf", force=true)
    rm("all.vcf", force=true)
    rm("admixed.vcf", force=true)
    rm("CEU.full.vcf", force=true)
    rm("YRI.full.vcf", force=true)
    rm("all.full.vcf", force=true)
    rm("admixed.full.vcf.gz", force=true)
end

function read_vcf_pos(vcffile)
    _,_,_,pos,_,_,_ = convert_ht(Bool, vcffile, save_snp_info=true, msg="importing..")
    return pos
end

function make_every_mapfile_needed()
    # make map file for rapid https://github.com/ZhiGroup/RaPID
    vcffile = "train.phased.vcf.gz"
    outfile = "train.rapid.map"
    run(`python $rapid_filter_exe /scratch/users/bbchu/RaPID/genetic_maps/genetic_map_GRCh38_chr20.txt raw_map.txt`)
    run(`python $rapid_interpolate_exe raw_map.txt $vcffile $outfile`)

    # make map file needed for knockoffgwas (todo: how to define map_cM and recombination rate for simulated data??)
    pos = read_vcf_pos(vcffile)
    map_cM = LinRange(0.0, round(pos[end] / 1e6), length(pos)) # even spacing
    recomb_rate = 0.01 .* rand(length(pos)) # random recomb rate
    open("sim.partition.map", "w") do io
        println(io, "Chromosome\tPosition(bp)\tRate(cM/Mb)\tMap(cM)")
        for i in 1:length(pos)
            println(io, "chr20\t", pos[i], '\t', recomb_rate[i], '\t', map_cM[i])
        end
    end

    # partition genotypes to groups (for making group knockoffs)
    partition(partition_exe, "train", "sim.partition.map", "variants_qc.txt", "sim.partition.txt")
end

function make_knockoffs(seed, outdir)
    cd(outdir)

    # make map files
    make_every_mapfile_needed()

    # run rapid to detect IBD segments
    Random.seed!(seed)
    vcffile = "train.phased.vcf.gz"
    mapfile = "train.rapid.map"
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
    make_bgen_samplefile("train.sample", VCFTools.nsamples(vcffile))

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

function get_diff_markers(data_dir)
    try
        return Vector{Int}(vec(readdlm(data_dir)))
    catch
        return Int[]
    end
end

function one_simulation(nYRI, nCEU, nmate, nadmixed, p, dist, seed; 
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
    isdir("PRS_Admixture_Simulation") || cp("/scratch/users/bbchu/PRS_Admixture_Simulation", data_dir * "PRS_Admixture_Simulation", force=true)
    simulate_genotypes(nYRI, nCEU, nmate, nadmixed, p, seed, data_dir)

    # make knockoffs
    make_knockoffs(seed, data_dir)
    GC.gc()

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
        mses = cv_iht(ytrain, xla, covar, d=dist, l=link, path=path, init_beta=init_beta, max_iter=100)
        GC.gc()
        Random.seed!(seed)
        k_rough_guess = path[argmin(mses)]
        dense_path = (k_rough_guess - 9):(k_rough_guess + 9)
        mses_new = cv_iht(ytrain, xla, covar, d=dist, l=link, path=dense_path, init_beta=init_beta, max_iter=100)
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
        mses = cv_iht(ytrain, xko_la, covar, d=dist, l=link, path=path, init_beta=init_beta, max_iter=100)
        GC.gc()
        Random.seed!(seed)
        k_rough_guess = path[argmin(mses)]
        dense_path = (k_rough_guess - 9):(k_rough_guess + 9)
        mses_new = cv_iht(ytrain, xko_la, covar, d=dist, l=link, path=dense_path, init_beta=init_beta, max_iter=100)
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
            for i in 1:size(xla, 1)
                println(io, "$i 1 $(ytrain[i])") 
            end
        end
        if use_PCA
            open(covfile, "w") do io
                println(io, "FID IID PC1 PC2 PC3 PC4 PC5") 
                for i in 1:size(xla, 1)
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

##
## Genotypes are simulated using msprime, following https://github.com/taylorcavazos/PRS_Admixture_Simulation
## Genotypes mimics chr20 linkage
## Founders used to make admixed samples are not included in the final sample used for training
## To train PRS, we use all samples (CEU + YRI + admixed)
## To test PRS, we use all testing samples (CEU + YRI + admixed) that are disjoint from training samples
##

nYRI = 5000
nCEU = 5000
nadmixed = 2000 # for admixed samples
nmate = 1000    # for generating founder population (1000 each from YRI and CEU) to make admixed samples; total samples = 5000+5000+2000-1000*2 = 10000
p = 50000
use_PCA = true
d = Normal()
seed = parse(Int, ARGS[1])
one_simulation(nYRI, nCEU, nmate, nadmixed, p, d, seed,
    use_PCA=use_PCA, simulate_interactions=false)
