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
BLAS.set_num_threads(1)

global qctools_exe = "/scratch/users/bbchu/qctool/build/release/qctool_v2.0.7"
global snpknock2_exe = "/scratch/users/bbchu/knockoffgwas/snpknock2/bin/snpknock2"
global rapid_exe = "/scratch/users/bbchu/RaPID/RaPID_v.1.7"
global partition_exe = "/scratch/users/bbchu/knockoffgwas/knockoffgwas/utils/partition.R";

"""
    simulate_pop_structure(n, p)

Simulate genotypes with K = 2 populations. 1% of SNPs will have different allele 
frequencies between the populations.

# Inputs
- `plinkfile`: Output plink file name. 
- `n`: Number of samples
- `p`: Number of SNPs

# Output
- `x1`: n×p matrix of the 1st haplotype for each sample. Each row is a haplotype
- `x2`: n×p matrix of the 2nd haplotype for each sample. `x = x1 + x2`
- `populations`: Vector of length `n` indicating population membership for eachsample. 
- `diff_markers`: Indices of the differentially expressed alleles.

# Reference
https://www.nature.com/articles/nrg2813
"""
function simulate_pop_structure(n::Int, p::Int)
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
    # assign populations and simulate 0.01p unually differentiated markers
    populations = rand(1:2, n)
    diff_markers = sample(1:p, Int(0.01p), replace=false)
    @inbounds for j in diff_markers
        pop1_allele_freq = 0.4rand()
        pop2_allele_freq = pop1_allele_freq + 0.6
        pop1_dist = Bernoulli(pop1_allele_freq)
        pop2_dist = Bernoulli(pop2_allele_freq)
        for i in 1:n
            d = isone(populations[i]) ? pop1_dist : pop2_dist
            x1[i, j] = rand(d)
            x2[i, j] = rand(d)
        end
    end
    return x1, x2, populations, diff_markers
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
    # combine offsprings and parents
    H1 = [h1; x1]
    H2 = [h2; x2]
    return H1, H2, fathers, mothers
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
    df = readdlm(inputfile)
    println(writer, "CHR ID1 HID1 ID2 HID2 BP.start BP.end site.start site.end cM FAM1 FAM2")
    for r in eachrow(df)
        chr, id1, id2, hap1, hap2, start_pos, end_pos, genetic_len, start_site, end_site = 
            Int(r[1]), Int(r[2]), Int(r[3]), Int(r[4]), Int(r[5]), Int(r[6]), Int(r[7]),
            r[8], Int(r[9]), Int(r[10])
        println(writer, chr, ' ', id1, ' ', hap1, ' ', id2, ' ', hap2, ' ', 
            start_pos, ' ', end_pos, ' ', start_site, ' ', end_site, ' ', 
            genetic_len, ' ', 1, ' ', 1)
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

function simulate_genotypes(n, p, offsprings, seed)
    Random.seed!(seed)
    h1, h2, populations, diff_markers = simulate_pop_structure(n, p)

    # simulate random mating to get IBD segments
    x1, x2 = simulate_IBD(h1, h2, populations, offsprings)

    # write phased genotypes to VCF format
    write_vcf("sim.phased.vcf.gz", x1, x2)

    # write unphased genotypes to PLINK binary format
    outfile = "sim"
    write_plink(outfile, x1, x2)

    # save pop1/pop2 index and unually differentiated marker indices
    writedlm("populations.txt", populations)
    writedlm("diff_markers.txt", diff_markers)
    
    # generate fake map file
    make_partition_mapfile("sim.partition.map", p)

    # also generate QC file that contains all SNPs and all samples
    snpdata = SnpData("sim")
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
    vcffile = "sim.phased.vcf.gz"
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
    outfile = "sim.bgen"
    run(`$qctools_exe -g $vcffile -og $outfile`)
    make_bgen_samplefile("sim.sample", n + offsprings)
    
    # snpknock2 arguments
    bgenfile = "sim"
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
    outfile = "sim.knockoffs"
    @time snpknock2(snpknock2_exe, bgenfile, sample_qc, variant_qc, mapfile, partfile, ibdfile, 
        K, cluster_size_min, cluster_size_max, hmm_rho, hmm_lambda, windows, n_threads, 
        seed, compute_references, generate_knockoffs, outfile)
end

# there is 100 causal snps, 50 on normally differentiated SNPs and 50 on abnormally diff snps
function simulate_beta_and_y(x::AbstractMatrix, diff_markers, seed)
    n, p = size(x)
    Random.seed!(seed)
    h2 = 0.5 # heritability
    k = 100 # number of causal snps
    d = Normal(0, sqrt(h2 / (2k))) # from paper: Efficient Implementation of Penalized Regression for Genetic Risk Prediction
    # simulate β
    β = zeros(p)
    norm_markers = setdiff(1:p, diff_markers)
    norm_causal_snps = sample(norm_markers, k >> 1, replace=false)
    β[norm_causal_snps] .= rand(d, k >> 1)
    shuffle!(diff_markers)
    diff_causal_markers = diff_markers[1:k >> 1]
    β[diff_causal_markers] .= rand(d, k >> 1)
    # simulate y
    ϵ = Normal(0, 1 - h2)
    y = x * β + rand(ϵ, n)
    # save results
    writedlm("y_true.txt", y)
    writedlm("beta_true.txt", β)
    writedlm("normal_markers.txt", norm_markers)
    writedlm("normal_causal_markers.txt", norm_causal_snps)
    writedlm("diff_markers.txt", diff_markers)
    writedlm("diff_causal_markers.txt", diff_causal_markers)
    return y, β
end

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

function summarize_result(groups, β_true, β_iht, β_iht_knockoff, β_iht_knockoff_cv, β_lasso, β_lasso_knockoff)
    length(groups) == length(β_true) || error("summarize_result: groups and β_true have different length!")
    df = DataFrame(Metric = String[], IHT = Float64[], IHT_ko = Float64[],
        IHT_ko_cv = Float64[], LASSO = Float64[], LASSO_ko = Float64[])
    
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
    correct_groups = get_signif_groups(β_true, groups)
    push!(df, hcat("TPP", TP(correct_snps, findall(!iszero, β_iht)),
        TP(correct_groups, get_signif_groups(β_iht_knockoff, groups)),
        TP(correct_groups, get_signif_groups(β_iht_knockoff_cv, groups)),
        TP(correct_groups, get_signif_groups(β_lasso, groups)),
        TP(correct_groups, get_signif_groups(β_lasso_knockoff, groups))
        ))
    # count FDR
    push!(df, hcat("FDR", FDR(correct_snps, findall(!iszero, β_iht)),
        FDR(correct_groups, get_signif_groups(β_iht_knockoff, groups)),
        FDR(correct_groups, get_signif_groups(β_iht_knockoff_cv, groups)),
        FDR(correct_groups, get_signif_groups(β_lasso, groups)),
        FDR(correct_groups, get_signif_groups(β_lasso_knockoff, groups))
        ))
    @show df
    CSV.write("summary.txt", df)
end

function one_simulation(n, p, offsprings, seed)
    cur_dir = pwd() * "/"

    # save all simulated data under data/sim$seed
    isdir("data") || mkdir("data")
    isdir("data/sim$seed") || mkdir("data/sim$seed")
    cd("data/sim$seed")
    data_dir = cur_dir * "data/sim$seed/"
    
    # simulate phased genotypes with pop structure and cryptic relatedness
    simulate_genotypes(n, p, offsprings, seed)
    
    # partition genotypes to groups (for making group knockoffs)
    partition(partition_exe, "sim", "sim.partition.map", "variants_qc.txt", "sim.partition.txt")
    
    # make knockoffs
    make_knockoffs(n, p, offsprings, seed)
    
    # loop over group resolution
    for group in 0:3
        x = SnpArray(data_dir * "knockoffs/sim.knockoffs_res$(group).bed")
        snpid = SnpData(data_dir * "knockoffs/sim.knockoffs_res$(group)").snp_info.snpid
        knockoff_idx = endswith.(snpid, ".k")
        original = findall(knockoff_idx .== false)
        knockoff = findall(knockoff_idx)
        groups = repeat(Vector{Int}(readdlm(data_dir * "knockoffs/sim.knockoffs_res$(group)_grp.txt", 
            header=true)[1][:, 2]), inner=2) .+ 1
        xla = convert(Matrix{Float64}, @view(x[:, original]), center=true, scale=true, impute=true)
        xko_la = convert(Matrix{Float64}, x, center=true, scale=true, impute=true)
        diff_markers = Vector{Int}(vec(readdlm(data_dir * "diff_markers.txt")))
        group_dir = cur_dir * "res$group/"
        isdir(group_dir) || mkdir(group_dir)
        cd(group_dir)

        for fdr in [0.05, 0.1, 0.25, 0.5]
            top_dir = group_dir * "fdr$fdr/"
            new_dir = group_dir * "fdr$fdr/sim$seed"
            isdir(top_dir) || mkdir(top_dir)
            isdir(new_dir) || mkdir(new_dir)
            cd(new_dir)
            
            #
            # simulate beta and y
            #
            Random.seed!(seed)
            y, β = simulate_beta_and_y(xla, diff_markers, seed)
            
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
            result = fit_iht(y, xla, k=dense_path[argmin(mses_new)], init_beta=true, max_iter=500)
            @show result
            β_iht = result.beta
            writedlm("iht.beta", β_iht)
            GC.gc()
            
            #
            # Run standard lasso
            #
            Random.seed!(seed)
            cv = glmnetcv(xla, y, nfolds=5, parallel=true) 
            β_lasso = GLMNet.coef(cv)
            writedlm("lasso.beta", β_lasso)

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
            result = fit_iht(y, xko_la, k=dense_path[argmin(mses_new)],
                init_beta=true, max_iter=500)
            @show result
            β_iht_knockoff = extract_beta(result.beta, fdr, groups, original, knockoff)
            writedlm("iht.knockoff.beta", result.beta)
            writedlm("iht.knockoff.postfilter.beta", β_iht_knockoff)

            #
            # run knockoff IHT with wrapped cross validation
            #
            Random.seed!(seed)
            path = 10:10:200
            z = ones(Float64, size(xla, 1))
            mses = cv_iht_knockoff(y, xko_la, z, original, knockoff, fdr, path=path,
                init_beta=true, group_ko=groups)
            GC.gc()
            Random.seed!(seed)
            k_rough_guess = path[argmin(mses)]
            dense_path = (k_rough_guess - 9):(k_rough_guess + 9)
            mses_new = cv_iht_knockoff(y, xko_la, z, original, knockoff, fdr,
                path=dense_path, init_beta=true, group_ko=groups)
            GC.gc()
            # adjust sparsity level so it best matches sparsity chosen by ko filter
            Random.seed!(seed)
            best_k = dense_path[argmin(mses_new)]
            best_β = tune_k(y, xko_la, original, knockoff, groups, fdr, best_k)
            β_iht_knockoff_cv = extract_beta(best_β, fdr, groups, original, knockoff)
            writedlm("iht.knockoff.cv.beta", best_β)
            writedlm("iht.knockoff.cv.postfilter.beta", β_iht_knockoff_cv)
            
            #
            # Run knockoff lasso
            #
            Random.seed!(seed)
            cv = glmnetcv(xko_la, y, nfolds=5, parallel=true)
            β_lasso_knockoff = extract_beta(GLMNet.coef(cv), fdr, groups, original, knockoff)
            writedlm("lasso.knockoff.beta", GLMNet.coef(cv))
            writedlm("lasso.knockoff.postfilter.beta", β_lasso_knockoff)

            #
            # process results
            #
            summarize_result(groups[original], β, β_iht, β_iht_knockoff, β_iht_knockoff_cv, β_lasso, β_lasso_knockoff)
            GC.gc()
        end
        GC.gc()
    end
end

n = 5000
p = 50000
offsprings = 500
seed = parse(Int, ARGS[1])
one_simulation(n, p, offsprings, seed)