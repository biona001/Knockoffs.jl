using SnpArrays
using LinearAlgebra
using Statistics
using DelimitedFiles

"""
    prune(plinkfile::AbstractString, threshold::Real, outfile::AbstractString)

Filters a PLINK file and remove all SNPs whose correlation with other SNPs
exceed `threshold`. Saves resulting plink file into `outfile`
"""
function prune(chr::Int, threshold::Real, outfile::AbstractString, ko_outfile::AbstractString)
    0 ≤ threshold ≤ 1 || error("threshold must be between 0 and 1 but was $threshold")
    plinkfile = "/scratch/users/bbchu/ukb/subset/ukb.10k.chr$chr"
    x = convert(Matrix{Float64}, SnpArray(plinkfile * ".bed"), center=true, scale=true)
    R = cor(x)
    # find all SNPs whose pairwise correlation exceeds threshold
    snps_to_exclude = Int[]
    for j in 1:size(R, 2)
        j ∈ snps_to_exclude && continue
        for i in j+1:size(R, 1)
            abs(R[i, j]) ≥ threshold && push!(snps_to_exclude, i)
        end
    end
    unique!(snps_to_exclude)
    # filter original PLINK file
    sample_idx = 1:size(x, 1) # keep all samples
    snp_idx = setdiff(1:size(x, 2), snps_to_exclude)
    SnpArrays.filter(plinkfile, sample_idx, snp_idx, des=outfile)
    writedlm("snps_kept.txt", snp_idx)
    # filter knockoff genotypes
    ko_plinkfile = "/scratch/users/bbchu/ukb/subset/ukb.10k.merged.chr$chr"
    original = vec(readdlm("/scratch/users/bbchu/ukb/subset/ukb.chr$chr.original.snp.index", Int))
    knockoff = vec(readdlm("/scratch/users/bbchu/ukb/subset/ukb.chr$chr.knockoff.snp.index", Int))
    ko_snp_idx = original[snp_idx] ∪ knockoff[snp_idx]
    SnpArrays.filter(ko_plinkfile, sample_idx, ko_snp_idx, des=ko_outfile)
    # also save which ones are knockoffs which are original SNPs
    new_original = Int[]
    new_knockoff = Int[]
    offset = 0
    for i in 1:size(x, 2)
        if i in snps_to_exclude
            offset += 2
            continue
        end
        push!(new_original, original[i] - offset)
        push!(new_knockoff, knockoff[i] - offset)
    end
    writedlm("/scratch/users/bbchu/ukb/low_LD/ukb.chr$chr.original.snp.index.lowLD", new_original)
    writedlm("/scratch/users/bbchu/ukb/low_LD/ukb.chr$chr.knockoff.snp.index.lowLD", new_knockoff)
end

#
# First filter 10k subset of UKB chr10
#
chr = 10
threshold = 0.7
outfile = "/scratch/users/bbchu/ukb/low_LD/ukb.10k.lowLD.chr$chr.threshold$threshold"
ko_outfile = "/scratch/users/bbchu/ukb/low_LD/ukb.10k.lowLD.chr$chr.threshold$threshold.knockoff"
prune(chr, threshold, outfile, ko_outfile)

#
# also need to filter test data the same way
#
snp_idx = Int.(vec(readdlm("snps_kept.txt")))
cd("/scratch/users/bbchu/ukb/populations/chr10/lowLD")

populations = ["african", "asian", "bangladeshi", "british", "caribbean", "chinese",
    "indian", "irish", "pakistani", "white_asian", "white_black", "white"]
for pop in populations
    plinkfile = "/scratch/users/bbchu/ukb/populations/chr10/ukb.chr$chr.$pop"
    outfile = "/scratch/users/bbchu/ukb/populations/chr10/lowLD/ukb.chr$chr.$pop.lowLD"
    x = SnpArray(plinkfile * ".bed")
    sample_idx = 1:size(x, 1)
    SnpArrays.filter(plinkfile, sample_idx, snp_idx, des=outfile)
end
writedlm("snps_kept.txt", snp_idx)





# 
# check correlation of PCs with SNPs
#
using SnpArrays
using LinearAlgebra
using Statistics
using DelimitedFiles
x = convert(Matrix{Float64}, SnpArray("/scratch/users/bbchu/ukb/subset/ukb.10k.chr10.bed"), center=true, scale=true)
z = readdlm("/scratch/users/bbchu/ukb/subset_pca/ukb.10k.chr10.projections.txt")

# many SNPs are highly correlated with PC1, and
# these SNPs are usually clumped together
pc1 = z[:, 1]
counter = 0
for i in 1:size(x, 2)
    if abs(cor(pc1, @view(x[:, i]))) ≥ 0.75
        counter += 1
        println("snp $i")
    end
end
counter # 8 SNPs from pos 15333 to 15346

# 0 SNPs in different population are correlated with PC1
x2 = convert(Matrix{Float64}, SnpArray("/scratch/users/bbchu/ukb/populations/chr10/ukb.chr10.african.bed"), center=true, scale=true)
pc1_sub = pc1[1:size(x2, 1)]
counter = 0
for i in 1:size(x2, 2)
    if abs(cor(pc1_sub, @view(x2[:, i]))) ≥ 0.1
        counter += 1
        println("snp $i")
    end
end
counter # 0: no SNPs >0.1 correlated with PC1 subset 







#
# Seed 1111: lasso knockoff predicts better than lasso, if PC1 is confounder
# Is this because LASSO had false positives associated with PC1?
# Need to confirm with other seed
#
pc1 = z[:, 1]
counter = 0
correlated_snps = Int[]
for i in 1:size(xla, 2)
    if abs(cor(pc1, @view(xla[:, i]))) ≥ 0.75
        counter += 1
        push!(correlated_snps, i)
    end
end
correlated_snps # 8 SNPs from pos 15333 to 15346

correlated_snps ∩ correct_snps



using UnicodePlots, DelimitedFiles
original = vec(readdlm("/scratch/users/bbchu/ukb/subset/ukb.chr10.original.snp.index", Int))
knockoff = vec(readdlm("/scratch/users/bbchu/ukb/subset/ukb.chr10.knockoff.snp.index", Int))
sim = 2
ko_β_lasso = vec(readdlm("/scratch/users/bbchu/ukb/prs/confound_pc/1PC_effect0.2_Radj100/fdr0.1/sim$sim/lasso.knockoff.beta"))
W = abs.(ko_β_lasso[original]) .- abs.(ko_β_lasso[knockoff])
histogram(W)


sim = 5
ko_β_lasso = vec(readdlm("/scratch/users/bbchu/ukb/prs/confound_pc/1PC_effect0.2_Radj100/fdr0.1/sim$sim/lasso.knockoff.beta"))
[ko_β_lasso[original][15333:15346] ko_β_lasso[knockoff][15333:15346]]



#
# Most real genotypes (UKB or msprime) have r2 ≈ 0.87 between original 
# SNP and its knockoff. But on my own simulated genotypes, r2 is around 0.4
#
using UnicodePlots, DelimitedFiles
using SnpArrays, LinearAlgebra
using Statistics

# UKB 10k subset (single-SNP resolution)
xko_la = convert(Matrix{Float64}, SnpArray("/scratch/users/bbchu/ukb/subset/ukb.10k.merged.chr10.bed"), center=true, scale=true)
r2 = 0
for snp in 1:size(xko_la, 2) >> 1
    r2 += cor(@view(xko_la[:, 2snp]), @view(xko_la[:, 2snp - 1]))
end
r2 /= size(xko_la, 2) >> 1 # 0.8321148302625268

# low LD (filtered highly correlated snps)
xko_la = convert(Matrix{Float64}, SnpArray("/scratch/users/bbchu/ukb/low_LD/ukb.10k.lowLD.chr10.threshold0.7.knockoff.bed"), center=true, scale=true)
r2 = 0
for snp in 1:size(xko_la, 2) >> 1
    r2 += cor(@view(xko_la[:, 2snp]), @view(xko_la[:, 2snp - 1]))
end
r2 /= size(xko_la, 2) >> 1 # 0.7967258603579113

# grouped UKB data (res5)
x = SnpArray("/scratch/users/bbchu/ukb/groups/Radj5_K50_s0/ukb_gen_chr10.bed")
xko_la = convert(Matrix{Float64}, @view(x[1:10000, 1:10000]), center=true, scale=true)
r2 = 0
for snp in 1:size(xko_la, 2) >> 1
    r2 += cor(@view(xko_la[:, 2snp]), @view(xko_la[:, 2snp - 1]))
end
r2 /= size(xko_la, 2) >> 1 # 0.6125851073067299

# grouped UKB data (res2)
x = SnpArray("/oak/stanford/groups/candes/ukbiobank_tmp/knockoffs/Radj2_K50_s1/ukb_gen_chr10.bed")
xko_la = convert(Matrix{Float64}, @view(x[1:10000, 1:10000]), center=true, scale=true)
r2 = 0
for snp in 1:size(xko_la, 2) >> 1
    r2 += cor(@view(xko_la[:, 2snp]), @view(xko_la[:, 2snp - 1]))
end
r2 /= size(xko_la, 2) >> 1 # 0.37299842907393405

# my simulated (independent) genotypes
xko_la = convert(Matrix{Float64}, SnpArray("/scratch/users/bbchu/PRS_sims/no_struct/data/sim20/knockoffs/train.knockoffs_res0.bed"), center=true, scale=true)
r2 = 0
success = 0
for snp in 1:size(xko_la, 2) >> 1
    tmp = cor(@view(xko_la[:, 2snp]), @view(xko_la[:, 2snp - 1]))
    if !isnan(tmp)
        r2 += tmp
        success += 1
    end
end
r2 /= success # 0.448942588361576



# for seed 1111
idx = findall(!iszero, β)
r2 = 0.0
for i in idx
    r2 += cor(@view(xko_la[:, 2i]), @view(xko_la[:, 2i - 1]))
end
r2 /= length(idx) # 0.8457489285303439



# try letting causal snps be those that are not highly correlated with its ko
possible_causal_snp_idx = Int[]
their_correlations = Float64[]
for snp in 1:size(xla, 2)
    r2 = cor(@view(xko_la[:, 2snp]), @view(xko_la[:, 2snp - 1]))
    if r2 ≤ 0.45
        push!(possible_causal_snp_idx, snp)
        push!(their_correlations, r2)
    end
end
[possible_causal_snp_idx their_correlations]
