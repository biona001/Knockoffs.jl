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
    # filter knockoff genotypes
    ko_plinkfile = "/scratch/users/bbchu/ukb/subset/ukb.10k.merged.chr$chr"
    original = vec(readdlm("/scratch/users/bbchu/ukb/subset/ukb.chr$chr.original.snp.index", Int))
    knockoff = vec(readdlm("/scratch/users/bbchu/ukb/subset/ukb.chr$chr.knockoff.snp.index", Int))
    ko_snp_idx = original[snp_idx] ∪ knockoff[snp_idx]
    SnpArrays.filter(ko_plinkfile, sample_idx, ko_snp_idx, des=ko_outfile)
    # also save which ones are knockoffs which are original SNPs
    writedlm("/scratch/users/bbchu/ukb/low_LD/ukb.chr$chr.original.snp.index.lowLD", original[snp_idx])
    writedlm("/scratch/users/bbchu/ukb/low_LD/ukb.chr$chr.knockoff.snp.index.lowLD", knockoff[snp_idx])
end

chr = 10
threshold = 0.7
outfile = "/scratch/users/bbchu/ukb/low_LD/ukb.10k.lowLD.chr$chr.threshold$threshold"
ko_outfile = "/scratch/users/bbchu/ukb/low_LD/ukb.10k.lowLD.chr$chr.threshold$threshold.knockoff"
prune(chr, threshold, outfile, ko_outfile)
