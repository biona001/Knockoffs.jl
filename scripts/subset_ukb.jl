using SnpArrays
using DelimitedFiles

# copy UKB data and its knockoff to scratch (>100GB)
# this data contains all samples and all genotypes, along with their knockoffs
# cp /oak/stanford/groups/candes/prs_zhimei/knockoffs/Radj100_s0/* .

# split ukb data and its knockoff by chromosome
cd("/scratch/users/bbchu/ukb/split")
ukb_knockoff = "/scratch/users/bbchu/ukb/merged_knockoff/ukb_gen_merged"
xdata = SnpData(ukb_knockoff) # 350119×1183026 (unrelated British samples)
splitted = SnpArrays.split_plink(xdata, :chromosome; prefix="split.chr.")

# filter data, keeping first 10000 samples, then create original UKB data without knockoffs
cd("/scratch/users/bbchu/ukb/subset")
original_snps = vec(readdlm("../merged_knockoff/ukb_gen_merged_original.txt"))
knockoff_snps = vec(readdlm("../merged_knockoff/ukb_gen_merged_knockoff.txt"))
sample_idx = 1:10000
for chr in 1:22
    plinkfile = "/scratch/users/bbchu/ukb/split/split.chr.$chr"
    p = size(SnpArray(plinkfile * ".bed"), 2)
    # save all snps (original + knockoff) in current chromosome
    SnpArrays.filter(plinkfile, sample_idx, 1:p; des = "ukb.10k.merged.chr$chr")
    # save only original SNPs in current chromosome
    chr_snp_list = SnpData(plinkfile).snp_info.snpid
    original_idx = indexin(original_snps, chr_snp_list)
    keep_idx = Vector{Int}(original_idx[original_idx .!= nothing])
    SnpArrays.filter("ukb.10k.merged.chr$chr", sample_idx, keep_idx; des = "ukb.10k.chr$chr")
    writedlm("ukb.chr$chr.original.snp.index", keep_idx)
    # also save knockoff index
    knockoff_idx = indexin(knockoff_snps, chr_snp_list)
    writedlm("ukb.chr$chr.knockoff.snp.index", knockoff_idx[knockoff_idx .!= nothing])
end
