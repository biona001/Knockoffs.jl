using SnpArrays
using CSV, DataFrames
#
# separate by population
# 
populations = ["african", "asian", "bangladeshi", "british", "caribbean", "chinese",
    "indian", "irish", "pakistani", "white_asian", "white_black", "white"]
xdata = SnpData("ukb_gen_merged") # 407434×591513 (all sample populations with all snps passing QC)
p = size(xdata.snparray, 2)
sampleIDs = parse.(Int, xdata.person_info.iid)
for pop in populations
    ids = CSV.read("samples_$pop.txt", DataFrame)[:, 1]
    sample_idx = sort!(Vector{Int}(indexin(ids, sampleIDs)))
    SnpArrays.filter("ukb_gen_merged", sample_idx, 1:p; des = "ukb.$pop")
end


#
# Subset chromosome 10 for testing
#
chr = 10
chr_snp_idx = findall(x -> x == "10", xdata.snp_info.chromosome)
mkdir("chr$chr")
cd("chr$chr")
for pop in populations
    sample_idx = 1:size(SnpArray("../ukb.$pop.bed"), 1)
    SnpArrays.filter("../ukb.$pop", sample_idx, chr_snp_idx; des = "ukb.chr$chr.$pop")
end
