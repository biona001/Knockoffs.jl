using SnpArrays

sample_idx = 1:10000
for chr in 1:22
    println("Processing chr $chr...")
    plinkname = "/scratch/users/bbchu/ukb_SHAPEIT/knockoffs/ukb_gen_chr$(chr)_ibd1_res0" #shapeit knockoffs
    xdata = SnpData(plinkname)
    isknockoff = endswith.(xdata.snp_info[!, :snpid], ".k")
    original, knockoff = Int[], Int[]
    n, p = size(xdata)
    for i in 1:p
        isknockoff[i] ? push!(knockoff, i) : push!(original, i)
    end
    # save original snps in current chromosome
    SnpArrays.filter(plinkname, sample_idx, original; des = "ukb.10k.chr$chr")
end
