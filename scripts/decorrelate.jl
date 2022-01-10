using SnpArrays
using Knockoffs
using DataFrames
using CSV

res = 0
for chr in 1:22
    chr == 10 && continue
    plinkfile = "/scratch/users/bbchu/ukb_SHAPEIT/knockoffs/ukb_gen_chr$(chr)_ibd1_res$(res)"
    xdata = SnpData(plinkfile)
    isknockoff = endswith.(xdata.snp_info[!, :snpid], ".k")
    original, knockoff = Int[], Int[]
    for i in 1:size(xdata)[2]
        isknockoff[i] ? push!(knockoff, i) : push!(original, i)
    end

    @time decorrelate_knockoffs(plinkfile, original, knockoff;
        outfile = "ukb_gen_chr$(chr)_ibd1_res$(res)_decorrelated",
        outdir = pwd(),
        r2_threshold = 0.95)
end
