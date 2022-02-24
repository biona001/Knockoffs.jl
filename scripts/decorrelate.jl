using SnpArrays
using DelimitedFiles
using ProgressMeter
using Knockoffs

xdata = SnpData("/scratch/users/bbchu/ukb_prs/fastphase/alpha0.00/ukb_gen_merged")
original = indexin(readdlm("/scratch/users/bbchu/ukb_prs/fastphase/alpha0.00/ukb_gen_merged_original.txt"), xdata.snp_info[!, :snpid])
original = Vector{Int}(vec(original))
knockoff = indexin(readdlm("/scratch/users/bbchu/ukb_prs/fastphase/alpha0.00/ukb_gen_merged_knockoff.txt"), xdata.snp_info[!, :snpid])
knockoff = Vector{Int}(vec(knockoff))
decorrelate_knockoffs(
    xdata;
    α = 0.05,
    original = original,
    knockoff = knockoff,
    outfile = "decorrelated_knockoffs",
    outdir = "/scratch/users/bbchu/ukb_prs/fastphase/alpha0.05",
    verbose=true
    )
