using SnpArrays
plinkname = "/scratch/users/bbchu/ukb_SHAPEIT/subset/ukb.10k.chr10"
xdata = SnpData(plinkname)
@time r, θ, α = fastphase(xdata; n=1000, out="ukb_chr10_n1000")

# for reading aggregated results later
# r, θ, α = process_fastphase_output(pwd(), T=1, extension="ukb_chr10_n1000")
