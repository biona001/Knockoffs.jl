# compute PCA using proPCA software
global propca_exe = "/scratch/users/bbchu/ProPCA/build/propca"
seed = 1

for chr in 1:22
    println("computing PCA for chr $chr")
    plinkname = "/scratch/users/bbchu/ukb/subset/ukb.10k.chr$chr"
    outfile = "/scratch/users/bbchu/ukb/subset_pca/ukb.10k.chr$chr."
    run(`$propca_exe -g $plinkname -k 10 -o $outfile -nt $(Threads.nthreads()) -seed $seed`)
end
