using Revise
using SnpArrays
using DelimitedFiles
using Random
using MendelIHT
using Knockoffs
using LinearAlgebra
using GLMNet
using Distributions
using DataFrames
using CSV
using Printf

fdr = 0.6
seed = 12
cd("/scratch/users/bbchu/ukb/prs/fdr$fdr/sim$seed")

function R2(X::AbstractMatrix, y::AbstractVector, β̂::AbstractVector)
    μ = y - X * β̂
    tss = y .- mean(y)
    return 1 - dot(μ, μ) / dot(tss, tss)
end

chr = 10
plinkname = "/scratch/users/bbchu/ukb/subset/ukb.10k.chr10"
knockoffname = "/scratch/users/bbchu/ukb/subset/ukb.10k.merged.chr10"
x = SnpArray(plinkname * ".bed")
xko = SnpArray(knockoffname * ".bed")
# xla = SnpLinAlg{Float64}(x, center=true, scale=true, impute=true)
xla = convert(Matrix{Float64}, x, center=true, scale=true, impute=true)
xko_la = convert(Matrix{Float64}, xko, center=true, scale=true, impute=true)
original = vec(readdlm("/scratch/users/bbchu/ukb/subset/ukb.chr$chr.original.snp.index", Int))
knockoff = vec(readdlm("/scratch/users/bbchu/ukb/subset/ukb.chr$chr.knockoff.snp.index", Int))

#
# simulate phenotypes using UKB chr10 subset
#
n, p = size(x)
# simulate β
Random.seed!(seed)
k = 100 # number of causal SNPs
h2 = 0.5 # heritability
d = Normal(0, sqrt(h2 / (2k))) # from paper: Efficient Implementation of Penalized Regression for Genetic Risk Prediction
β = zeros(p)
β[1:k] .= rand(d, k)
shuffle!(β)
# simulate y
ϵ = Normal(0, 1 - h2)
y = xla * β + rand(ϵ, n)

#
# compare R2 across populations, save result in a dataframe
#
β_iht = vec(readdlm("iht.beta"))
β_iht_knockoff = extract_beta(vec(readdlm("iht.knockoff.beta")), fdr, original, knockoff)
β_iht_knockoff_cv = extract_beta(vec(readdlm("iht.knockoff.cv.beta")), fdr, original, knockoff)
β_iht_knockoff_combined_cv = extract_combine_beta(vec(readdlm("iht.knockoff.cv.beta")), original, knockoff)
β_lasso = vec(readdlm("lasso.beta"))
β_lasso_knockoff = extract_beta(vec(readdlm("lasso.knockoff.beta")), fdr, original, knockoff)
β_lasso_knockoff_combined = extract_combine_beta(vec(readdlm("lasso.knockoff.beta")), original, knockoff)

pop = "african"

xpop = SnpArray("/scratch/users/bbchu/ukb/populations/chr10/ukb.chr$chr.$pop.bed")
Xpop = SnpLinAlg{Float64}(xpop, center=true, scale=true, impute=true)
# simulate "true" phenotypes for these populations
ytrue = Xpop * β + rand(ϵ, size(Xpop, 1))



# IHT
iht_r2 = R2(Xpop, ytrue, β_iht)
iht_ko_r2 = R2(Xpop, ytrue, β_iht_knockoff) # knockoff IHT
iht_ko_cv_r2 = R2(Xpop, ytrue, β_iht_knockoff_cv) # better cv routine
iht_ko_cv_combined_r2 = R2(Xpop, ytrue, β_iht_knockoff_combined_cv) # original β combined with its knockoff

# LASSO
lasso_r2 = R2(Xpop, ytrue, β_lasso)
lasso_ko_r2 = R2(Xpop, ytrue, β_lasso_knockoff) # knockoff IHT
lasso_ko_combined_r2 = R2(Xpop, ytrue, β_lasso_knockoff_combined) # original β combined with its knockoff

