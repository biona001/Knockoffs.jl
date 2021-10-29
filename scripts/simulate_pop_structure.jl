using Revise
using SnpArrays
using DelimitedFiles
using Random
using Knockoffs
using LinearAlgebra
using Distributions
using DataFrames
using CSV
using ProgressMeter
using UnicodePlots
using MendelIHT

"""
    simulate_pop_structure(plinkfile, n, p)

Simulate genotypes with K = 2 populations. 50 SNPs will have different allele 
frequencies between the populations. 

# Inputs
- `plinkfile`: Output plink file name. 
- `n`: Number of samples
- `p`: Number of SNPs

# Output
- `x`: A simulated `SnpArray`. Also saves binary plink files `plinkfile.bed`,
    `plinkfile.bim`, `plinkfile.fam` to the current directory
- `populations`: Vector of length `n` indicating population membership for each
    sample. Also saved as `populations.txt`
- `diff_markers`: Indices of the differentially expressed alleles. Also saved 
    as `diff_markers.txt`
"""
function simulate_pop_structure(plinkfile::AbstractString, n::Int, p::Int)
    # first simulate genotypes treating all samples equally
    x = SnpArray(plinkfile * ".bed", n, p)
    pmeter = Progress(p)
    for j in 1:p
        allele_freq = rand()
        d = Binomial(2, allele_freq)
        for i in 1:n
            c = rand(d)
            if c == 0
                x[i, j] = 0x00
            elseif c == 1
                x[i, j] = 0x02
            elseif c == 2
                x[i, j] = 0x03
            else
                throw(MissingException("matrix shouldn't have missing values!"))
            end
        end
        next!(pmeter) # update progress
    end
    # assign populations and simulate 50 unually differentiated markers
    populations = rand(1:2, n)
    diff_markers = sample(1:p, 50, replace=false)
    for j in diff_markers
        pop1_allele_freq = 0.4rand()
        pop2_allele_freq = pop1_allele_freq + 0.6
        pop1_dist = Binomial(2, pop1_allele_freq)
        pop2_dist = Binomial(2, pop2_allele_freq)
        for i in 1:n
            c = isone(populations[i]) ? rand(pop1_dist) : rand(pop2_dist)
            if c == 0
                x[i, j] = 0x00
            elseif c == 1
                x[i, j] = 0x02
            elseif c == 2
                x[i, j] = 0x03
            else
                throw(MissingException("matrix shouldn't have missing values!"))
            end
        end
    end
    # generate bim and fam file
    make_bim_fam_files(x, -9ones(Int, n), plinkfile)
    # save pop1/pop2 index and unually differentiated marker indices
    writedlm("populations.txt", populations)
    writedlm("diff_markers.txt", diff_markers)
    return x, populations, diff_markers
end

# cd("/scratch/users/bbchu/ukb/pop_struct/")
seed = 2021
Random.seed!(seed)
n = 2000
p = 50000
plinkfile = "sim"
x, populations, diff_markers = simulate_pop_structure(plinkfile, n, p)




#
# check performance of IHT
#
# simulate β
xla = SnpLinAlg{Float64}(x, center=true, scale=true, impute=true)
Random.seed!(seed)
k = 100 # number of causal SNPs
h2 = 0.5 # heritability
d = Normal(0, sqrt(h2 / (2k))) # from paper: Efficient Implementation of Penalized Regression for Genetic Risk Prediction
β = zeros(p)
β[1:k>>1] .= rand(d, k>>1)
shuffle!(β)
β[diff_markers] .= rand(d, k>>1)
# simulate y
ϵ = Normal(0, 1 - h2)
y = xla * β + rand(ϵ, n)
writedlm("y_true.txt", y)
writedlm("beta_true.txt", β)

#
# Run standard IHT
#
Random.seed!(seed)
path = 10:10:200
mses = cv_iht(y, xla, path=path, init_beta=true)
GC.gc()
Random.seed!(seed)
k_rough_guess = path[argmin(mses)]
dense_path = (k_rough_guess - 9):(k_rough_guess + 9)
mses_new = cv_iht(y, xla, path=dense_path, init_beta=true)
GC.gc()
Random.seed!(seed)
result = fit_iht(y, xla, k=dense_path[argmin(mses_new)], init_beta=true, max_iter=500)
@show result
writedlm("iht.beta", result.beta)
GC.gc()

# check correctness
function TP(correct_snps, signif_snps)
    return length(signif_snps ∩ correct_snps) / length(correct_snps)
end
function FDR(correct_snps, signif_snps)
    FP = length(signif_snps) - length(signif_snps ∩ correct_snps) # number of false positives
    # FPR = FP / (FP + TN) # https://en.wikipedia.org/wiki/False_positive_rate#Definition
    FDR = FP / length(signif_snps)
    return FDR
end
correct_snps = findall(!iszero, β)
@show TP(correct_snps, findall(!iszero, result.beta)) # 0.44
@show FDR(correct_snps, findall(!iszero, result.beta)) # 0.6857142857142857


count(!iszero, result.beta)
[β[correct_snps] result.beta[correct_snps]]

