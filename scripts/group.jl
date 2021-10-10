using CSV
using DataFrames
using DelimitedFiles
using SnpArrays
using Random

"""
Generate numeric matrix of `x` where columns are groups.

# Inputs
- `x`: A `SnpArray` that contain original snps and its knockoffs
- `groups`: A `Vector{Int}`. `groups[i]` is the group membership of snp `i`
- `knockoff` A `BitVector`. `knockoff[i] == true` means snp `i` is the knockoff

# Outputs
-`X`: numeric matrix of `x` where columns are groups
"""
function make_X_by_group(x::SnpArray, groups::Vector{Int}, knockoff::BitVector)
    n, p = size(x)
    length(groups) == p || error("Number of SNPs in `x` has different length than the `groups` vector")
    X = zeros(n, length(unique(groups)))
    snp = zeros(n)
    # loop over snp by snp
    @inbounds for i in 1:p
        copyto!(snp, @view(x[:, i]))
        knockoff[i] && continue # skip knockoffs if requested
        X[:, groups[i]] .+= snp
    end
    return X
end

"""
Generate numeric matrix of `x` where columns are groups and their knockoffs. 

# Outputs
- `X`: numeric matrix of `x` where columns are groups and their knockoffs
- `original_list`: `original_list[i]` is the column of `X` that stores the `i`th group 
- `knockoff_list`: `knockoff_list[i]` is the column of `X` that stores the knockoff 
    for the `i`th group. 
"""
function make_Xko_by_group(x::SnpArray, groups::Vector{Int}, knockoff::BitVector)
    n, p = size(x)
    length(groups) == p || error("Number of SNPs in `x` has different length than the `groups` vector")
    P = length(unique(groups))
    X = zeros(n, 2P)
    snp = zeros(n)
    # loop over snp by snp, adding knockoffs to index 2g and originals to 2g-1
    @inbounds for i in 1:p
        copyto!(snp, @view(x[:, i]))
        col = knockoff[i] ? 2groups[i] : 2groups[i] - 1
        X[:, col] .+= snp
    end
    # permute knockoffs and originals with probability 0.5
    original_list = [2i - 1 for i in 1:P]
    knockoff_list = [2i for i in 1:P]
    for j in 1:P
        if rand() < 0.5
            col1, col2 = 2j - 1, 2j
            @inbounds for k in 1:n
                X[k, col1], X[k, col2] = X[k, col2], X[k, col1]
            end
            original_list[j], knockoff_list[j] = knockoff_list[j], original_list[j]
        end
    end
    return X, original_list, knockoff_list
end

#
# import key and data
#
chr = 10
keyfile = "/scratch/users/bbchu/ukb/groups/Radj20_K50_s0/ukb_gen_chr$chr.key"
df = CSV.read(keyfile, DataFrame)
groups = df[!, :Group]
knockoff = convert(BitVector, df[!, :Knockoff])
x = SnpArray("/scratch/users/bbchu/ukb/groups/Radj20_K50_s0/ukb_gen_chr$chr.bed")

#
# generate grouped X (without knockoff) and save
#
# @time X = make_X_by_group(x, groups, knockoff)
# @time writedlm("chr$chr.noKnockoffs.data.txt", X)

#
# generate grouped X (includes knockoff) and save
#
# Random.seed!(2021)
# @time Xko, original_list, knockoff_list = make_Xko_by_group(x, groups, knockoff)
# @time writedlm("chr$chr.data.txt", Xko)
# @time writedlm("chr$chr.original.txt", Xko)
# @time writedlm("chr$chr.knockoff.txt", Xko)
