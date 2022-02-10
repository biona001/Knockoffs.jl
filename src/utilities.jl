"""
    compare_correlation()

Computes correlation between X[:, i] and X̃[:, i] for each i.
"""
function compare_correlation(X::SnpArray, X̃::SnpArray)
    n, p = size(X)
    n == size(X̃, 1) || error("Number of samples does not match")
    p == size(X̃, 2) || error("Number of SNPs does not match")
    r2, snp1, snp2 = sizehint!(Float64[], p), zeros(n), zeros(n)
    for i in 1:p
        copyto!(snp1, @view(X̃[:, i]), center=true, scale=true, impute=true)
        copyto!(snp2, @view(X[:, i]), center=true, scale=true, impute=true)
        push!(r2, cor(snp1, snp2))
    end
    return r2
end

function compare_correlation(
    original_plink::AbstractString,
    knockoff_plink::AbstractString
    )
    X = SnpArray(original_plink)
    X̃ = SnpArray(knockoff_plink)
    return compare_correlation(X, X̃)
end

"""
    compare_pairwise_correlation(X::SnpArray, X̃::SnpArray, snps::Int = size(X, 2))

Computes and returns

+ `r1`: correlation between X[:, i] and X[:, j]
+ `r2`: correlation between X[:, i] and X̃[:, i]
"""
function compare_pairwise_correlation(X::SnpArray, X̃::SnpArray; snps::Int = size(X, 2))
    n, p = size(X)
    n == size(X̃, 1) || error("Number of samples does not match")
    p == size(X̃, 2) || error("Number of SNPs does not match")
    snps ≤ p || error("snps = $snps exceeds total number of SNPs, which was $p")
    r1, r2 = sizehint!(Float64[], snps*(snps-1)>>1), sizehint!(Float64[], snps*(snps-1)>>1)
    snp1, snp2 = zeros(n), zeros(n)
    for i in 1:snps, j in 1:i
        copyto!(snp1, @view(X[:, i]), center=true, scale=true, impute=true)
        copyto!(snp2, @view(X[:, j]), center=true, scale=true, impute=true)
        push!(r1, cor(snp1, snp2))
        copyto!(snp2, @view(X̃[:, j]), center=true, scale=true, impute=true)
        push!(r2, cor(snp1, snp2))
    end
    return r1, r2
end

function compare_pairwise_correlation(
    original_plink::AbstractString,
    knockoff_plink::AbstractString,
    snps::Int = countlines(original_plink * ".bim")
    )
    X = SnpArray(original_plink)
    X̃ = SnpArray(knockoff_plink)
    return compare_pairwise_correlation(X, X̃; snps=snps)
end


"""
    standardize!(z::AbstractVecOrMat)

Standardizes each column of `z` to mean 0 and variance 1. Make sure you 
do not standardize the intercept. 
"""
function standardize!(z::AbstractVecOrMat)
    n, q = size(z)
    μ = _mean(z)
    σ = _std(z, μ)

    @inbounds for j in 1:q
        @simd for i in 1:n
            z[i, j] = (z[i, j] - μ[j]) * σ[j]
        end
    end
end
# from https://github.com/OpenMendel/MendelIHT.jl/blob/master/src/utilities.jl

function _mean(z)
    n, q = size(z)
    μ = zeros(q)
    @inbounds for j in 1:q
        tmp = 0.0
        @simd for i in 1:n
            tmp += z[i, j]
        end
        μ[j] = tmp / n
    end
    return μ
end

function _std(z, μ)
    n, q = size(z)
    σ = zeros(q)

    @inbounds for j in 1:q
        @simd for i in 1:n
            σ[j] += (z[i, j] - μ[j])^2
        end
        σ[j] = 1.0 / sqrt(σ[j] / (n - 1))
    end
    return σ
end

"""
    merge_knockoffs_with_original(xdata, x̃data; des::AbstractString = "knockoff")

Interleaves the original PLINK genotypes with its knockoff into a single PLINK file.

# Inputs
+ `xdata`: A `SnpData` or `Array{T, 2}` of original covariates, or a `String` that points to the original PLINK file (without .bed/bim/fam suffix)
+ `x̃data`: A `SnpData` or `Array{T, 2}` of knockoff covariates, or a `String` that points to the knockoff PLINK file (without .bed/bim/fam suffix)
+ `des`: A `String` for output PLINK file name (without .bed/bim/fam suffix)

# Outputs
+ `xfull`: A `n × 2p` array of original and knockoff genotypes. 
+ `original`: Indices of original genotypes. `original[i]` is the column number for the `i`th SNP. 
+ `knockoff`: Indices of knockoff genotypes. `knockoff[i]` is the column number for the `i`th SNP. 
"""
function merge_knockoffs_with_original(
    xdata::SnpData,
    x̃data::SnpData;
    des::AbstractString = "merged.knockoff"
    )
    n, p = size(xdata)
    x, x̃ = xdata.snparray, x̃data.snparray
    xfull = SnpArray(des * ".bed", n, 2p)
    original, knockoff = sizehint!(Int[], p), sizehint!(Int[], p)
    for i in 1:p
        # decide which of original or knockoff SNP comes first
        orig, knoc = rand() < 0.5 ? (2i - 1, 2i) : (2i, 2i - 1)
        copyto!(@view(xfull[:, orig]), @view(x[:, i]))
        copyto!(@view(xfull[:, knoc]), @view(x̃[:, i]))
        push!(original, orig)
        push!(knockoff, knoc)
    end
    # copy fam files
    cp(xdata.srcfam, des * ".fam", force=true)
    # copy bim file, knockoff SNPs end in ".k"
    new_bim = copy(xdata.snp_info)
    empty!(new_bim)
    for i in 1:p
        if original[i] < knockoff[i]
            push!(new_bim, xdata.snp_info[i, :])
            push!(new_bim, x̃data.snp_info[i, :])
        else
            push!(new_bim, x̃data.snp_info[i, :])
            push!(new_bim, xdata.snp_info[i, :])
        end
    end
    CSV.write(des * ".bim", new_bim, delim='\t', header=false)
    return xfull, original, knockoff
end

function merge_knockoffs_with_original(
    x_path::AbstractString,
    x̃_path::AbstractString;
    des::AbstractString = "merged.knockoff"
    )
    xdata = SnpData(x_path)
    x̃data = SnpData(x̃_path)
    return merge_knockoffs_with_original(xdata, x̃data, des=des)
end

function merge_knockoffs_with_original(
    X::AbstractMatrix{T},
    X̃::AbstractMatrix{T}
    ) where T
    n, p = size(X)
    Xfull = zeros(n, 2p)
    original, knockoff = sizehint!(Int[], p), sizehint!(Int[], p)
    for i in 1:p
        # decide which of original or knockoff SNP comes first
        orig, knoc = rand() < 0.5 ? (2i - 1, 2i) : (2i, 2i - 1)
        copyto!(@view(Xfull[:, knoc]), @view(X[:, i]))
        copyto!(@view(Xfull[:, orig]), @view(X̃[:, i]))
        push!(original, orig)
        push!(knockoff, knoc)
    end
    return Xfull, original, knockoff
end
