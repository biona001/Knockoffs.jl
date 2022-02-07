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
    r1, r2,  = sizehint!(Float64[], snps*(snps-1)>>1), sizehint!(Float64[], snps*(snps-1)>>1)
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