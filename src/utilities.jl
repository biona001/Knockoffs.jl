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
        push!(r2, abs(cor(snp1, snp2)))
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