function predict(genetic_beta, nongenetic_beta, x::SnpArray, Z, intercept, test_idx)
    n = length(test_idx)
    ŷ = zeros(n)
    storage=zeros(n)
    Ztest = @view(Z[test_idx, :])
    @showprogress for snp in findall(!iszero, genetic_beta)
        copyto!(storage, @view(x[test_idx, snp]))
        storage .= 2 .- storage # estimated beta is for allele 1, but copyto! copies number of allele 2
        @inbounds @simd for i in 1:n
            ŷ[i] += storage[i] * genetic_beta[snp]
        end
    end
    @inbounds for j in findall(!iszero, nongenetic_beta)
        @simd for i in 1:n
            ŷ[i] += Ztest[i, j] * nongenetic_beta[j]
        end
    end
    ŷ .+= intercept
    return ŷ
end
