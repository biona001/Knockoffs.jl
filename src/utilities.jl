"""
    solve_SDP(Σ::AbstractMatrix)

Solves the SDP problem for fixed-X and model-X knockoffs. The optimization problem
is stated in equation 3.13 of "Panning for Gold: Model-X Knocko s for High-dimensional
Controlled Variable Selection" by Candes et al. 
"""
function solve_SDP(Σ::AbstractMatrix)
    svar = Variable(size(Σ, 1), Convex.Positive())
    add_constraint!(svar, svar ≤ 1)
    constraint = 2*Symmetric(Σ) - diagm(svar) in :SDP
    problem = maximize(sum(svar), constraint)
    solve!(problem, Hypatia.Optimizer; silent_solver=true)
    s = clamp.(evaluate(svar), 0, 1) # make sure s_j ∈ (0, 1)
    return s
end

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
    normalize_col!(X::AbstractVecOrMat)

Normalize each column of `X` so they sum to 1. 
"""
function normalize_col!(X::AbstractVecOrMat; center::Bool=false)
    @inbounds for x in eachcol(X)
        μi = center ? mean(x) : zero(eltype(X))
        xnorm = norm(x)
        @simd for i in eachindex(x)
            x[i] = (x[i] - μi) / xnorm
        end
    end
    return X
end
normalize_col(X) = normalize_col!(copy(X))

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
        copyto!(@view(Xfull[:, orig]), @view(X[:, i]))
        copyto!(@view(Xfull[:, knoc]), @view(X̃[:, i]))
        push!(original, orig)
        push!(knockoff, knoc)
    end
    return Xfull, original, knockoff
end

"""
    decorrelate_knockoffs(plinkfile, original, knockoff, α)

For each knockoff `x̃j`, we will randomly choose `α`% samples uniformly and set 
`x̃j[i] ~ binomail(2, ρj)` where `ρj ∈ [0, 1]` is the alternate allele frequency of SNP j.

# Inputs
+ `xdata`: A `SnpArrays.SnpData` storing original and knockoff genotypes from binary PLINK trios

# Optional inputs
+ `α`: A number between 0 and 1 specifying how many samples for each knockoff should be resampled (defualt 0.1)
+ `original`: Indices of original genotypes. `@view(xdata.snparray[:, original])` would be the original genotype (default: entries in 2nd column `bim` file not ending with `.k`)
+ `knockoff`: Indices of knockoff genotypes. `@view(xdata.snparray[:, knockoff])` would be the knockoff genotype (default: entries in 2nd column `bim` file ending with `.k`)
+ `outfile`: Output file name (defaults to "decorrelated_knockoffs")
+ `outdir`: Directory for storing output file (defaults to current directory)

# Output
+ `xnew`: A `n × 2p` `SnpArray` where the knockoffs `@view(xnew[:, knockoff])` have been decorrelated
"""
function decorrelate_knockoffs(
    xdata::SnpData;
    α::Number = 0.1,
    original::AbstractVector{Int} = findall(!endswith(".k"), xdata.snp_info[!, 2]),
    knockoff::AbstractVector{Int} = findall(endswith(".k"), xdata.snp_info[!, 2]),
    outfile = "decorrelated_knockoffs",
    outdir = pwd(),
    verbose::Bool = true
    )
    0.0 ≤ α ≤ 1.0 || error("decorrelate_knockoffs_maf: α must be between 0 and 1 but was $α")
    length(original) == length(knockoff) || 
        error("decorrelate_knockoffs_maf: original and knockoff SNPs have different numbers!")
    # import genotypes
    x = xdata.snparray
    n = size(x, 1)
    p = length(original)
    # display progress
    pmeter = verbose ? Progress(p, "Decorrelating...") : nothing
    # output array
    xnew = SnpArray(joinpath(outdir, outfile * ".bed"), n, 2p)
    # alternate allele freq for original genotypes
    alternate_allele_freq = maf_noflip(x)[original]
    # variables needed for sampling uniform genotypes
    d = Categorical([1/3 for i in 1:3])
    geno = [0x00, 0x02, 0x03]
    # loop over snps
    for j in 1:p
        # copy original snp
        copyto!(@view(xnew[:, original[j]]), @view(x[:, original[j]]))
        # change probabilities based on alternate allele freq
        alf = alternate_allele_freq[j]
        d.p[1] = (1 - alf)^2
        d.p[2] = 2 * (1 - alf) * alf
        d.p[3] = alf^2
        # randomly change α% of genotypes in knockoffs
        jj = knockoff[j]
        for i in 1:n
            if rand() < α
                xnew[i, jj] = geno[rand(d)] # uniformly sample 0, 1, 2
            else
                xnew[i, jj] = x[i, jj]
            end
        end
        # update progress
        verbose && next!(pmeter)
    end
    return xnew
end

# adapted from https://github.com/OpenMendel/SnpArrays.jl/blob/d63c0162338e98b74ccefce7440c95281ad1ff12/src/snparray.jl#L258
function maf_noflip!(out::AbstractVector{T}, s::AbstractSnpArray) where T <: AbstractFloat
    cc = SnpArrays._counts(s, 1)
    @inbounds for j in 1:size(s, 2)
        out[j] = (cc[3, j] + 2cc[4, j]) / 2(cc[1, j] + cc[3, j] + cc[4, j])
    end
    out
end
maf_noflip(s::AbstractSnpArray) = maf_noflip!(Vector{Float64}(undef, size(s, 2)), s)

function sample_DMC(q, Q; n=1)
    p = size(Q, 3)
    d = Categorical(q)
    X = zeros(Int, n, p)
    for i in 1:n
        X[i, 1] = rand(d)
        for j in 2:p
            d.p .= @view(Q[X[i, j-1], :, j])
            X[i, j] = rand(d)
        end
    end
    return X
end
