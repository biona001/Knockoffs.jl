
"""
    get_haplotype_transition_matrix(r, θ, α)

Compute transition matrices for the hidden Markov chains in haplotypes. 
This is 2 equations above eq8 in "Gene hunting with hidden Markov model knockoffs" by Sesia et al.

# Inputs
`r`: Length `p` vector, the "recombination rates"
`θ`: Size `p × K` matrix, `θ[j, k]` is probability that the allele is 1 for SNP `p` at `k`th haplotype motif
`α`: Size `p × K` matrix, probabilities that haplotype motifs succeed each other. Rows should sum to 1. 

# Output
`Q`: A `p`-dimensional vector of `K × K` matrices. `Q[:, :, j]` is the `j`th transition matrix. 
"""
function get_haplotype_transition_matrix(
    r::AbstractVecOrMat, # p × 1
    θ::AbstractMatrix,   # p × K
    α::AbstractMatrix,   # p × K
    )
    K = size(θ, 2)
    p = size(r, 1)
    Q = Array{Float64, 3}(undef, K, K, p)
    @inbounds for j in 1:p
        Qj = @view(Q[:, :, j])
        for k in 1:K, knew in 1:K
            Qj[k, knew] = (1 - exp(-r[j])) * α[j, knew] # note: Pr(j|i) = Q_{i,j} (i.e. rows of Q must sum to 1)
            if k == knew
                Qj[k, knew] += exp(-r[j])
            end
        end
    end
    return Q # Rows of Q should sum to 1
end

"""
    get_genotype_transition_matrix(r, θ, α, q, table)

Compute transition matrices for the hidden Markov chains in unphased genotypes. 
This is equation 9 of "Gene hunting with hidden Markov model knockoffs" by Sesia et al.

# Inputs
`r`: Length `p` vector, the "recombination rates"
`θ`: Size `p × K` matrix, `θ[j, k]` is probability that the allele is 1 for SNP `p` at `k`th haplotype motif
`α`: Size `p × K` matrix, probabilities that haplotype motifs succeed each other. Rows should sum to 1. 
`q`: Length `K` vector of initial probabilities
`table`: a `MarkovChainTable` that maps markov chain states k = 1, ..., K+(K+1)/2
    to haplotype pairs (ka, kb). 
"""
function get_genotype_transition_matrix(
    r::AbstractVecOrMat, # p × 1
    θ::AbstractMatrix,   # p × K
    α::AbstractMatrix,   # p × K
    q::AbstractVector,   # p × 1
    table::MarkovChainTable
    )
    # first compute haplotype transition matrix
    H = get_haplotype_transition_matrix(r, θ, α)
    K = size(H, 2)
    p = size(H, 3)
    statespace = (K * (K + 1)) >> 1

    # now compute genotype transition matrix
    Q = Array{Float64, 3}(undef, statespace, statespace, p)
    @inbounds for l in 1:statespace
        @view(Q[l, :, 1]) .= q
    end
    # for l in 1:statespace
    #     Q[:, :, 1] .= NaN # Q1 should never be used anywhere, so we fill it with NaN
    # end
    @inbounds for j in 2:p
        Qj, Hj = @view(Q[:, :, j]), @view(H[:, :, j])
        for (row, geno) in enumerate(table)
            for (col, geno_new) in enumerate(table)
                Qj[row, col] = Hj[geno.a, geno_new.a] * Hj[geno.b, geno_new.b] # note: Pr(j|i) = Q_{i,j} (i.e. rows of Q must sum to 1)
                if geno_new.a != geno_new.b
                    Qj[row, col] += Hj[geno.a, geno_new.b] * Hj[geno.b, geno_new.a]
                end
            end
        end
    end
    return Q #Rows of Q should sum to 1
end
# helper functions from HMMBase.jl
# istransmat(A::AbstractMatrix) =
#     issquare(A) && all([isprobvec(A[i, :]) for i = 1:size(A, 1)])
# @assert all(i -> istransmat(@view(Q[:, :, i])), 1:size(Q, 3))

function get_initial_probabilities(α::AbstractMatrix, table::MarkovChainTable)
    K = size(α, 2)
    statespace = (K * (K + 1)) >> 1
    q = zeros(statespace)
    α1 = α[1, :]
    @inbounds for (i, geno) in enumerate(table)
        ka, kb = geno.a, geno.b
        q[i] = (ka == kb ? abs2(α1[ka]) : 2 * α1[ka] * α1[kb])
    end
    @assert sum(q) ≈ 1 "initial probability should sum to 1!"
    return q
end

"""
    get_haplotype_emission_probabilities(θ::AbstractMatrix, j::Int, hj::Number, zj::Int)

Computes emission probabilities for unphased HMM. This is the equation above eq8 of 
"Gene hunting with hidden Markov model knockoffs" by Sesia et al.
"""
function get_haplotype_emission_probabilities(θ::AbstractMatrix, j::Int, hj::Number, zj::Int)
    if hj == 0
        return 1 - θ[j, zj]
    elseif hj == 1
        return θ[j, zj]
    else
        error("hj should be 0 or 1 but was $hj")
    end
end

"""
    get_genotype_emission_probabilities(θ::AbstractMatrix, xj::Number, ka::Int, kb::Int, j::Int)

Computes P(xj | k={ka,kb}, θ): emission probabilities for genotypes. This is eq 10 of 
"Gene hunting with hidden Markov model knockoffs" by Sesia et al.
"""
function get_genotype_emission_probabilities(θ::AbstractMatrix, xj::Number, ka::Int, kb::Int, j::Int)
    if xj == 0
        return (1 - θ[j, ka]) * (1 - θ[j, kb])
    elseif xj == 1
        return θ[j, ka] * (1 - θ[j, kb]) + θ[j, kb] * (1 - θ[j, ka])
    elseif xj == 2
        return θ[j, ka] * θ[j, kb]
    else
        error("xj should be 0, 1, or 2 but was $xj")
    end
end
# for j in 1:p, (k, (ka, kb)) in enumerate(with_replacement_combinations(1:K, 2))
#     p0 = get_genotype_emission_probabilities(θ, 0, ka, kb, j)
#     p1 = get_genotype_emission_probabilities(θ, 1, ka, kb, j)
#     p2 = get_genotype_emission_probabilities(θ, 2, ka, kb, j)
#     p0 + p1 + p2 ≈ 1 || error("shouldn't happen")
# end

"""
    form_emission_prob_matrix(a, θ, xi::AbstractVector)

# Inputs
+ `a`: `p × K` matrix with values estimated from fastPHASE (i.e. they called it the α parameter)
+ `θ`: `p × K` matrix with values estimated from fastPHASE
+ `xi`: Length `p` vector with sample `i`'s genotypes (entries 0, 1 or 2) 
"""
function form_emission_prob_matrix(a, θ, xi::AbstractVector, table::MarkovChainTable)
    p, K = size(a)
    statespace = (K * (K + 1)) >> 1
    f = zeros(p, statespace)
    @inbounds for j in 1:p, (k, geno) in enumerate(table)
        f[j, k] = get_genotype_emission_probabilities(θ, xi[j], geno.a, geno.b, j)
    end
    return f
end

"""
    forward_backward_sampling!(Z, X, Q, q, θ, ...)

Samples Z, the hidden states of a HMM, from observed sequence of unphased genotypes X.

# Inputs
`Z`: Length `p` vector of integers. This will store the sampled Markov states
`X`: Length `p` vector of genotypes (0, 1, or 2)
`Q`: `K × K × p` array. `Q[:, :, j]` is a `K × K` matrix of transition
    probabilities for `j`th state, i.e. Q[l, k, j] = P(X_{j} = k | X_{j - 1} = l).
    The first transition matrix is not used. 
`q`: Length `K` vector of initial probabilities
`θ`: The θ parameter estimated from fastPHASE

# Preallocated storage variables
`table`: a `MarkovChainTable` that maps markov chain states to haplotype 
    pairs (ka, kb)
`d`: Sampling distribution, probabilities in d.p are mutated
`α̂`: `p × K` scaled forward probability matrix, where 
    `α̂[j, k] = P(x_1,...,x_k, z_k) / P(x_1,...,x_k)`
`c`: normalizing constants, `c[k] = p(x_k | x_1,...,x_{k-1})`

# Reference
Algorithm 3 of "Gene hunting with hidden Markov model knockoffs" by Sesia et al
"""
function forward_backward_sampling!(
    Z::Vector{Int},
    X::Vector,
    Q::Array{T, 3},
    q::Vector{T},
    θ::AbstractMatrix,
    table::MarkovChainTable,
    d::Categorical,
    α̂::AbstractMatrix,
    c::AbstractVector,
    ) where T
    statespace, p = size(Q, 2), size(Q, 3)
    length(X) == p || error("length(X) not equal to p")
    fill!(c, 0)

    # (scaled) forward probabilities
    @inbounds for (k, geno) in enumerate(table)
        α̂[1, k] = q[k] * get_genotype_emission_probabilities(θ, X[1], geno.a, geno.b, 1)
        c[1] += α̂[1, k]
    end
    @view(α̂[1, :]) ./= c[1]
    @inbounds for j in 2:p
        mul!(@view(α̂[j, :]), Transpose(@view(Q[:, :, j])), @view(α̂[j - 1, :])) # note: Pr(j|i) = Q_{i,j} (i.e. rows of Q must sum to 1)
        for (k, geno) in enumerate(table)
            α̂[j, k] *= get_genotype_emission_probabilities(θ, X[j], geno.a, geno.b, j)
            c[j] += α̂[j, k]
        end
        @view(α̂[j, :]) ./= c[j]
    end

    # backwards sampling
    denom = sum(@view(α̂[p, :]))
    @inbounds for k in 1:statespace
        d.p[k] = α̂[p, k] / denom
    end
    Z[end] = rand(d)
    @inbounds for j in Iterators.reverse(1:p-1)
        denom = 0.0
        for k in 1:statespace
            denom += Q[k, Z[j + 1], j + 1] * α̂[j, k]
        end
        for zj in 1:statespace
            d.p[zj] = Q[zj, Z[j + 1], j + 1] * α̂[j, zj] / denom
        end
        @assert sum(d.p) ≈ 1 "forward_backward_sampling!: probability should sum to 1 but was $(sum(d.p))"
        Z[j] = rand(d)
    end

    return Z
end
# todo: how to test correctness?

function forward_backward_sampling(
    X::Vector,
    Q::Array{T, 3},
    q::Vector{T},
    θ::AbstractMatrix
    ) where T
    p = length(X)
    K = size(Q, 1)
    Z = zeros(Int, p)
    table = MarkovChainTable(K)
    statespace = statespace(table)
    d = Categorical([1 / statespace for _ in 1:statespace])
    α̂ = zeros(p, statespace) # scaled α, where α̂[j, k] = P(x_1,...,x_k, z_k) / P(x_1,...,x_k)
    c = zeros(p) # normalizing constants, c[k] = p(x_k | x_1,...,x_{k-1})
    forward_backward_sampling!(Z, X, Q, q, θ, table, d, α̂, c)
end

"""
    hmm_knockoff(plinkname, fastphase_outfile, T=10, datadir=pwd())

Generates HMM knockoffs from binary PLINK formatted files. This is done by
first running fastPHASE, then running Algorithm 2 of "Gene hunting with hidden
Markov model knockoffs" by Sesia, Sabatti, and Candes

# Input
+ `plinkname`: Binary PLINK file names without the `.bed/.bim/.fam` suffix. 

# Optional arguments
+ `T`: Number of initial starts used in fastPHASE EM algorithm (default = number
    of parallel julia threads available as measured by Threads.nthreads())
+ `datadir`: Full path to the PLINK and fastPHASE files (default = current directory)
+ `plink_outfile`: Output PLINK format name
+ `fastphase_outfile`: The output file name from fastPHASE's alpha, theta, r files
+ `args...`: Any parameter specified in `fastphase`

# Output
+ `plink_outfile.bed`: `n × p` knockoff genotypes
+ `plink_outfile.bim`: SNP mapping file. Knockoff have SNP names ending in ".k"
+ `plink_outfile.fam`: Sample mapping file, this is a copy of the original `plinkname.fam` file
+ `fastphase_outfile_rhat.txt`: averaged r hat file from fastPHASE
+ `fastphase_outfile_alphahat.txt`: averaged alpha hat file from fastPHASE
+ `fastphase_outfile_thetahat.txt`: averaged theta hat file from fastPHASE
"""
function hmm_knockoff(
    plinkname::AbstractString;
    T::Int = Threads.nthreads(),
    datadir::AbstractString = pwd(),
    plink_outfile::AbstractString = "knockoff",
    fastphase_outfile::AbstractString = "fastphase_out",
    outdir::AbstractString = datadir,
    args...
    )
    snpdata = SnpData(joinpath(datadir, plinkname))
    r, θ, α = fastphase_estim_param(snpdata; T=T, out=fastphase_outfile, args...)
    return hmm_knockoff(snpdata, r, θ, α, outdir=outdir, plink_outfile=plink_outfile)
end

function hmm_knockoff(
    plinkname::AbstractString,
    fastphase_outfile::AbstractString;
    T::Int = 1,
    datadir::AbstractString = pwd(),
    plink_outfile::AbstractString = "knockoff",
    outdir::AbstractString = datadir
    )
    snpdata = SnpData(joinpath(datadir, plinkname))
    r, θ, α = process_fastphase_output(datadir, T=T, extension=fastphase_outfile)
    return hmm_knockoff(snpdata, r, θ, α, plink_outfile=plink_outfile, outdir=outdir)
end

"""
    hmm_knockoff(snpdata, r, θ, α)

Generates knockoff of `snpdata` with loaded r, θ, α
"""
function hmm_knockoff(
    snpdata::SnpData,
    r::AbstractVecOrMat,
    θ::AbstractMatrix,
    α::AbstractMatrix;
    outdir = pwd(),
    plink_outfile::AbstractString = "knockoff",
    )
    Xfull = snpdata.snparray
    n, p = size(Xfull)
    K = size(θ, 2)
    statespace = (K * (K + 1)) >> 1
    table = MarkovChainTable(K)

    # get initial states (marginal distribution vector) and Markov transition matrices
    q = get_initial_probabilities(α, table)
    Q = get_genotype_transition_matrix(r, θ, α, q, table)

    # preallocated arrays
    X̃full = SnpArray(joinpath(outdir, plink_outfile * ".bed"), n, p)
    X = zeros(Float64, p)
    Z = zeros(Int, p)
    Z̃ = zeros(Int, p)
    X̃ = zeros(Int, p)
    N = zeros(p, statespace)
    d_K = Categorical([1 / statespace for _ in 1:statespace]) # for sampling markov chains (length statespace)
    d_3 = Categorical([1 / statespace for _ in 1:statespace]) # for sampling genotypes (length 3)
    α̂ = zeros(p, statespace) # scaled α, where α̂[j, k] = P(x_1,...,x_k, z_k) / P(x_1,...,x_k)
    c = zeros(p) # normalizing constants, c[k] = p(x_k | x_1,...,x_{k-1})

    @showprogress for i in 1:n
        # sample hidden states (algorithm 3 in Sesia et al)
        copyto!(X, @view(Xfull[i, :]))
        forward_backward_sampling!(Z, X, Q, q, θ, table, d_K, α̂, c)

        # sample knockoff of markov chain (algorithm 2 in Sesia et al)
        markov_knockoffs!(Z̃, Z, N, d_K, Q, q)

        # sample knockoffs of genotypes (eq 6 in Sesia et al)
        sample_markov_chain!(X̃, Z̃, table, θ, d_3)

        # save knockoff
        write_plink!(X̃full, X̃, i)
    end

    # copy .bim and .fam files
    new_bim = copy(snpdata.snp_info)
    for i in 1:p
        new_bim[i, :snpid] = new_bim[i, :snpid] * ".k"
    end
    CSV.write(joinpath(outdir, plink_outfile * ".bim"), new_bim, delim='\t', header=false)
    cp(snpdata.srcfam, joinpath(outdir, plink_outfile * ".fam"), force=true)

    return X̃full
end

function genotype_knockoffs(
    Z̃::AbstractVector,
    table::MarkovChainTable,
    θ::AbstractMatrix
    )
    p = length(Z̃)
    X̃ = zeros(eltype(Z̃), p)
    d = Categorical([1/3 for _ in 1:3])
    return sample_markov_chain!(X̃, Z̃, table, θ, d)
end

function sample_markov_chain!(
    X̃::AbstractVector,
    Z̃::AbstractVector,
    table::MarkovChainTable,
    θ::AbstractMatrix,
    d::Categorical # Categorical distribution from Distributions.jl
    )
    p = length(Z̃)
    @inbounds for j in 1:p
        a, b = index_to_pair(table, Z̃[j])
        d.p[1] = get_genotype_emission_probabilities(θ, 0, a, b, j)
        d.p[2] = get_genotype_emission_probabilities(θ, 1, a, b, j)
        d.p[3] = get_genotype_emission_probabilities(θ, 2, a, b, j)
        X̃[j] = rand(d)
    end
    X̃ .-= 1 # sampling d returns states 1~3, but genotypes are 0~2
    return X̃
end

function write_plink!(
    X̃full::SnpArray,
    X̃::AbstractVector,
    i::Int # sample index
    )
    p = length(X̃)
    @inbounds for j in 1:p
        if X̃[j] == 0
            X̃full[i, j] = 0x00
        elseif X̃[j] == 1
            X̃full[i, j] = 0x02
        elseif X̃[j] == 2
            X̃full[i, j] = 0x03
        else
            error("Genotypes should only be 0, 1, or 2 but got $(X̃[j])")
        end
    end
    return X̃full
end
