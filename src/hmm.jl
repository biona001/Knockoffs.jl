
"""
get_haplotype_transition_matrix(r, θ, α)

Compute transition matrices for the hidden Markov chains in haplotypes. 
This is 2 equations above eq8 in "Gene hunting with hidden Markov model knockoffs" by Sesia et al.

# Inputs
`r`: Length `p` vector, the "recombination rates"
`θ`: Size `p × K` matrix, `θ[j, k]` is probability that the allele is 1 for SNP `p` at `k`th haplotype motif
`α`: Size `p × K` matrix, probabilities that haplotype motifs succeed each other. Rows should sum to 1. 
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
    get_genotype_transition_matrix(H::AbstractArray{T, 3})

Compute transition matrices for the hidden Markov chains in unphased genotypes. 
This is equation 9 of "Gene hunting with hidden Markov model knockoffs" by Sesia et al.

# Inputs
`H`: A `p`-dimensional vector of `K × K` matrices. `H[:, :, j]` is the `j`th transition matrix. 
"""
function get_genotype_transition_matrix(H::AbstractArray{T, 3}, table::MarkovChainTable) where T <: AbstractFloat
    K = size(H, 2)
    p = size(H, 3)
    statespace = (K * (K + 1)) >> 1
    Q = Array{Float64, 3}(undef, statespace, statespace, p)
    for j in 1:p
        Qj, Hj = @view(Q[:, :, j]), @view(H[:, :, j])
        @inbounds for (row, geno) in enumerate(table)
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
    for j in 1:p, (k, geno) in enumerate(table)
        f[j, k] = get_genotype_emission_probabilities(θ, xi[j], geno.a, geno.b, j)
    end
    return f
end

"""
    forward_backward_sampling(x::SnpArray)

Samples Z, the hidden states of a HMM, from observed sequence of unphased genotypes X.

# Inputs
`Z`: Length `p` vector of integers. This will store the sampled Markov states
`xi`: Length `p` vector of genotypes (0, 1, or 2)
`Q`: `K × K × p` array. `Q[:, :, j]` is a `K × K` matrix of transition
    probabilities for `j`th state, i.e. Q[l, k, j] = P(X_{j} = k | X_{j - 1} = l).
    The first transition matrix is not used. 
`q`: Length `p` vector of initial probabilities
`θ`: The θ parameter estimated from fastPHASE
`table`: a `MarkovChainTable` that maps markov chain states to haplotype 
    pairs (ka, kb). 

# Reference
Algorithm 3 of "Gene hunting with hidden Markov model knockoffs" by Sesia et al
"""
function forward_backward_sampling!(
    Z::Vector{Int},
    xi::Vector,
    d::Categorical,
    Q::Array{T, 3},
    q::Vector{T},
    θ::AbstractMatrix,
    table::MarkovChainTable,
    ) where T
    statespace, p = size(Q, 2), size(Q, 3)
    length(xi) == p || error("length(xi) not equal to p")

    # (scaled) forward probabilities
    α̂ = zeros(p, statespace) # scaled α, where α̂[j, k] = P(x_1,...,x_k, z_k) / P(x_1,...,x_k)
    c = zeros(p) # normalizing constants, c[k] = p(x_k | x_1,...,x_{k-1})
    for (k, geno) in enumerate(table)
        α̂[1, k] = q[k] * get_genotype_emission_probabilities(θ, xi[1], geno.a, geno.b, 1)
        c[1] += α̂[1, k]
    end
    α̂[1, :] ./= c[1]
    for j in 2:p
        mul!(@view(α̂[j, :]), Transpose(@view(Q[:, :, j])), @view(α̂[j - 1, :])) # note: Pr(j|i) = Q_{i,j} (i.e. rows of Q must sum to 1)
        for (k, geno) in enumerate(table)
            α̂[j, k] *= get_genotype_emission_probabilities(θ, xi[j], geno.a, geno.b, j)
            c[j] += α̂[j, k]
        end
        α̂[j, :] ./= c[j]
    end

    # backwards sampling
    denom = sum(@view(α̂[p, :]))
    for k in 1:statespace
        d.p[k] = α̂[p, k] / denom
    end
    Z[end] = rand(d)
    for j in Iterators.reverse(1:p-1)
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
    xi::Vector,
    Q::Array{T, 3},
    q::Vector{T},
    θ::AbstractMatrix,
    table::MarkovChainTable
    ) where T
    Z = zeros(Int, p)
    statespace = statespace(table)
    d = Categorical([1 / statespace for _ in 1:statespace])
    forward_backward_sampling!(Z, xi, d, Q, q, θ, table)
end

"""
    hmm_knockoff(plinkname, fastphase_outfile, T=10, datadir=pwd())

Main entry point of generating HMM knockoffs from binary PLINK formatted files.

# Input
+ `plinkname`: Binary PLINK file names without the `.bed/.bim/.fam` suffix. 
+ `fastphase_outfile`: The output file name from fastPHASE's alpha, theta, r files
    (e.g. input "x_" if the files are called "x_thetahat.txt", "x_rhat.txt"...etc)

# Optional arguments
+ `T`: Number of initial starts used in fastPHASE EM algorithm (default = 10)
+ `datadir`: Full path to the PLINK and fastPHASE files (default = current directory)
+ `outfile`: Output PLINK format name

# Output
+ `outfile.bed`: `n × 2p` genotypes, including the original genotypes and the knockoffs
+ `outfile.bim`: SNP mapping file. Knockoff have SNP names ending in ".k"
+ `outfile.fam`: Sample mapping file, this is a copy of the original `plinkname.fam` file
"""
function hmm_knockoff(
    plinkname::AbstractString,
    fastphase_outfile::AbstractString;
    T::Int = 10,
    datadir::AbstractString = pwd(),
    outfile::AbstractString = "knockoff"
    )
    snpdata = SnpData(joinpath(datadir, plinkname))
    Xfull = snpdata.snparray
    n, p = size(Xfull)

    # get r, α, θ estimated by fastPHASE
    r, θ, α = process_fastphase_output(datadir, T, extension=fastphase_outfile)
    K = size(θ, 2)
    statespace = (K * (K + 1)) >> 1
    table = MarkovChainTable(K)

    # transition matrices, initial states (marginal distribution vector), and emission probabilities
    H = get_haplotype_transition_matrix(r, θ, α)
    Q = get_genotype_transition_matrix(H, table)
    q = get_initial_probabilities(α, table)

    # preallocated arrays
    # full_knockoff = SnpArray(outfile * ".bed", n, 2p)
    full_knockoff = zeros(Int, n, p)
    X = zeros(Float64, p)
    Z = zeros(Int, p)
    Z̃ = zeros(Int, p)
    X̃ = zeros(Int, p)
    N = zeros(p, statespace)
    d_K = Categorical([1 / statespace for _ in 1:statespace]) # for sampling markov chains (length statespace)
    d_3 = Categorical([1 / statespace for _ in 1:statespace]) # for sampling genotypes (length 3)

    @showprogress for i in 1:n
        # sample hidden states (algorithm 3 in Sesia et al)
        copyto!(X, @view(Xfull[i, :]))
        forward_backward_sampling!(Z, X, d_K, Q, q, θ, table)

        # sample knockoff of markov chain (algorithm 2 in Sesia et al)
        markov_knockoffs!(Z̃, Z, N, d_K, Q, q)

        # sample knockoffs of genotypes (eq 6 in Sesia et al)
        genotype_knockoffs!(X̃, Z̃, table, θ, d_3)

        # save knockoff
        full_knockoff[i, :] .= X̃
    end

    return full_knockoff
end

function genotype_knockoffs(
    Z̃::AbstractVector,
    table::MarkovChainTable,
    θ::AbstractMatrix
    )
    p = length(Z̃)
    X̃ = zeros(eltype(Z̃), p)
    d = Categorical([1/3 for _ in 1:3])
    return genotype_knockoffs!(X̃, Z̃, table, θ, d)
end

function genotype_knockoffs!(
    X̃::AbstractVector,
    Z̃::AbstractVector,
    table::MarkovChainTable,
    θ::AbstractMatrix,
    d::Categorical # Categorical distribution from Distributions.jl
    )
    p = length(Z̃)
    for j in 1:p
        a, b = index_to_pair(table, Z̃[j])
        d.p[1] = get_genotype_emission_probabilities(θ, 0, a, b, j)
        d.p[2] = get_genotype_emission_probabilities(θ, 1, a, b, j)
        d.p[3] = get_genotype_emission_probabilities(θ, 2, a, b, j)
        X̃[j] = rand(d)
    end
    X̃ .-= 1 # sampling d returns states 1~3, but genotypes are 0~2
    return X̃
end

# function write_plink!(
#     full_knockoff::SnpArray,
#     X::AbstractVector,
#     X̃::AbstractVector,
#     j::Int
#     )
#     n = size(full_knockoff, 1)
#     x1, x2 = rand() < 0.5 ? (X, X̃) : (X̃, X) # decide whether the original or the knockoff will come first
#     col1, col2 = 2j - 1, 2j
#     for i in 1:n
#         if x1[i]
#             full_knockoff[i, col1] = 
#     end
# end