function partition(
    rscriptPath::AbstractString,
    plinkfile::AbstractString,
    mapfile::AbstractString,
    qc_variants::AbstractString,
    outfile::AbstractString
    )
    bimfile = plinkfile * ".bim"
    run(`Rscript --vanilla $rscriptPath $mapfile $bimfile $qc_variants $outfile`)
end

"""
    rapid(rapid_exe, vcffile, mapfile, d, outfolder, w, r, s, [a])

Wrapper for the RaPID program. 

# Inputs
- `rapid_exe`: Full path to the `RaPID_v.1.7` executable file
- `vcffile`: Phased VCF file name
- `mapfile`: Map file name
- `d`: Actual Minimum IBD length in cM
- `outfolder`: Output folder name
- `w`: Number of SNPs in a window for sub-sampling
- `r`: Number of runs
- `s`: Minimum number of successes to consider a hit

# Optional Inputs
- `a`: If `true`, ignore MAFs. By default (`a=false`) the sites are selected at random weighted
    by their MAFs.
"""
function rapid(
    rapid_exe::AbstractString,
    vcffile::AbstractString,
    mapfile::AbstractString,
    d::Number,
    outfolder::AbstractString,
    w::Int,
    r::Int,
    s::Int;
    a::Bool = false
    )
    d ≥ 0 || error("d must be ≥ 0 but was $d.")
    s ≥ 0 || error("s must be ≥ 1 but was $s.")
    isdir(outfolder) || mkdir(outfolder)
    args = String[]
    push!(args, "-i", vcffile)
    push!(args, "-g", mapfile)
    push!(args, "-d", string(d))
    push!(args, "-o", outfolder)
    push!(args, "-w", string(w))
    push!(args, "-r", string(r))
    push!(args, "-s", string(s))
    a && push!(args, "-a")
    cmd = `$rapid_exe $args`
    @info "RaPID command:\n$cmd\n"
    @info "Output directory: $outfolder\n"
    run(cmd)
end

function snpknock2(
    snpknock2_exe::AbstractString,
    bgenfile::AbstractString,
    sample_qc::AbstractString,
    variant_qc::AbstractString,
    mapfile::AbstractString,
    partfile::AbstractString,
    ibdfile::AbstractString,
    K::Int,
    cluster_size_min::Int,
    cluster_size_max::Int, 
    hmm_rho::Number, # recombination scale
    hmm_lambda::AbstractFloat,
    windows::Int,
    n_threads::Int,
    seed::Int,
    compute_references::Bool,
    generate_knockoffs::Bool,
    outfile::AbstractString
    )
    args = String[]
    isdir("knockoffs") || mkdir("knockoffs")
    push!(args, "--bgen", bgenfile)
    push!(args, "--keep", sample_qc)
    push!(args, "--extract", variant_qc)
    push!(args, "--map", mapfile)
    push!(args, "--part", partfile)
    push!(args, "--ibd", ibdfile)
    push!(args, "--K", string(K))
    push!(args, "--cluster_size_min", string(cluster_size_min))
    push!(args, "--cluster_size_max", string(cluster_size_max))
    push!(args, "--hmm-rho", string(hmm_rho))
    push!(args, "--hmm-lambda", string(hmm_lambda))
    push!(args, "--windows", string(windows))
    push!(args, "--n_threads", string(n_threads))
    push!(args, "--seed", string(seed))
    compute_references && push!(args, "--compute-references")
    generate_knockoffs && push!(args, "--generate-knockoffs")
    push!(args, "--out", "./knockoffs/$outfile")
    cmd = `$snpknock2_exe $args`
    @info "snpknock2 command:\n$cmd\n"
    @info "Output directory: $(pwd() * "/knockoffs")\n"
    run(cmd)
end

"""
    decorrelate_knockoffs(plinkfile, original, knockoff)

If a SNP and its knockoffs has correlation > `r2_threshold`, this function is randomly
change some entries in the knockoff variable, i.e. decorrelate the knockoff. 
"""
function decorrelate_knockoffs(
    plinkfile::AbstractString,
    original::Vector{Int},
    knockoff::Vector{Int};
    outfile = "decorrelated_knockoffs",
    outdir = pwd(),
    r2_threshold = 0.95
    )
    x = SnpArray(plinkfile * ".bed")
    n, p = size(x)
    p >> 1 == length(original) == length(knockoff) || error("Number of SNPs should be the same")
    xnew = SnpArray(joinpath(outdir, outfile * ".bed"), n, p)
    swap_probability = 1 - r2_threshold
    # calculate correlation of knockoffs with their original snps
    r2, snp1, snp2 = sizehint!(Float64[], p >> 1), zeros(n), zeros(n)
    for i in 1:p>>1
        copyto!(snp1, @view(x[:, original[i]]), center=true, scale=true)
        copyto!(snp2, @view(x[:, knockoff[i]]), center=true, scale=true)
        push!(r2, abs(cor(snp1, snp2)))
    end
    # loop over snps
    for j in 1:p>>1
        # copy original snp
        copyto!(@view(xnew[:, original[j]]), @view(x[:, original[j]]))
        # copy knockoffs
        jj = knockoff[j]
        if r2[j] ≤ r2_threshold
            copyto!(@view(xnew[:, jj]), @view(x[:, jj]))
        else
            # loop over each sample
            for i in 1:n
                # We change the an entry of knockoff with probability `swap_probability`
                # if xij is 0 or 2, set it equal to 1. If xij is 1, let it equal 0 or 2 randomly
                if rand() < swap_probability
                    if x[i, jj] == 0x01 || x[i, jj] == 0x03
                        xnew[i, jj] = 0x02
                    else
                        xnew[i, jj] = (rand() < 0.5 ? 0x02 : 0x03)
                    end
                else
                    xnew[i, jj] = x[i, jj]
                end
            end
        end
    end
    # copy bim and fam files
    cp(plinkfile * ".bim", joinpath(outdir, outfile * ".bim"), force=true)
    cp(plinkfile * ".fam", joinpath(outdir, outfile * ".fam"), force=true)
    return xnew
end

function fastphase(
    xdata::SnpData;
    n::Int = size(xdata.snparray, 1), # number of samples used to fit HMM
    T::Int = 10, # number of different initial conditions for EM
    K::Int = 10, # number of clusters
    C::Int = 25, # number of EM iterations
    out::AbstractString = "out"
    )
    x = xdata.snparray
    n ≤ size(x, 1) || error("n must be smaller than the number of samples!")
    sampleid = xdata.person_info[!, :iid]
    # create input format for fastPHASE software
    p = size(x, 2)
    open("fastphase.inp", "w") do io
        println(io, n)
        println(io, p)
        for i in 1:n
            println(io, "ID ", sampleid[i])
            # print genotypes for each sample on 2 lines. The "1" for heterozygous
            # genotypes will always go on the 1st line.
            for j in 1:p
                if x[i, j] == 0x00
                    print(io, 0)
                elseif x[i, j] == 0x02 || x[i, j] == 0x03
                    print(io, 1)
                else
                    print(io, '?')
                end
            end
            print(io, "\n")
            for j in 1:p
                if x[i, j] == 0x00 || x[i, j] == 0x02
                    print(io, 0)
                elseif x[i, j] == 0x03
                    print(io, 1)
                else
                    print(io, '?')
                end
            end
            print(io, "\n")
        end
    end
    run(`./fastPHASE -T$T -K$K -C$C -o$(out) -Pp fastphase.inp`)
    return nothing
end

"""
    process_fastphase_output(datadir::AbstractString, T::Int)

Reads fastPHASE results and performs averaging over `T` runs

# Inputs
+ `datadir`: Directory that stores fastPHASE's results (`out_rhat.txt`, `out_thetahat.txt`, `out_alphahat.txt`)
+ `T`: the number of different initial conditions for EM used in fastPHASE
"""
function process_fastphase_output(datadir::AbstractString, T::Int; extension="out_")
    # read full data 
    rfile = joinpath(datadir, "$(extension)rhat.txt") # T*p × 1
    θfile = joinpath(datadir, "$(extension)thetahat.txt") # T*p × K
    αfile = joinpath(datadir, "$(extension)alphahat.txt") # T*p × K
    isfile(rfile) && isfile(θfile) && isfile(αfile) || error("Files not found!")
    r_full = readdlm(rfile, comments=true, comment_char = '>', header=false)
    θ_full = readdlm(θfile, comments=true, comment_char = '>', header=false)
    α_full = readdlm(θfile, comments=true, comment_char = '>', header=false)

    # compute averages across T simulations as suggested by Scheet et al 2006
    p = Int(size(r_full, 1) / T)
    K = size(θ_full, 2)
    r, θ, α = zeros(p), zeros(p, K), zeros(p, K)
    for i in 1:T
        rows = (i - 1) * p + 1:p*i
        r .+= @view(r_full[rows])
        θ .+= @view(θ_full[rows, :])
        α .+= @view(α_full[rows, :])
    end
    r ./= T
    θ ./= T
    α ./= T
    α ./= sum(α, dims = 2) # normalize rows to sum to 1
    return r, θ, α
end

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
    Q = [Matrix{Float64}(undef, K, K) for _ in 1:p]
    @inbounds for j in 1:p
        Qj = Q[j]
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
    get_genotype_transition_matrix(H::Vector{Matrix}, K::Int)

Compute transition matrices for the hidden Markov chains in unphased genotypes. 
This is equation 9 of "Gene hunting with hidden Markov model knockoffs" by Sesia et al.

# Inputs
`H`: Length `p` vector of haplotype transition matrices, each with dimension `K × K`
"""
function get_genotype_transition_matrix(H::Vector{Matrix{T}}) where T <: AbstractFloat
    p = length(H)
    K = size(H[1], 1)
    statespace = (K * (K + 1)) >> 1
    Q = [Matrix{Float64}(undef, statespace, statespace) for _ in 1:p]
    @showprogress for j in 1:p
        Qj, Hj = Q[j], H[j]
        @inbounds for (row, (ka, kb)) in enumerate(with_replacement_combinations(1:K, 2))
            for (col, (ka_new, kb_new)) in enumerate(with_replacement_combinations(1:K, 2))
                Qj[row, col] = Hj[ka, ka_new] * Hj[kb, kb_new]
                if ka_new != kb_new
                    Qj[row, col] += Hj[ka, kb_new] * Hj[kb, ka_new]
                end
            end
        end
    end
    return Q #Rows of Q should sum to 1
end

function get_initial_probabilities(α::AbstractMatrix)
    K = size(α, 2)
    statespace = (K * (K + 1)) >> 1
    q = zeros(statespace)
    α1 = α[1, :]
    @inbounds for (i, (ka, kb)) in enumerate(with_replacement_combinations(1:K, 2))
        q[i] = (ka == kb ? abs2(α1[ka]) : 2 * α1[ka] * α1[kb])
    end
    @assert sum(q) ≈ 1 "initial probability should sum to 1!"
    return q
end

# function get_emission_probabilities(θ::AbstractMatrix, Z::Vector{Int})
#     p, K = size(θ)
#     f = zeros(p, 3) # f[j, 1] = Pr(x_j = 0), f[j, 2] = Pr(x_j = 1), f[j, 3] = Pr(x_j = 3)
#     for j in 1:p
#         ka, kb = Z[j]
#         f[j, 1] = (1 - θ[j, ka]) * (1 - θ[j, kb])
#         f[j, 2] = 
#         f[j, 3] = 
#     end
# end

"""
    forward_backward_sampling()

Samples Z, the hidden states of a HMM, from observed sequence of unphased genotypes X.
This is algorithm 3 of "Gene hunting with hidden Markov model knockoffs" by Sesia et al
"""
function forward_backward_sampling()
    # get r, α, θ estimated by fastPHASE
    r, θ, α = process_fastphase_output(datadir, T, extension="ukb_chr10_n1000_")

    # form transition matrices, initial state and emission probabilities
    H = get_haplotype_transition_matrix(r, θ, α)
    Q = get_genotype_transition_matrix(H)
    q = get_initial_probabilities(α)

    # forward probabilities
    # a = 

    # Z = sample_markov_chain(Q, q)

    # f = get_emission_probabilities(θ, Z)
end
