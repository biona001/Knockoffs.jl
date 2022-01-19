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
    T::Int = Threads.nthreads(), # number of different initial conditions for EM
    K::Int = 12, # number of clusters
    C::Int = 10, # number of EM iterations
    out::AbstractString = "fastphase_out",
    fastphase_infile::AbstractString = "fastphase.inp",
    outdir::AbstractString = pwd()
    )
    x = xdata.snparray
    n ≤ size(x, 1) || error("n must be smaller than the number of samples!")
    sampleid = xdata.person_info[!, :iid]
    # create input format for fastPHASE software
    p = size(x, 2)
    open(fastphase_infile, "w") do io
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
    # run fastPHASE, each thread runs a different start
    Threads.@threads for i in 1:T
        tmp_out = joinpath(outdir, "tmp$(i)")
        seed = i
        s = 0 # no simulation
        U = 0 # no simulation
        H = 0
        run(`./fastPHASE -T1 -S$seed -K$K -C$C -o$tmp_out -Pp -H$H -s$s -U$U $fastphase_infile`)
    end
    # aggregate results into final output
    r, θ, α = zeros(p), zeros(p, K), zeros(p, K)
    for i in 1:T
        rtmp, θtmp, αtmp = process_fastphase_output(outdir; T=1, extension="tmp$(i)")
        r .+= rtmp
        θ .+= θtmp
        α .+= αtmp
    end
    r ./= T
    θ ./= T
    α ./= T
    α ./= sum(α, dims = 2) # normalize rows to sum to 1
    # save averaged results
    writedlm(out * "_rhat.txt", r, ' ')
    writedlm(out * "_thetahat.txt", θ, ' ')
    writedlm(out * "_alphahat.txt", α, ' ')
    # clean up
    for i in 1:T
        rm(joinpath(outdir, "tmp$(i)_rhat.txt"), force=true)
        rm(joinpath(outdir, "tmp$(i)_thetahat.txt"), force=true)
        rm(joinpath(outdir, "tmp$(i)_alphahat.txt"), force=true)
        rm(joinpath(outdir, "tmp$(i)_hapsfrommodel.out"), force=true)
        rm(joinpath(outdir, "tmp$(i)_hapguess_switch.out"), force=true)
        rm(joinpath(outdir, "tmp$(i)_origchars"), force=true)
        rm(joinpath(outdir, "tmp$(i)_sampledHgivG.txt"), force=true)
    end
    return r, θ, α
end

"""
    process_fastphase_output(datadir, T; [extension])

Reads r, θ, α into memory, averaging over `T` simulations. 

# Inputs
`datadir`: Directory that the `rhat.txt`, `thetahat.txt`, `alphahat.txt` files are stored
`T`: Number of different runs excuted by fastPHASE. This is the number of different 
    initial conditions used for EM algorithm. All files `rhat.txt`, `thetahat.txt`,
    `alphahat.txt` would therefore have `T × p` rows
`extension`: Name in front of the output fastPHASE files. E.g. Use `extension=out_`
    if your output files are `out_rhat.txt`, `out_thetahat.txt`, `out_alphahat.txt`
"""
function process_fastphase_output(
    datadir::AbstractString;
    T::Int=1,
    extension="out"
    )
    # read full data 
    rfile = joinpath(datadir, "$(extension)_rhat.txt") # T*p × 1
    θfile = joinpath(datadir, "$(extension)_thetahat.txt") # T*p × K
    αfile = joinpath(datadir, "$(extension)_alphahat.txt") # T*p × K
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
