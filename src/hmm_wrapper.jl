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
