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

function rapid(
    rapid_exe::AbstractString,
    vcffile::AbstractString,
    mapfile::AbstractString,
    min_length::Int,
    outfolder::AbstractString,
    window_size::Int,
    r::Int,
    s::Int
    )
    isdir(outfolder) || mkdir(outfolder)
    run(`$rapid_exe -i $vcffile -g $mapfile -d $min_length -o $outfolder -w $window_size -r $r -s $s`)
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
    cmd = pipeline(`$snpknock2_exe $args`)
    @info "snpknock2 command:\n$cmd\n"
    @info "Output directory: $(pwd() * "/knockoffs")\n"
    run(cmd)
end
