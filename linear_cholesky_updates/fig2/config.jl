using BlockDiagonals
using LinearAlgebra
using Random

const COVARIANCE_STRUCTURES = split(get(ENV, "FIG2_COVARIANCE_STRUCTURES", "AR1,ER,block,stress"), ",")
const TIMING_P_VALUES = [500, 1000, 2000, 5000, 10000, 20000]
const FIG2_REPS = 5
const FIG2_WORKERS = parse(Int, get(ENV, "FIG2_WORKERS", "8"))

const METHOD_CONFIGS = [
    ("ME", :maxent, :maxent_fast),
    ("MVR", :mvr, :mvr_fast),
    ("SDP", :sdp, :sdp_fast),
]

const UPDATE_CONFIGS = [
    ("serial", 1, false, 0, :buffered),
    ("serial_robust", 1, true, 0, :buffered),
    ("local0", FIG2_WORKERS, false, 0, :block),
    ("buffer16", FIG2_WORKERS, false, 16, :buffered),
    ("buffer32", FIG2_WORKERS, false, 32, :buffered),
    ("buffer64", FIG2_WORKERS, false, 64, :buffered),
]

function parse_task_id(args)
    if !isempty(args)
        return parse(Int, args[1])
    end
    task = get(ENV, "SLURM_ARRAY_TASK_ID", "")
    isempty(task) && error("Pass a task id argument or set SLURM_ARRAY_TASK_ID.")
    return parse(Int, task)
end

function block_size_for(p::Int; max_cores::Int=32)
    nblocks = max(4max_cores, cld(p, 16))
    return max(2, cld(p, nblocks))
end

function block_ranges(p::Int, blocksize::Int)
    return [lo:min(p, lo + blocksize - 1) for lo in 1:blocksize:p]
end

function block_ar1_covariance(p::Int; rho=0.85, blocksize=block_size_for(p))
    blocks = Matrix{Float64}[]
    for idx in block_ranges(p, blocksize)
        q = length(idx)
        push!(blocks, [rho^abs(i - j) for i in 1:q, j in 1:q])
    end
    return Symmetric(Matrix(BlockDiagonal(blocks)))
end

function local_er_covariance(p::Int; blocksize=block_size_for(p), phi=0.25, lb=0.20, ub=0.60, lambda_min=0.20)
    blocks = Matrix{Float64}[]
    for idx in block_ranges(p, blocksize)
        q = length(idx)
        B = Matrix{Float64}(I, q, q)
        for j in 1:q, i in (j + 1):q
            if rand() < phi
                B[i, j] = rand((-1.0, 1.0)) * (lb + rand() * (ub - lb))
                B[j, i] = B[i, j]
            end
        end
        lambda = eigmin(Symmetric(B))
        lambda < lambda_min && (B += (lambda_min - lambda) * I)
        d = sqrt.(diag(B))
        B ./= d * d'
        push!(blocks, B)
    end
    return Symmetric(Matrix(BlockDiagonal(blocks)))
end

function local_block_covariance(p::Int; blocksize=block_size_for(p), rho=0.65)
    blocks = Matrix{Float64}[]
    for idx in block_ranges(p, blocksize)
        q = length(idx)
        B = (1 - rho) * Matrix{Float64}(I, q, q) .+ rho
        push!(blocks, B)
    end
    return Symmetric(Matrix(BlockDiagonal(blocks)))
end

function stress_block_covariance(p::Int; nblocks=8, rho=0.8, gamma=0.9)
    nblocks = max(1, min(nblocks, p))
    edges = round.(Int, range(0, p; length=nblocks + 1))
    groups = Vector{Int}(undef, p)
    @inbounds for b in 1:nblocks
        groups[(edges[b] + 1):edges[b + 1]] .= b
    end
    return Symmetric(simulate_block_covariance(groups, rho, gamma))
end

function covariance_matrix(kind::AbstractString, p::Int; seed::Int=1)
    Random.seed!(seed)
    kind == "AR1" && return block_ar1_covariance(p)
    kind == "ER" && return local_er_covariance(p)
    kind == "block" && return local_block_covariance(p)
    kind == "stress" && return stress_block_covariance(
        p;
        nblocks=parse(Int, get(ENV, "STRESS_BLOCKS", "8")),
        rho=parse(Float64, get(ENV, "STRESS_RHO", "0.8")),
        gamma=parse(Float64, get(ENV, "STRESS_GAMMA", "0.9")),
    )
    error("Unknown covariance structure: $kind")
end

function solver_method(base_method::Symbol, parallel_method::Symbol, update_strategy::AbstractString)
    return update_strategy in ("local0", "buffer16", "buffer32", "buffer64") ? parallel_method : base_method
end

function solver_kwargs(update_strategy::AbstractString, nworkers::Int, robust::Bool, buffer_size::Int, local_window_mode::Symbol)
    kwargs = Dict{Symbol, Any}(
        :niter => parse(Int, get(ENV, "NITER", "100")),
        :tol => parse(Float64, get(ENV, "TOL", "1e-6")),
        :verbose => parse(Bool, get(ENV, "VERBOSE_SOLVER", "false")),
    )
    if update_strategy in ("local0", "buffer16", "buffer32", "buffer64")
        kwargs[:nworkers] = nworkers
        kwargs[:feature_order] = nothing
        kwargs[:window_corr_tol] = parse(Float64, get(ENV, "WINDOW_CORR_TOL", "0.8"))
        kwargs[:buffer_size] = buffer_size
        kwargs[:local_window_mode] = local_window_mode
        kwargs[:factor_check] = parse(Bool, get(ENV, "FACTOR_CHECK", "false"))
    else
        kwargs[:robust] = robust
    end
    return kwargs
end

function result_path(outdir::AbstractString, parts...)
    mkpath(outdir)
    return joinpath(outdir, join(string.(parts), "_") * ".csv")
end
