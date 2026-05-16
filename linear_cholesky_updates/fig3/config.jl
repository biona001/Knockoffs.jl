using LinearAlgebra
using Random

const COVARIANCE_STRUCTURES = split(get(ENV, "FIG3_COVARIANCE_STRUCTURES", "AR1,ER,block,stress"), ",")
const FDR_GRID = collect(0.01:0.01:0.2)
const FIG3_WORKERS = parse(Int, get(ENV, "FIG3_WORKERS", "32"))
const LD_DOMAINS = 100

const METHOD_CONFIGS = [
    ("ME", :maxent, :maxent_fast),
    ("MVR", :mvr, :mvr_fast),
    ("SDP", :sdp, :sdp_fast),
]

const UPDATE_CONFIGS = [
    ("serial", 1, false, 0, :buffered),
    ("serial_robust", 1, true, 0, :buffered),
    ("local0", FIG3_WORKERS, false, 0, :block),
    ("buffer16", FIG3_WORKERS, false, 16, :buffered),
    ("buffer32", FIG3_WORKERS, false, 32, :buffered),
    ("buffer64", FIG3_WORKERS, false, 64, :buffered),
]

function parse_task_id(args)
    if !isempty(args)
        return parse(Int, args[1])
    end
    task = get(ENV, "SLURM_ARRAY_TASK_ID", "")
    isempty(task) && error("Pass a task id argument or set SLURM_ARRAY_TASK_ID.")
    return parse(Int, task)
end

function domain_size_for(p::Int; ndomains::Int=LD_DOMAINS)
    return max(16, cld(p, ndomains))
end

function domain_ranges(p::Int, domain_size::Int)
    return [lo:min(p, lo + domain_size - 1) for lo in 1:domain_size:p]
end

function background_ar1!(Σ::Matrix{Float64}, weight::Float64, rho::Float64)
    p = size(Σ, 1)
    @inbounds for j in 1:p, i in j:p
        val = weight * rho^(i - j)
        Σ[i, j] += val
        i == j || (Σ[j, i] += val)
    end
    return Σ
end

function local_ar1_covariance(
    p::Int;
    domain_size=domain_size_for(p),
    rho=0.85,
    background_weight=0.02,
    background_rho=0.80
    )
    Σ = zeros(Float64, p, p)
    background_ar1!(Σ, background_weight, background_rho)
    local_weight = 1 - background_weight
    for idx in domain_ranges(p, domain_size)
        for jj in eachindex(idx), ii in jj:length(idx)
            i = idx[ii]
            j = idx[jj]
            Σ[i, j] += local_weight * rho^(ii - jj)
            i == j || (Σ[j, i] += local_weight * rho^(ii - jj))
        end
    end
    return Symmetric(Σ)
end

function local_er_covariance(
    p::Int;
    domain_size=domain_size_for(p),
    phi=0.08,
    lb=0.10,
    ub=0.35,
    lambda_min=0.20,
    background_weight=0.02,
    background_rho=0.80
    )
    Σ = zeros(Float64, p, p)
    background_ar1!(Σ, background_weight, background_rho)
    local_weight = 1 - background_weight
    for idx in domain_ranges(p, domain_size)
        q = length(idx)
        B = Matrix{Float64}(I, q, q)
        for j in 1:q, i in (j + 1):q
            if rand() < phi
                B[i, j] = rand((0.6, 1.0)) * (lb + rand() * (ub - lb)) * exp(-abs(i - j) / 20)
                B[j, i] = B[i, j]
            end
        end
        lambda = eigmin(Symmetric(B))
        lambda < lambda_min && (B += (lambda_min - lambda) * I)
        d = sqrt.(diag(B))
        B ./= d * d'
        @inbounds for j in 1:q, i in j:q
            Σ[idx[i], idx[j]] += local_weight * B[i, j]
            i == j || (Σ[idx[j], idx[i]] += local_weight * B[i, j])
        end
    end
    return Symmetric(Σ)
end

function local_block_covariance(
    p::Int;
    domain_size=domain_size_for(p),
    rho=0.65,
    background_weight=0.02,
    background_rho=0.80
    )
    Σ = zeros(Float64, p, p)
    background_ar1!(Σ, background_weight, background_rho)
    local_weight = 1 - background_weight
    for idx in domain_ranges(p, domain_size)
        q = length(idx)
        B = (1 - rho) * Matrix{Float64}(I, q, q) .+ rho
        @inbounds for j in 1:q, i in j:q
            Σ[idx[i], idx[j]] += local_weight * B[i, j]
            i == j || (Σ[idx[j], idx[i]] += local_weight * B[i, j])
        end
    end
    return Symmetric(Σ)
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
    kind == "AR1" && return local_ar1_covariance(p)
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
        kwargs[:min_window_size] = parse(Int, get(ENV, "MIN_WINDOW_SIZE", "150"))
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
