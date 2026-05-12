#!/usr/bin/env julia

using CSV
using DataFrames
using Dates
using Distributions
using GLMNet
using Knockoffs
using LinearAlgebra
using Random
using Statistics

BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "config.jl"))

const OUTDIR = joinpath(@__DIR__, "results")
const P = parse(Int, get(ENV, "FIG3_P", "5000"))
const N = parse(Int, get(ENV, "FIG3_N", "800"))
const K = parse(Int, get(ENV, "FIG3_K", "50"))
const FIG3_REPS = parse(Int, get(ENV, "FIG3_REPS", "50"))
const EFFECT_SIZE = parse(Float64, get(ENV, "FIG3_EFFECT_SIZE", "0.45"))

function fig3_design()
    combos = NamedTuple[]
    for covariance in COVARIANCE_STRUCTURES,
        (method_name, base_method, parallel_method) in METHOD_CONFIGS,
        (update_strategy, nworkers, robust) in UPDATE_CONFIGS,
        rep in 1:FIG3_REPS
        method = solver_method(base_method, parallel_method, update_strategy)
        push!(combos, (; covariance, method_name, update_strategy, method, nworkers, robust, rep))
    end
    return combos
end

function simulate_design(Σ::Symmetric, n::Int)
    X = randn(n, size(Σ, 1)) * cholesky(Σ).U
    X .-= mean(X, dims=1)
    X ./= std(X, dims=1)
    return X
end

function power_and_fdr(selected, causal)
    selected = Set(selected)
    causal = Set(causal)
    discoveries = length(selected)
    true_positives = length(intersect(selected, causal))
    power = true_positives / length(causal)
    fdr = discoveries == 0 ? 0.0 : (discoveries - true_positives) / discoveries
    return power, fdr, discoveries
end

function write_failure(path, task_id, combo, seed, message)
    df = DataFrame(
        task_id=task_id,
        timestamp=string(now()),
        status="failed",
        covariance=combo.covariance,
        p=P,
        n=N,
        k=K,
        method_name=combo.method_name,
        update_strategy=combo.update_strategy,
        method=string(combo.method),
        nworkers=combo.nworkers,
        robust=combo.robust,
        rep=combo.rep,
        seed=seed,
        target_fdr=missing,
        power=missing,
        fdr=missing,
        discoveries=missing,
        ko_elapsed_sec=missing,
        fit_elapsed_sec=missing,
        error_message=message,
    )
    CSV.write(path, df)
end

function main()
    task_id = parse_task_id(ARGS)
    design = fig3_design()
    1 <= task_id <= length(design) || error("Task id $task_id is outside 1:$(length(design)).")
    combo = design[task_id]

    seed = 20260510 + 100000task_id + combo.rep
    path = result_path(OUTDIR, "fig3", "task$(task_id)")
    Random.seed!(seed)
    rows = DataFrame()
    try
        Σ = covariance_matrix(combo.covariance, P; seed)
        X = simulate_design(Σ, N)
        causal = sort(sample(1:P, K; replace=false))
        β = zeros(P)
        β[causal] .= EFFECT_SIZE .* rand((-1.0, 1.0), K)
        y = X * β + randn(N)
        y .-= mean(y)

        kwargs = solver_kwargs(combo.update_strategy, combo.nworkers, combo.robust)
        ko_elapsed = @elapsed ko = modelX_gaussian_knockoffs(X, combo.method, zeros(P), Matrix(Σ); kwargs...)
        fit_elapsed = @elapsed fit = fit_lasso(y, ko; filter_method=:knockoff, debias=nothing)

        for q in FDR_GRID
            selected = select_variables(fit, q)
            pow, fdr, discoveries = power_and_fdr(selected, causal)
            push!(rows, (
                task_id=task_id,
                timestamp=string(now()),
                status="completed",
                covariance=combo.covariance,
                p=P,
                n=N,
                k=K,
                method_name=combo.method_name,
                update_strategy=combo.update_strategy,
                method=string(combo.method),
                nworkers=combo.nworkers,
                robust=combo.robust,
                rep=combo.rep,
                seed=seed,
                target_fdr=q,
                power=pow,
                fdr=fdr,
                discoveries=discoveries,
                ko_elapsed_sec=ko_elapsed,
                fit_elapsed_sec=fit_elapsed,
                error_message="",
            ))
        end
    catch err
        write_failure(path, task_id, combo, seed, sprint(showerror, err))
        println("Failed task $task_id. Wrote $path")
        rethrow()
    end

    CSV.write(path, rows)
    println("Wrote $path")
end

main()
