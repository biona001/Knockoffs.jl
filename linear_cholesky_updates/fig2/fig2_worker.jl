#!/usr/bin/env julia

using CSV
using DataFrames
using Dates
using Knockoffs
using LinearAlgebra

BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "config.jl"))

const OUTDIR = joinpath(@__DIR__, "results")

function fig2_design()
    update_filter = get(ENV, "FIG2_UPDATE_STRATEGY", "")
    combos = NamedTuple[]
    for covariance in COVARIANCE_STRUCTURES, p in TIMING_P_VALUES,
        (method_name, base_method, parallel_method) in METHOD_CONFIGS,
        (update_strategy, nworkers, robust) in UPDATE_CONFIGS,
        rep in 1:FIG2_REPS
        !isempty(update_filter) && update_strategy != update_filter && continue
        method = solver_method(base_method, parallel_method, update_strategy)
        push!(combos, (; covariance, p, method_name, update_strategy, method, nworkers, robust, rep))
    end
    return combos
end

function should_skip(combo)
    skip_serial_p20000 = parse(Bool, get(ENV, "FIG2_SKIP_P20000_SERIAL", "false"))
    return combo.p == 20000 && !startswith(combo.update_strategy, "parallel") && skip_serial_p20000
end

function write_result(path, task_id, combo, seed, status; elapsed_sec=missing, error_message="")
    df = DataFrame(
        task_id=task_id,
        timestamp=string(now()),
        status=status,
        covariance=combo.covariance,
        p=combo.p,
        method_name=combo.method_name,
        update_strategy=combo.update_strategy,
        method=string(combo.method),
        nworkers=combo.nworkers,
        robust=combo.robust,
        rep=combo.rep,
        seed=seed,
        julia_threads=Threads.nthreads(),
        elapsed_sec=elapsed_sec,
        error_message=error_message,
    )
    CSV.write(path, df)
end

function main()
    task_id = parse_task_id(ARGS)
    design = fig2_design()
    1 <= task_id <= length(design) || error("Task id $task_id is outside 1:$(length(design)).")
    combo = design[task_id]

    seed = 20260509 + 100000task_id + combo.rep
    path = result_path(OUTDIR, "fig2", combo.update_strategy, "task$(task_id)")
    if should_skip(combo)
        write_result(path, task_id, combo, seed, "skipped_serial_p20000")
        println("Skipped p=20000 single-thread task. Set FIG2_SKIP_P20000_SERIAL=false to run it. Wrote $path")
        return nothing
    end

    Σ = covariance_matrix(combo.covariance, combo.p; seed)
    kwargs = solver_kwargs(combo.update_strategy, combo.nworkers, combo.robust)

    GC.gc()
    elapsed = missing
    try
        elapsed = @elapsed begin
            s = solve_s(Σ, combo.method; kwargs...)
            minimum(s) < -1e-6 && error("Solver returned a negative s entry: $(minimum(s)).")
        end
    catch err
        write_result(path, task_id, combo, seed, "failed"; elapsed_sec=elapsed, error_message=sprint(showerror, err))
        println("Failed task $task_id. Wrote $path")
        rethrow()
    end

    write_result(path, task_id, combo, seed, "completed"; elapsed_sec=elapsed)
    println("Wrote $path")
end

main()
