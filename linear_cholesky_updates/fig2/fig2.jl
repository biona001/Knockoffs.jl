#!/usr/bin/env julia

using CSV
using DataFrames
using Plots
using Statistics

include(joinpath(@__DIR__, "config.jl"))

const RESULT_DIR = joinpath(@__DIR__, "results")
const FIGURE_DIR = joinpath(@__DIR__, "..", "figures")

function read_results()
    files = filter(f -> endswith(f, ".csv"), readdir(RESULT_DIR; join=true))
    isempty(files) && error("No CSV files found in $RESULT_DIR.")
    return vcat((CSV.read(f, DataFrame) for f in files)...; cols=:union)
end

function strategy_label(strategy)
    strategy == "standard" && return "standard"
    strategy == "early" && return "early stop"
    return replace(strategy, "parallel" => "parallel ")
end

function main()
    df = read_results()
    if "status" in names(df)
        df = df[df.status .== "completed", :]
    end
    summary = combine(groupby(df, [:covariance, :p, :method_name, :update_strategy, :nworkers]),
        :elapsed_sec => mean => :mean_sec,
        :elapsed_sec => std => :sd_sec,
        nrow => :nruns)
    summary.strategy_label = strategy_label.(summary.update_strategy)

    mkpath(FIGURE_DIR)
    CSV.write(joinpath(@__DIR__, "fig2_summary.csv"), summary)

    palette = Dict(
        "standard" => :black,
        "early" => :dodgerblue3,
        "parallel2" => :darkorange2,
        "parallel4" => :seagreen3,
        "parallel8" => :firebrick2,
        "parallel16" => :mediumpurple3,
        "parallel32" => :deeppink3,
    )
    plots = []
    for method_name in first.(METHOD_CONFIGS), covariance in COVARIANCE_STRUCTURES
        sub = summary[(summary.covariance .== covariance) .& (summary.method_name .== method_name), :]
        p = plot(
            xlabel="p",
            ylabel="mean solve time (sec)",
            title="$method_name, $covariance",
            xscale=:log10,
            yscale=:log10,
            legend=:outerright,
            framestyle=:box,
            grid=:y,
            dpi=300,
        )
        for strategy in first.(UPDATE_CONFIGS)
            cur = sort(sub[sub.update_strategy .== strategy, :], :p)
            isempty(cur) && continue
            plot!(p, cur.p, cur.mean_sec;
                marker=:circle,
                linewidth=2,
                label=strategy_label(strategy),
                color=palette[strategy])
        end
        push!(plots, p)
    end
    fig = plot(plots...; layout=(3, 3), size=(1700, 1200), bottom_margin=7Plots.mm, left_margin=7Plots.mm)
    savefig(fig, joinpath(FIGURE_DIR, "fig2_timing.png"))
    savefig(fig, joinpath(FIGURE_DIR, "fig2_timing.pdf"))
end

main()
