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
    summary = combine(groupby(df, [:covariance, :method_name, :update_strategy, :target_fdr]),
        :power => mean => :power,
        :fdr => mean => :fdr,
        nrow => :nruns)
    summary.strategy_label = strategy_label.(summary.update_strategy)

    mkpath(FIGURE_DIR)
    CSV.write(joinpath(@__DIR__, "fig3_summary.csv"), summary)

    palette = Dict("ME" => :black, "MVR" => :dodgerblue3, "SDP" => :firebrick2)
    linestyles = Dict("standard" => :solid, "early" => :dash, "parallel32" => :dot)
    panels = []
    for covariance in COVARIANCE_STRUCTURES
        sub = summary[summary.covariance .== covariance, :]
        p_power = plot(xlabel="target FDR", ylabel="power", title="$covariance power",
            ylim=(0, 1), legend=:bottomright, framestyle=:box, dpi=300)
        p_fdr = plot(xlabel="target FDR", ylabel="empirical FDR", title="$covariance FDR",
            xlim=(0, 0.2), ylim=(0, 0.25), legend=:topleft, framestyle=:box, dpi=300)
        plot!(p_fdr, [0, 0.2], [0, 0.2]; linestyle=:dash, color=:gray40, label=false, linewidth=1.5)
        for method_name in first.(METHOD_CONFIGS), strategy in first.(UPDATE_CONFIGS)
            cur = sort(sub[(sub.method_name .== method_name) .& (sub.update_strategy .== strategy), :], :target_fdr)
            isempty(cur) && continue
            label = "$method_name, $(strategy_label(strategy))"
            plot!(p_power, cur.target_fdr, cur.power;
                marker=:circle, linewidth=2, color=palette[method_name], linestyle=linestyles[strategy], label=label)
            plot!(p_fdr, cur.target_fdr, cur.fdr;
                marker=:circle, linewidth=2, color=palette[method_name], linestyle=linestyles[strategy], label=label)
        end
        push!(panels, p_power, p_fdr)
    end
    fig = plot(panels...; layout=(2, 3), size=(1500, 850), bottom_margin=7Plots.mm, left_margin=7Plots.mm)
    savefig(fig, joinpath(FIGURE_DIR, "fig3_power_fdr.png"))
    savefig(fig, joinpath(FIGURE_DIR, "fig3_power_fdr.pdf"))
end

main()
