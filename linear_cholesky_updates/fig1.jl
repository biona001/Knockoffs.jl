using Knockoffs
using LinearAlgebra
using Plots
using Random
using Statistics

const SHADOW_THRESHOLD = 1e-8
const HEAT_FLOOR = 1e-12
const TITLE_FONT = 18
const GUIDE_FONT = 15
const TICK_FONT = 13
const ANNOTATION_FONT = 14

function sparse_cholesky_update_delta(Σ::Symmetric, k::Int; δ=0.2)
    L = cholesky(Σ).L |> Matrix
    L0 = copy(L)
    v = zeros(size(L, 1))
    v[k] = sqrt(δ)

    @inbounds for i in k:size(L, 1)
        c, s, r = LinearAlgebra.givensAlgorithm(L[i, i], v[i])
        L[i, i] = r
        for j in i+1:size(L, 1)
            Lji = L[j, i]
            vj = v[j]
            L[j, i] = c * Lji + s * vj
            v[j] = -s * Lji + c * vj
        end
        v[i] = 0.0
    end

    return abs.(L .- L0)
end

function influence_profile(ΔL::AbstractMatrix, k::Int)
    p = size(ΔL, 1)
    raw = [maximum(@view ΔL[i, 1:i]) for i in k:p]
    scale = max(first(raw), maximum(raw), eps())
    return raw ./ scale
end

function propagation_radius(profile::AbstractVector; threshold=SHADOW_THRESHOLD)
    tailmax = similar(profile)
    tailmax[end] = profile[end]
    for i in length(profile)-1:-1:1
        tailmax[i] = max(profile[i], tailmax[i + 1])
    end
    idx = findfirst(<=(threshold), tailmax)
    return isnothing(idx) ? length(profile) - 1 : idx - 1
end

function dense_equicorrelation(p::Int, ρ::Real)
    Σ = fill(Float64(ρ), p, p)
    Σ[diagind(Σ)] .= 1.0
    return Symmetric(Σ, :L)
end

function banded_correlation(p::Int; bandwidth=18, ρ=0.82)
    Σ = Matrix{Float64}(I, p, p)
    for j in 1:p, i in j+1:min(p, j + bandwidth)
        Σ[i, j] = ρ^abs(i - j)
        Σ[j, i] = Σ[i, j]
    end
    λ = eigmin(Symmetric(Σ))
    λ < 1e-4 && (Σ += (1e-4 - λ) * I)
    d = sqrt.(diag(Σ))
    Σ ./= d * d'
    return Symmetric(Σ, :L)
end

function local_er_blocks(p::Int; blocksize=80, ϕ=0.45, lb=0.35, ub=0.8, λmin=0.2)
    Σ = zeros(p, p)
    for lo in 1:blocksize:p
        hi = min(p, lo + blocksize - 1)
        idx = lo:hi
        B = Matrix{Float64}(I, length(idx), length(idx))
        for j in 1:length(idx), i in j+1:length(idx)
            if rand() < ϕ
                B[i, j] = rand((-1.0, 1.0)) * (lb + rand() * (ub - lb))
                B[j, i] = B[i, j]
            end
        end
        λ = eigmin(Symmetric(B))
        λ < λmin && (B += (λmin - λ) * I)
        d = sqrt.(diag(B))
        B ./= d * d'
        Σ[idx, idx] .= B
    end
    return Symmetric(Σ, :L)
end

function shadow_case_panel(Σ::Symmetric, k::Int, title::String; δ=0.2, threshold=SHADOW_THRESHOLD,
    show_ylabel=false)
    p = size(Σ, 1)
    ΔL = sparse_cholesky_update_delta(Σ, k; δ)
    profile = influence_profile(ΔL, k)
    radius = propagation_radius(profile; threshold)
    touched = count(>(threshold), profile) / length(profile)
    distances = 0:(p - k)

    window = k:p
    heat = log10.(max.(@view(ΔL[window, window]) ./ max(maximum(ΔL), eps()), HEAT_FLOOR))

    hm = heatmap(distances, distances, heat;
        title,
        xlabel="column distance from k",
        ylabel=show_ylabel ? "row distance from k" : "",
        yflip=true,
        aspect_ratio=:equal,
        color=:viridis,
        colorbar=false,
        clims=(log10(HEAT_FLOOR), 0),
        framestyle=:box,
        tickfontsize=TICK_FONT,
        guidefontsize=GUIDE_FONT,
        titlefontsize=TITLE_FONT,
        left_margin=show_ylabel ? 8Plots.mm : 2Plots.mm,
        bottom_margin=6Plots.mm)
    vline!(hm, [0]; color=:white, linewidth=1.3, label=false)
    hline!(hm, [0]; color=:white, linewidth=1.3, label=false)
    radius < last(distances) && hline!(hm, [radius]; color=:white, linestyle=:dash, linewidth=1.5, label=false)
    annotate!(hm, [(0.62last(distances), 0.12last(distances),
        text("R=$(radius), touched=$(round(100touched; digits=1))%", ANNOTATION_FONT, :white))])

    prof = plot(distances, max.(profile, HEAT_FLOOR);
        yscale=:log10,
        ylim=(HEAT_FLOOR, 1.2),
        xlabel="distance from updated coordinate",
        ylabel=show_ylabel ? "relative influence" : "",
        color=:black,
        linewidth=2,
        label=false,
        framestyle=:box,
        tickfontsize=TICK_FONT,
        guidefontsize=GUIDE_FONT,
        left_margin=show_ylabel ? 16Plots.mm : 2Plots.mm,
        bottom_margin=7Plots.mm)
    hline!(prof, [threshold]; color=:gray40, linestyle=:dot, linewidth=1.5, label=false)
    radius < last(distances) && vline!(prof, [radius]; color=:firebrick, linestyle=:dash, linewidth=1.6, label=false)
    annotate!(prof, [(0.72last(distances), 3threshold,
        text(radius < last(distances) ? "early stop plausible" : "no local cutoff", ANNOTATION_FONT, :firebrick))])

    return hm, prof
end

Random.seed!(2026)
p = 240
k = 40
block_groups = repeat(1:3, inner=80)

cases = [
    ("AR(1) decay", Symmetric(simulate_AR1(p; rho=0.85), :L)),
    ("Banded correlation", banded_correlation(p; bandwidth=18, ρ=0.82)),
    ("Block-local graph", local_er_blocks(p; blocksize=80, ϕ=0.45, lb=0.35, ub=0.8)),
    ("Dense equicorrelation", dense_equicorrelation(p, 0.98)),
]

panels = [shadow_case_panel(Σ, k, title; show_ylabel=(i == 1)) for (i, (title, Σ)) in enumerate(cases)]
fig = plot(first.(panels)..., last.(panels)...;
    layout=(2, 4),
    size=(2200, 1100),
    dpi=300,
    plot_title="Shadow-column locality after a sparse rank-1 Cholesky update at k = $k",
    plot_titlefontsize=24)

outdir = joinpath(@__DIR__, "figures")
mkpath(outdir)
savefig(fig, joinpath(outdir, "shadow_column_property.png"))
savefig(fig, joinpath(outdir, "shadow_column_property.pdf"))
