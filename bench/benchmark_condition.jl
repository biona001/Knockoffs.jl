"""
Benchmark: old vs new `condition` function for m=5 knockoffs, varying p.

Old approach: builds mp×mp covariance matrix → O(m²p²) memory + O(m³p³) Cholesky.
New approach: Jiaqi's trick with two p×p Cholesky factors → O(p²) memory + O(p³) Cholesky.

Run with:
  julia --project=bench bench/benchmark_condition.jl
"""

using Knockoffs
using BenchmarkTools
using BlockDiagonals, PositiveFactorizations
using LinearAlgebra
using Random
using Printf

# ──────────────────────────────────────────────────────────────────────────────
# Old implementation (verbatim from before issue-54 fix)
# ──────────────────────────────────────────────────────────────────────────────
function condition_old(
    X::AbstractMatrix,
    μ::AbstractVector,
    Σ::AbstractMatrix,
    S::AbstractMatrix;
    m::Number = 1
    )
    n, p = size(X)
    m = Int(m)
    Σinv  = inv(Symmetric(Σ))
    ΣinvS = Σinv * S
    C     = 2S - S*ΣinvS
    if m == 1
        L = cholesky(PositiveFactorizations.Positive, Symmetric(C)).L
        return X - (X .- μ') * ΣinvS + randn(n, p) * L
    end
    Σ̃  = repeat(C - S, m, m)
    Σ̃ += BlockDiagonal([S for _ in 1:m])
    μi = X - (X .- μ') * ΣinvS
    L  = cholesky(PositiveFactorizations.Positive, Symmetric(Σ̃)).L
    return repeat(μi, 1, m) + randn(n, m*p) * L
end

# New implementation is Knockoffs.condition (the public API)

# ──────────────────────────────────────────────────────────────────────────────
# Benchmark
# ──────────────────────────────────────────────────────────────────────────────
function run_benchmark(p_values; m=5, n=200, ρ=0.4)
    println("="^76)
    println("Benchmark: condition (old) vs condition (new)  [m=$m, n=$n, ρ=$ρ]")
    println()
    println("OLD: forms mp×mp covariance matrix  → O(m²p²) mem, O(m³p³) Cholesky")
    println("NEW: Jiaqi's trick, 2 p×p Choleskys → O(p²)   mem, O(p³)   Cholesky")
    println("="^76)
    @printf("%-8s  %-16s  %-16s  %-10s  %-14s  %-14s\n",
            "p", "old_time (ms)", "new_time (ms)", "speedup",
            "old_mem (MB)", "new_mem (MB)")
    println("-"^76)

    for p in p_values
        Random.seed!(42)
        Σ = [ρ^abs(i-j) for i in 1:p, j in 1:p]
        μ = zeros(p)
        X = randn(n, p)
        # Use a valid S that satisfies (m+1)/m * Σ - S ⪰ 0
        s_val = (m+1)/m * eigmin(Symmetric(Σ)) * 0.9
        S = Diagonal(fill(s_val, p))

        old_mem_est_mb = (m*p)^2 * 8 / 1e6  # mp×mp Float64 matrix in MB

        if old_mem_est_mb > 2048
            b_new = @benchmark Knockoffs.condition($X, $μ, $Σ, $S; m=$m) samples=5 evals=1
            t_new = minimum(b_new).time / 1e6
            mem_new = minimum(b_new).memory / 1e6
            @printf("%-8d  %-16s  %-16.1f  %-10s  %-14s  %-14.1f\n",
                    p, "OOM(>$(round(Int,old_mem_est_mb))MB)",
                    t_new, "N/A", ">$(round(Int,old_mem_est_mb))MB", mem_new)
        else
            b_old = @benchmark condition_old($X, $μ, $Σ, $S; m=$m) samples=5 evals=1
            b_new = @benchmark Knockoffs.condition($X, $μ, $Σ, $S; m=$m) samples=5 evals=1
            t_old = minimum(b_old).time / 1e6
            t_new = minimum(b_new).time / 1e6
            speedup  = t_old / t_new
            mem_old  = minimum(b_old).memory / 1e6
            mem_new  = minimum(b_new).memory / 1e6
            @printf("%-8d  %-16.1f  %-16.1f  %-10.2fx  %-14.1f  %-14.1f\n",
                    p, t_old, t_new, speedup, mem_old, mem_new)
        end
        GC.gc(); GC.gc()
    end
    println("="^76)
    println("Note: time = min over 5 samples; mem = peak allocation.")
end

run_benchmark([100, 200, 500, 1000, 2000], m=5)
