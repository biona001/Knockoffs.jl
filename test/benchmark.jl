using Knockoffs
using Random
using Distributions
using LinearAlgebra
using ToeplitzMatrices
using StatsBase
using DataFrames, CSV

function time_solvers(p; ρ = 0.4, tol=1e-6)
    # sample Σ
    Σ = Matrix(SymmetricToeplitz(ρ.^(0:(p-1)))) # true covariance matrix
    GC.gc();GC.gc();GC.gc();

    # get timings
    t1 = @elapsed Knockoffs.solve_equi(Σ)
    GC.gc();GC.gc();GC.gc();
    t2 = p > 1000 ? NaN : @elapsed Knockoffs.solve_SDP(Σ)
    GC.gc();GC.gc();GC.gc();
    t3 = @elapsed Knockoffs.solve_MVR(Σ, tol=tol)
    GC.gc();GC.gc();GC.gc();
    t4 = @elapsed Knockoffs.solve_max_entropy(Σ, tol=tol)
    GC.gc();GC.gc();GC.gc();
    t5 = p > 2000 ? NaN : @elapsed Knockoffs.solve_sdp_fast(Σ, tol=tol)
    GC.gc();GC.gc();GC.gc();

    return t1, t2, t3, t4, t5
end

function benchmark_solvers(p_dimentions::AbstractVector, tol::AbstractFloat)
    n = length(p_dimentions)
    equi_times, sdp_times, mvr_times, me_times, sdp_fast_times = 
        zeros(n), zeros(n), zeros(n), zeros(n), zeros(n)
    for (i, p) in enumerate(p_dimentions)
        println("Running p = $p")
        t1, t2, t3, t4, t5 = time_solvers(p, tol=tol)
        open("p$p", "w") do io
            println(io, "equi,sdp,mvr,me,sdp_fast")
            println(io, "$t1,$t2,$t3,$t4,$t5")
        end
        equi_times[i] = t1
        sdp_times[i] = t2
        mvr_times[i] = t3
        me_times[i] = t4
        sdp_fast_times[i] = t5
    end
    # put results into dataframe
    timings = DataFrame(dimension = p_dimentions, equi = equi_times, 
        SDP = sdp_times, MVR = mvr_times, ME = me_times, sdp_fast = sdp_fast_times)
    return timings
end

p_dimentions = [100, 500, 1000, 2500, 5000, 7500, 10000]
tol = 0.001 # default is 1e-6
timings = benchmark_solvers(p_dimentions, tol)
@show timings
