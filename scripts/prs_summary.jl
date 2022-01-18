#
# Summarize PRS simulation result
#
using DelimitedFiles
using Random
using DataFrames
using CSV
using Printf

# view one result
sim = 1
df = CSV.read("sim$sim/summary.txt", DataFrame)
# for sim in 1:20
#     df = CSV.read("sim$sim/summary.txt", DataFrame)
#     @show df
# end

# check cross validation result is the same for every run
# for fdr in [0.05, 0.1, 0.25, 0.5], sim in 1:100
#     f = "/scratch/users/bbchu/ukb/prs/Radj20_K50_s0/fdr$fdr/sim$sim/summary.txt"
#     g = "/scratch/users/bbchu/ukb/prs/Radj20_K50_s0/fdr$fdr/sim$sim/summary.txt"
#     if isfile(f) && isfile(g)
#         df = CSV.read(f, DataFrame)
#         dg = CSV.read(g, DataFrame)
#         if !all(df[!, 2] .≈ df[!, 2])
#             println("sim $sim fdr $fdr failed!")
#             println(df[!, 2])
#             println(dg[!, 2])
#         end
#     end
# end

# all results
df_sum = zeros(16, 5)
successes = 0
for sim in 1:100
    f = "sim$sim/summary.txt"
    if isfile(f)
        df = CSV.read(f, DataFrame)
        m = Matrix{Float64}(df[:, 2:end])
        any(isnan.(m)) && continue
        df_sum += m
        successes += 1
    else
        @warn "sim $sim failed!"
    end
end
df_mean = df_sum ./ successes
# get std
df_std = zeros(16, 5)
for sim in 1:100
    f = "sim$sim/summary.txt"
    if isfile(f)
        df = Matrix{Float64}(CSV.read(f, DataFrame)[:, 2:end])
        df_std += abs2.(df .- df_mean)
    end
end
df_std ./= successes
df_std .= sqrt.(df_std)
# save result 
writedlm("prs_population_R2_mean.txt", df_mean)
writedlm("prs_population_R2_std.txt", df_std)

# print latex table
column_name = ["African", "Asian", "Bangladeshi", "British", "Caribbean", "Chinese",
    "Indian", "Irish", "Pakistani", "White-asian", "White-black", "White",
    "non-zero betas", "beta selected", "TPP", "FDP"]

for i in 1:size(df_mean, 1)
    print(column_name[i])
    for j in 1:5
        μ = @sprintf "%0.3f" df_mean[i, j]
        σ = @sprintf "%0.3f" df_std[i, j]
        print(" & \$", μ, "\\pm", σ, "\$")
        # print(" & \$", round(df_mean[i, j], digits=3), "\\pm",
        #     round(df_std[i, j], digits=3), "\$")
    end
    print("\\\\ \n")
    i == 12 && println("\\hline\n\\hline")
end
df_mean
