#
# Summarize PRS simulation result
#
using DelimitedFiles
using Random
using DataFrames
using CSV
using Printf

# view one result
f = "sim1/summary.txt"
df = CSV.read(f, DataFrame)

# all results
df_sum = zeros(15, 5)
successes = 0
for sim in 1:100
    f = "sim$sim/summary.txt"
    if isfile(f)
        df = CSV.read(f, DataFrame)
        df_sum += Matrix{Float64}(df[:, 2:end])
        successes += 1
    else
        @warn "sim $sim failed!"
    end
end
df_mean = df_sum ./ successes
# get std
df_std = zeros(15, 5)
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
    "beta selected", "TPP", "FDP"]

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
