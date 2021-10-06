using Statistics

"""
Find the `nsnps` most significant SNPs for mvPLINK in simulation `sim`.
"""
function get_top_mvPLINK_SNPs(set::Int, sim::Int, nsnps::Int)
    n, p = 5340, 24523
    dir = "set$set/sim$sim/"
    
    # read mvPLINK result
    mvplink_df = CSV.read(dir * "plink.mqfam.total", DataFrame, delim=' ', ignorerepeated=true)

    # get pvalues, possibly accounting for "NA"s
    if eltype(mvplink_df[!, :P]) == Float64
        pval = mvplink_df[!, :P]
    else
        mvplink_df[findall(x -> x == "NA", mvplink_df[!, :P]), :P] .= "1.0"
        pval = parse.(Float64, mvplink_df[!, :P])
    end
    perm = sortperm(pval)
    return perm[1:nsnps]
end

"""
Find the position of the `nsnps` most significant SNPs for GEMMA in simulation `sim`.
"""
function get_top_GEMMA_SNP_ids(set::Int, sim::Int, nsnps::Int)
    dir = "set$set/sim$sim/"
    gemma_df = CSV.read(dir * "gemma.sim$sim.assoc.txt", DataFrame)
    pval_wald = gemma_df[!, :p_wald]
    perm = sortperm(pval_wald)
    return perm[1:nsnps]
end

"""
Find significant SNPs return by IHT in simulation `sim`.
"""
function get_IHT_SNPs(set::Int, sim::Int)
    dir = "set$set/sim$sim/"
    iht_β1 = vec(readdlm(dir * "iht_beta1.txt"))
    iht_β2 = vec(readdlm(dir * "iht_beta2.txt"))
    detected_snps = findall(!iszero, iht_β1) ∪ findall(!iszero, iht_β2)
    return unique(detected_snps)
end

"""
Get positions for the truly causal SNPs in simulation `sim`. 
"""
function get_true_SNPs(set::Int, sim::Int)
    dir = "set$set/sim$sim/"
    trueB = readdlm(dir * "trueb.txt")
    causal_snps = unique([x[1] for x in findall(!iszero, trueB)])
    return causal_snps
end

"""
Get causal SNPs' position in GEMMA's result for simulation `sim`. Note gemma have snp filtering.
"""
function get_gemma_causal_snp_pos(set::Int, sim::Int)
    nfbc = SnpData("NFBC.qc.imputeBy0.chr.1")
    dir = "set$set/sim$sim/"
    trueB = readdlm(dir * "trueb.txt")
    causal_snps = unique([x[1] for x in findall(!iszero, trueB)])
    causal_snp_rsID = nfbc.snp_info.snpid[causal_snps]
    gemma_df = CSV.read(dir * "gemma.sim$sim.assoc.txt", DataFrame)
    gemma_snps = gemma_df[!, :rs]
    causal_snp_idx = convert(Vector{Int}, indexin(causal_snp_rsID, gemma_snps))
    
    # also need IHT's selected SNPs
    iht_snps_rsID = nfbc.snp_info.snpid[get_IHT_SNPs(sim)]
    iht_snps_idx = convert(Vector{Int}, indexin(iht_snps_rsID, gemma_snps))
    
    # also need SNP positions in GEMMA dataframe
    gemma2nfbc_idx = convert(Vector{Int}, indexin(gemma_snps, nfbc.snp_info.snpid))
    gemma_snp_pos = Vector{Int}(undef, size(gemma_df, 1))
    for i in 1:size(gemma_df, 1)
        gemma_snp_pos[i] = nfbc.snp_info.position[gemma2nfbc_idx[i]]
    end
    insertcols!(gemma_df, size(gemma_df, 2) + 1, :pos => gemma_snp_pos)
    
    return gemma_df, causal_snp_idx, iht_snps_idx
end

"""
Imports gemma p-values, causal SNPs, and IHT selected SNP, and plot manhattan plot using MendelPlots.jl
"""
function plot_gemma_manhattan(sim::Int)
    # GEMMA causal SNPs
    gemma_df, causal_snps, iht_snps = get_gemma_causal_snp_pos(sim)
    rename!(gemma_df, [:p_wald => :pval, :rs => :snpid])
    gemma_df[findall(x -> x < 1e-50, gemma_df[!, :pval]), :pval] .= 1e-50
    empty_col = ["" for i in 1:size(gemma_df, 1)]
    insertcols!(gemma_df, size(gemma_df, 2) + 1, :empty_col => empty_col)

    manhattan(gemma_df, outfile = "NFBCsim/manhattan_gemma_sim$sim.png",
        annotateinds = causal_snps, annotateinds2 = iht_snps,
        annotatevar=:empty_col, titles="GEMMA simulation $sim")
    display("image/png", read("NFBCsim/manhattan_gemma_sim$sim.png"))
end

"""
Imports mvPLINK p-values, causal SNPs, and IHT selected SNP, and plot manhattan plot using MendelPlots.jl
"""
function plot_mvPLINK_manhattan(sim::Int)
    # mvPLINK
    filename = "NFBCsim/sim$sim/plink.mqfam.total"
    mvplink_df = CSV.read(filename, DataFrame, delim=' ', ignorerepeated=true)
    if eltype(mvplink_df[!, :P]) == Float64
        pval = mvplink_df[!, :P]
    else
        mvplink_df[findall(x -> x == "NA", mvplink_df[!, :P]), :P] .= "1.0"
        pval = parse.(Float64, mvplink_df[!, :P])
    end
    pval[findall(x -> x < 1e-50, pval)] .= 1e-50 # limit smallest pvalues

    # causal SNPs
    causal_snps = get_true_SNPs(sim)
    
    # IHT SNPs
    iht_snps = get_IHT_SNPs(sim)

    # make dataframe to input into MendelPlots
    snpdata = SnpData("/Users/biona001/Benjamin_Folder/UCLA/research/stampeed/imputed_with_0/NFBC_imputed_with_0")
    rename!(snpdata.snp_info, [:chr, :snpid, :genetic_distance, :pos, :allele1, :allele2])
    insertcols!(snpdata.snp_info, size(snpdata.snp_info, 2) + 1, :pval => pval)
    empty_col = ["" for i in 1:size(snpdata.snp_info, 1)]
    insertcols!(snpdata.snp_info, size(snpdata.snp_info, 2) + 1, :empty_col => empty_col)

    # plot
    manhattan(snpdata.snp_info, outfile = "NFBCsim/manhattan_mvPLINK_sim$sim.png",
        annotateinds = causal_snps, annotateinds2 = iht_snps, 
        annotatevar=:empty_col, titles="mvPLINK simulation $sim")
    display("image/png", read("NFBCsim/manhattan_mvPLINK_sim$sim.png"))
end

"""
Imports mvPLINK p-values and plot QQ plot using MendelPlots.jl
"""
function plot_mvPLINK_QQ(sim::Int)
    filename = "NFBCsim/sim$sim/plink.mqfam.total"
    mvplink_df = CSV.read(filename, DataFrame, delim=' ', ignorerepeated=true)
    if eltype(mvplink_df[!, :P]) == Float64
        pval = mvplink_df[!, :P]
    else
        mvplink_df[findall(x -> x == "NA", mvplink_df[!, :P]), :P] .= "1.0"
        pval = parse.(Float64, mvplink_df[!, :P])
    end
    pval[findall(x -> x < 1e-50, pval)] .= 1e-50 # limit smallest pvalues
    qq(pval, outfile = "NFBCsim/QQ_mvPLINK_sim$sim.png",
        ylabel="mvPLINK observed -log10(p)", titles="mvPLINK simulation $sim")
    display("image/png", read("NFBCsim/QQ_mvPLINK_sim$sim.png"))
end

"""
Imports gemma p-values and plot QQ plot using MendelPlots.jl
"""
function plot_gemma_QQ(sim::Int)
    filename = "NFBCsim/sim$sim/gemma.sim$sim.assoc.txt"
    gemma_df = CSV.read(filename, DataFrame)
    pval_wald = gemma_df[!, :p_wald]
    pval_wald[findall(x -> x < 1e-50, pval_wald)] .= 1e-50 # limit smallest pvalues
    qq(pval_wald, outfile = "NFBCsim/QQ_gemma_sim$sim.png",
        ylabel="GEMMA observed -log10(p)", titles="GEMMA simulation $sim")
    display("image/png", read("NFBCsim/QQ_gemma_sim$sim.png"))
end

# """
# Summarize all simulations for IHT, mvPLINK, GEMMA in computation time, true positives,
# false positives, and false positive rates. 
# """
# function summarize_repeats()
#     model = "NFBCsim"
#     n, p = 5340, 24523
#     sims = 1:50 # k = 10, r = 2, βoverlap=2, polygenic model
#     nfbc = SnpData("/Users/biona001/Benjamin_Folder/UCLA/research/stampeed/imputed_with_0/NFBC_imputed_with_0")
#     snp_rsID = nfbc.snp_info.snpid

#     iht_time, iht_power, iht_FP, iht_FPR = Float64[], Float64[], Float64[], Float64[]
#     mvPLINK_time, mvPLINK_power, mvPLINK_FP, mvPLINK_FPR = Float64[], Float64[], Float64[], Float64[]
#     gemma_time, gemma_power, gemma_FP, gemma_FPR = Float64[], Float64[], Float64[], Float64[]

#     for sim in sims
#         dir = "NFBCsim/sim$sim/"
#         try
#             # correct SNPs
#             trueB = readdlm(dir * "trueb.txt")
#             causal_snps = unique([x[1] for x in findall(!iszero, trueB)])
#             causal_snps_rsID = snp_rsID[causal_snps]

#             # IHT
#             iht_β1 = vec(readdlm(dir * "iht_beta1.txt"))
#             iht_β2 = vec(readdlm(dir * "iht_beta2.txt"))
#             detected_snps = findall(!iszero, iht_β1) ∪ findall(!iszero, iht_β2)
#             ihtpower, ihtFP, ihtFPR = power_and_fpr(p, causal_snps, detected_snps)

#             # MVPLINK
#             plinkpower, plinkFP, plinkFPR = process_mvPLINK(dir * "plink.mqfam.total", causal_snps)

#             # GEMMA 
#             gemmapower, gemmaFP, gemmaFPR = process_gemma_result(dir * "gemma.sim$sim.assoc.txt", causal_snps_rsID)
            
#             push!(iht_power, ihtpower); push!(iht_FP, ihtFP); push!(iht_FPR, ihtFPR); 
#             push!(mvPLINK_power, plinkpower); push!(mvPLINK_FP, plinkFP); push!(mvPLINK_FPR, plinkFPR); 
#             push!(gemma_power, gemmapower); push!(gemma_FP, gemmaFP); push!(gemma_FPR, gemmaFPR);
#         catch
#             println("simulation $sim failed!")
#         end
#     end

#     return iht_time, iht_power, iht_FP, iht_FPR,
#         mvPLINK_time, mvPLINK_power, mvPLINK_FP, mvPLINK_FPR,
#         gemma_time, gemma_power, gemma_FP, gemma_FPR
# end

"""
For each simulation set, after performing n simulations using `run_repeats`,
this function reads the summary files for each simulation and summarizes the result. 
"""
function read_summary(; verbose=true)    
    mIHT_time, mIHT_plei_power, mIHT_indp_power, mIHT_FP, mIHT_FDR = 
        Float64[], Float64[], Float64[], Float64[], Float64[]
    koIHT_time, koIHT_plei_power, koIHT_indp_power, koIHT_FP, koIHT_FDR = 
        Float64[], Float64[], Float64[], Float64[], Float64[]

    regex = r"= (\d+\.\d+) seconds, pleiotropic power = (.+), independent power = (\d+\.\d+), FP = (\d+), FDR = (\d\.\d+e?-?\d*), λ = (.+)"

    # polygenic beta = sign * Uniform{0.05, …, 0.5}
    # max condition number = 10
    # q = 5, iterates ≥5 times, init_beta=true, debias=false)
    # sim set 1 are for k = 10, r = 2, βoverlap = 3, path = 5:5:50 (then search around best k)
    # sim set 2 are for k = 20, r = 3, βoverlap = 5, path = 5:5:50 (then search around best k)
    # sim set 3 are for k = 100, r = 3, βoverlap = 7, path = 10:10:200 (then search around best k)

    # compute summary statistics
    open("summary.txt", "w") do summary_io
        for set in 1:2
            successes = 0
            empty!(mIHT_time); empty!(mIHT_plei_power); empty!(mIHT_indp_power); empty!(mIHT_FP); empty!(mIHT_FDR)
            empty!(koIHT_time); empty!(koIHT_plei_power); empty!(koIHT_indp_power); empty!(koIHT_FP); empty!(koIHT_FDR)

            # read each simulation's result
            for sim in 1:100
                if !isfile("set$set/sim$sim/summary.txt")
                    continue
                end
                try
                    open("set$set/sim$sim/summary.txt", "r") do io
                        readline(io); readline(io); readline(io); readline(io); readline(io)

                        # parse mIHT result
                        mIHT = match(regex, readline(io))
                        push!(mIHT_time, parse(Float64, mIHT[1]))
                        push!(mIHT_plei_power, parse(Float64, mIHT[2]))
                        push!(mIHT_indp_power, parse(Float64, mIHT[3]))
                        push!(mIHT_FP, parse(Float64, mIHT[4]))
                        push!(mIHT_FDR, parse(Float64, mIHT[5]))

                        # parse uIHT result
                        koIHT = match(regex, readline(io))
                        push!(koIHT_time, parse(Float64, koIHT[1]))
                        push!(koIHT_plei_power, parse(Float64, koIHT[2]))
                        push!(koIHT_indp_power, parse(Float64, koIHT[3]))
                        push!(koIHT_FP, parse(Float64, koIHT[4]))
                        push!(koIHT_FDR, parse(Float64, koIHT[5]))
                    end
                    successes += 1
                catch
                    continue
                end
            end
            
            # summary statistics
            mIHT_time_mean, mIHT_time_std = round(mean(mIHT_time), digits=1), round(std(mIHT_time), digits=1)
            mIHT_plei_TP_mean, mIHT_plei_TP_std = round(mean(mIHT_plei_power), digits=2), round(std(mIHT_plei_power), digits=2)
            mIHT_indp_TP_mean, mIHT_indp_TP_std = round(mean(mIHT_indp_power), digits=2), round(std(mIHT_indp_power), digits=2)
            mIHT_FDR_mean, mIHT_FDR_std = round(mean(mIHT_FDR), digits=3), round(std(mIHT_FDR), digits=3)
            
            koIHT_time_mean, koIHT_time_std = round(mean(koIHT_time), digits=1), round(std(koIHT_time), digits=1)
            koIHT_plei_TP_mean, koIHT_plei_TP_std = round(mean(koIHT_plei_power), digits=2), round(std(koIHT_plei_power), digits=2)
            koIHT_indp_TP_mean, koIHT_indp_TP_std = round(mean(koIHT_indp_power), digits=2), round(std(koIHT_indp_power), digits=2)
            koIHT_FDR_mean, koIHT_FDR_std = round(mean(koIHT_FDR), digits=3), round(std(koIHT_FDR), digits=3)

            println(summary_io, "set $set summary (successful run = $successes):")
            println(summary_io, "mIHT time = $mIHT_time_mean ± $mIHT_time_std, plei TP = $mIHT_plei_TP_mean ± $mIHT_plei_TP_std, indep TP = $mIHT_indp_TP_mean ± $mIHT_indp_TP_std, FDR = $mIHT_FDR_mean ± $mIHT_FDR_std")
            println(summary_io, "mIHT + knockoff time = $koIHT_time_mean ± $koIHT_time_std, plei TP = $koIHT_plei_TP_mean ± $koIHT_plei_TP_std, indep TP = $koIHT_indp_TP_mean ± $koIHT_indp_TP_std, FDR = $koIHT_FDR_mean ± $koIHT_FDR_std")

            # @show koIHT_FDR

            # if verbose
            #     println("set $set summary (successful run = $successes):")
            #     println("mIHT time = $mIHT_time_mean ± $mIHT_time_std, plei TP = $mIHT_plei_TP_mean ± $mIHT_plei_TP_std, indep TP = $mIHT_indp_TP_mean ± $mIHT_indp_TP_std, FDR = $mIHT_FDR_mean ± $mIHT_FDR_std")
            #     println("mIHT + knockoff time = $koIHT_time_mean ± $koIHT_time_std, plei TP = $koIHT_plei_TP_mean ± $koIHT_plei_TP_std, indep TP = $koIHT_indp_TP_mean ± $koIHT_indp_TP_std, FDR = $koIHT_FDR_mean ± $koIHT_FDR_std\n")
            # end
            
            # for latex table
            if verbose
                println("set $set summary (successful run = $successes):")
                println("\\texttt{mIHT} & \$$mIHT_time_mean \\pm $mIHT_time_std\$ & \$$mIHT_plei_TP_mean \\pm $mIHT_plei_TP_std\$ & \$$mIHT_indp_TP_mean \\pm $mIHT_indp_TP_std\$ & \$$mIHT_FDR_mean \\pm $mIHT_FDR_std\$\\\\")
                println("\\texttt{mIHT-knockoff} & \$$koIHT_time_mean \\pm $koIHT_time_std\$ & \$$koIHT_plei_TP_mean \\pm $koIHT_plei_TP_std\$ & \$$koIHT_indp_TP_mean \\pm $koIHT_indp_TP_std\$ & \$$koIHT_FDR_mean \\pm $koIHT_FDR_std\$\\\\")
            end
        end
    end
    
    return nothing
end
read_summary()
