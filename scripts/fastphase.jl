using SnpArrays

function fastphase(xdata::SnpData;
    n::Int = size(xdata.snparray, 1),
    T::Int = 10, # number of different initial conditions for EM
    K::Int = 10, # number of clusters
    C::Int = 25, # number of EM iterations
    out::AbstractString = "out"
    )
    x = xdata.snparray
    n ≤ size(x, 1) || error("n must be smaller than the number of samples!")
    sampleid = xdata.person_info[!, :iid]
    # create input format for fastPHASE software
    p = size(x, 2)
    open("fastphase.inp", "w") do io
        println(io, n)
        println(io, p)
        for i in 1:n
            println(io, "ID ", sampleid[i])
            # print genotypes for each sample on 2 lines. The "1" for heterozygous
            # genotypes will always go on the 1st line.
            for j in 1:p
                if x[i, j] == 0x00
                    print(io, 0)
                elseif x[i, j] == 0x02 || x[i, j] == 0x03
                    print(io, 1)
                else
                    print(io, '?')
                end
            end
            print(io, "\n")
            for j in 1:p
                if x[i, j] == 0x00 || x[i, j] == 0x02
                    print(io, 0)
                elseif x[i, j] == 0x03
                    print(io, 1)
                else
                    print(io, '?')
                end
            end
            print(io, "\n")
        end
    end
    # T = 10 different initial conditions for EM, K = 10 clusters, C=25 for 25 EM iterations
    run(`./fastPHASE -T$T -K$K -C$C -o$(out) -Pp fastphase.inp`)
    return nothing
end
# plinkname = "/scratch/users/bbchu/ukb_fastPHASE/subset/ukb.10k.chr10"
# xdata = SnpData(plinkname)
# @time fastphase(xdata; n=100, T=10, K=10, C=35) # 4512.960104 seconds (44.70 M allocations: 1.528 GiB, 0.00% gc time, 0.00% compilation time)
# @time fastphase(xdata; n=100, T=2, K=3, C=5) # 51.376597 seconds (11.79 M allocations: 539.831 MiB, 0.07% gc time)
# @time fastphase(xdata; n=100, T=3, K=4, C=5, out="test") # 51.376597 seconds (11.79 M allocations: 539.831 MiB, 0.07% gc time)
# @time fastphase(xdata; n=100, T=3, K=4, C=5, out="test") # 51.376597 seconds (11.79 M allocations: 539.831 MiB, 0.07% gc time)

plinkname = "/scratch/users/bbchu/ukb_SHAPEIT/subset/ukb.10k.chr10"
xdata = SnpData(plinkname)
@time fastphase(xdata; n=1000, T=10, K=10, C=25, out="ukb_chr10_n1000")
