using Revise
using SnpArrays
using DelimitedFiles
using Random
using Knockoffs
using LinearAlgebra
using Distributions
using DataFrames
using CSV
using ProgressMeter
using UnicodePlots

function recombination_segments(breakpoints::Vector{Int}, snps::Int)
    start = 1
    result = UnitRange{Int}[]
    for bkpt in breakpoints
        push!(result, start:bkpt)
        start = bkpt + 1
    end
    push!(result, breakpoints[end]+1:snps)
    return result
end

"""
    simulate_IBD(parent_plinkfile, offsprings)

Simulate genotypes where half the samples are siblings with the other half.
This is done by first randomly sampling a male and female from real genotype data 
to represent parent. Assume they have 2 children. Then generate offspring
individuals by copying segments of the parents genotype directly to the
offspring to represent IBD segments. The number of segments is between 2 and 5
per chromosome and is chosen uniformly across the chromosome. 

# References
https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1003520
"""
function simulate_IBD(parent_plinkfile::AbstractString, offsprings::Int)
    destin = "/scratch/users/bbchu/ukb/siblings/"
    iseven(offsprings) || error("number of offsprings should be even")
    # read original data
    xdata = SnpData(parent_plinkfile)
    x = xdata.snparray
    p = size(x, 2)
    male_idx = findall(x -> x == "1", xdata.person_info[!, :sex])
    female_idx = findall(x -> x == "2", xdata.person_info[!, :sex])
    # simulate new samples
    xsib = SnpArray(destin * "ukb.10k.sibs.chr10.bed", offsprings, p)
    fathers = Int[]
    mothers = Int[]
    pmeter = Progress(offsprings >> 1)
    for i in 1:(offsprings >> 1)
        # assign parents
        dad = rand(male_idx)
        mom = rand(female_idx)
        push!(fathers, dad)
        push!(mothers, mom)
        # child 1
        recombinations = rand(1:4)
        breakpoints = sort!(rand(1:p, recombinations))
        parent1, parent2 = rand() < 0.5 ? (dad, mom) : (mom, dad)
        segments = recombination_segments(breakpoints, p)
        for j in 1:length(segments)
            parent = isodd(j) ? parent1 : parent2
            segment = segments[j]
            copyto!(@view(xsib[2i - 1, segment]), @view(x[parent, segment]))
        end
        # child 2
        recombinations = rand(1:4)
        breakpoints = sort!(rand(1:p, recombinations))
        parent1, parent2 = rand() < 0.5 ? (dad, mom) : (mom, dad)
        segments = recombination_segments(breakpoints, p)
        for j in 1:length(segments)
            parent = isodd(j) ? parent1 : parent2
            segment = segments[j]
            copyto!(@view(xsib[2i, segment]), @view(x[parent, segment]))
        end
        # update progress
        next!(pmeter)
    end
    # also generate bim and fam file
    cp(parent_plinkfile * ".bim", destin * "ukb.10k.sibs.chr10.bim")
    open(destin * "ukb.10k.sibs.chr10.fam", "w") do io
        for i in 1:(offsprings >> 1)
            println(io, 2i - 1, ' ', 2i - 1, ' ', fathers[i], ' ', mothers[i], " -9")
            println(io, 2i,     ' ', 2i,     ' ', fathers[i], ' ', mothers[i], " -9")
        end
    end
    return xsib
end

Random.seed!(2021)
parent_plinkfile = "/scratch/users/bbchu/ukb/subset/ukb.10k.chr10"
offsprings = 10000
xsib = simulate_IBD(parent_plinkfile, offsprings)

# check GRM
Φ = grm(xsib, method=:Robust)
histogram(vec(Φ))
