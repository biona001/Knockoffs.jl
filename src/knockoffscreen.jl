# this file contains routines to generate KnockoffScreen knockoffs
# as described in the paper https://www.nature.com/articles/s41467-021-22889-4

struct KnockoffScreen
    x::SnpArray
    windowsize::Int
    β̂::Vector{Float64}         # Intercept is always last column
    X::ElasticArray{Float64}   # Original genotypes come first, then knockoff genotypes, then column of 1s (intercept)
    Xj::Vector{Float64}        # The `j`th SNP, treated as the response
    X̂j::Vector{Float64}        # Intermediate variable needed for computing X̃j
    ϵ ::Vector{Float64}        # Intermediate variable needed for computing X̃j
    X̃j::Vector{Float64}        # Knockoff of the `j`th SNP
end
function KnockoffScreen(x::SnpArray, windowsize::Int)
    n = size(x, 1)
    β̂ = zeros(windowsize)
    X = ElasticArray{Float64}(undef, n, 0)
    X̂j, ϵ, X̃j = zeros(n), zeros(n), zeros(n)
    return KnockoffScreen(x, windowsize, β̂, X, Xj, X̂j, ϵ, X̃j)
end

function Base.iterate(ks::KnockoffScreen, state=1)
    if state > size(ks.x, 2)
        return nothing
    else
        # todo
        # return result, state + 1
    end
end

"""
    full_knockoffscreen(x::SnpArray; windowsize::Int=100)

Generates knockoffs `X̃ⱼ` by on regressing `Xⱼ` on SNPs knockoffs within a sliding window of width `windowsize`. 

# Inputs
+ `x`: A `SnpArray` or `String` for the path of the PLINK `.bed` file
+ `windowsize`: `Int` specifying window width. Defaults to 100

# Outputs
+ `X̃`: A `n × p` dense matrix of `Float64`, each row is a sample.

# References
+ He, Zihuai, Linxi Liu, Chen Wang, Yann Le Guen, Justin Lee, Stephanie Gogarten, Fred Lu et al. "Identification of putative causal loci in whole-genome sequencing data via knockoff statistics." Nature communications 12, no. 1 (2021): 1-18.
+ He, Zihuai, Yann Le Guen, Linxi Liu, Justin Lee, Shiyang Ma, Andrew C. Yang, Xiaoxia Liu et al. "Genome-wide analysis of common and rare variants via multiple knockoffs at biobank scale, with an application to Alzheimer disease genetics." The American Journal of Human Genetics 108, no. 12 (2021): 2336-2353.

# TODO
+ Use `ElasticArrays.jl` to avoid reallocating design matrix in each loop
+ Write iterator interface to avoid allocating and storing all knockoffs at once
"""
function full_knockoffscreen(x::SnpArray; windowsize::Int=100)
    n, p = size(x)
    windows = floor(Int, p / windowsize)
    remainder = p - floor(Int, p / windowsize) * windowsize
    # preallocated vectors
    X̃ = zeros(n, p)
    X̂j = zeros(n)
    β̂ = zeros(windowsize)
    y = zeros(n)
    residual = zeros(n)
    pmeter = Progress(windows + (remainder > 0), 1, "Generating knockoffs")
    # loop over each window independently
    for window in 1:windows
        snps = (window - 1)*windowsize + 1:window * windowsize
        for j in 1:windowsize
            snp = j + (window - 1)*windowsize
            # form design matrix
            G = SnpArrays.convert(Matrix{Float64}, @view(x[:, setdiff(snps, snp)]),
                center=true, scale=true, impute=true)
            G̃ = @view(X̃[:, (window - 1)*windowsize + 1:snp - 1])
            SnpArrays.copyto!(y, @view(x[:, snp]), center=true, scale=true, impute=true)
            Gfull = hcat(G, G̃, ones(n))
            # linear regression
            β̂ = Gfull \ y
            mul!(X̂j, Gfull, β̂)
            # compute residual, then permute it
            residual .= y .- X̂j
            shuffle!(residual)
            # save knockoff
            X̃[:, snp] .= y .+ residual
        end
        next!(pmeter) # update progress
    end
    # last window
    snps = (windows - 1)*windowsize + 1:p
    for snp in p-remainder+1:p
        # form design matrix
        G = SnpArrays.convert(Matrix{Float64}, @view(x[:, setdiff(snps, snp)]),
            center=true, scale=true, impute=true)
        G̃ = @view(X̃[:, (windows - 1)*windowsize + 1:snp - 1])
        SnpArrays.copyto!(y, @view(x[:, snp]), center=true, scale=true, impute=true)
        Gfull = hcat(G, G̃, ones(n))
        # linear regression
        β̂ = Gfull \ y
        mul!(X̂j, Gfull, β̂)
        # compute residual, then permute it
        residual .= y .- X̂j
        shuffle!(residual)
        # save knockoff
        X̃[:, snp] .= y .+ residual
    end
    next!(pmeter) # update progress
    return X̃
end

full_knockoffscreen(plinkfile::AbstractString; windowsize::Int=100) = 
    full_knockoffscreen(SnpData(plinkfile).snparray, windowsize=windowsize)

# function full_knockoffscreen(x::SnpArray, windowsize::Int=100)
#     n, p = size(x)
#     ks = KnockoffScreen(x, windowsize)
#     X̃ = zeros(n, p)
#     # first `windowsize` knockoffs
#     ElasticArrays.resize!(ks.X, n, windowsize)
#     SnpArrays.copyto!(ks.X, @view(x[:, 2:windowsize]), center=true, scale=true, impute=true)
#     fill!(@view(ks.X[:, end]), 1)
#     SnpArrays.copyto!(ks.Xj, @view(x[:, 1]), center=true, scale=true, impute=true)
#     for j in 2:windowsize
#         ElasticArrays.resize!(ks.X, n, windowsize + j - 1)
#         SnpArrays.copyto!(ks.X, @view(x[:, 1:j-1]), center=true, scale=true, impute=true)
#         SnpArrays.copyto!(ks.X, @view(x[:, j:windowsize+j-1]), center=true, scale=true, impute=true)
#     end
# end
