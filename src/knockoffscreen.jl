# this file contains routines to generate KnockoffScreen knockoffs
# as described in the paper https://www.nature.com/articles/s41467-021-22889-4

struct KnockoffScreen
    x::SnpArray
    windowsize::Int
    β̂::Vector{Float64}         # Intercept is always last column
    X::ElasticArray{Float64}   # Original genotypes come first, then knockoff genotypes, then column of 1s (intercept)
    y::Vector{Float64}         # The `j`th SNP, treated as the response
end
function KnockoffScreen(x::SnpArray, windowsize::Int)
    n = size(x, 1)
    β̂ = zeros(windowsize)
    X = ElasticArray{Float64}(undef, n, 0)
    y = zeros(n)
    return KnockoffScreen(x, windowsize, β̂, X, y)
end

# todo: use elastic array to avoid reallocating design matrix in each loop
function full_knockoffscreen(x::SnpArray, windowsize::Int=100)
    n, p = size(x)
    windows = ceil(Int, p / windowsize)
    # preallocated vectors
    X̃ = zeros(n, p)
    X̂j = zeros(n)
    β̂ = zeros(windowsize)
    y = zeros(n)
    residual = zeros(n)
    pmeter = Progress(windows, 1, "Generating knockoffs", 1)
    # loop over each window independently
    for window in 1:windows-1
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
    remainder = p - floor(Int, p / windowsize) * windowsize
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
