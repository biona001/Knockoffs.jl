"""
Computes A^{-1/2} via eigen-decomposition
"""
function inverse_mat_sqrt(A::Symmetric; tol=1e-4)
    λ, ϕ = eigen(A)
    for i in eachindex(λ)
        λ[i] < tol && (λ[i] = tol)
    end
    return ϕ * Diagonal(1 ./ sqrt.(λ)) * ϕ'
end

"""
Solves the equi-correlated group knockoff problem. Here
`Σ` is the true covariance matrix (scaled so that it has 1 on its diagonal)
and `Σblocks` is the block-diagonal covariance matrix where each 
block corresponds to groups.

Details can be found in
Dai & Barber 2016, The knockoff filter for FDR control in group-sparse and multitask regression
"""
function solve_group_equi(Σ::AbstractMatrix, Σblocks::BlockDiagonal)
    Db = Matrix{eltype(Σ)}[]
    for Σbi in Σblocks.blocks
        push!(Db, inverse_mat_sqrt(Symmetric(Σbi)))
    end
    Db = BlockDiagonal(Db)
    λmin = Symmetric(Db * Σ * Db) |> eigmin
    γ = min(1, 2λmin)
    S = BlockDiagonal(γ .* Σblocks.blocks)
    return S, [γ]
end

"""
Solves the SDP group knockoff problem using analogy to the equi-correlated
group knockoffs. Basically, the idea is to optimize a vector `γ` where `γ[j]` 
multiplies Σ_jj. In the equi-correlated setting, all `γ[j]` is forced to be equal.

Details can be found in
Dai & Barber 2016, The knockoff filter for FDR control in group-sparse and multitask regression
"""
function solve_group_SDP(Σ::AbstractMatrix, Σblocks::BlockDiagonal)
    model = Model(() -> Hypatia.Optimizer(verbose=false))
    n = nblocks(Σblocks)
    block_sizes = size.(Σblocks.blocks, 1)
    @variable(model, 0 <= γ[1:n] <= 1)
    blocks = BlockDiagonal([γ[i] * Σblocks.blocks[i] for i in 1:n]) |> Matrix
    @objective(model, Max, block_sizes' * γ)
    @constraint(model, Symmetric(2Σ - blocks) in PSDCone())
    JuMP.optimize!(model)
    γs = clamp!(JuMP.value.(γ), 0, 1)
    S = BlockDiagonal(γs .* Σblocks.blocks)
    return S, γs
end

"""
    solve_s_group(Σ, Sblocks, groups, [method=:equi]; kwargs...)

Solves the group knockoff problem, returns block diagonal matrix S
satisfying `2Σ - S ⪰ 0` and the constant(s) γ.

# Inputs 
+ `Σ`: A covariance matrix that has been scaled to a correlation matrix.
+ `Sblocks`: A `BlockDiagonal` matrix that approximates `Σ` using group
    structure. Each block should be `pi × pi` where `pi` is number of variables
    in group `i`
+ `groups`: Vector of group membership
+ `method`: Method for constructing knockoffs. Options are `:equi` or `:sdp`
"""
function solve_s_group(
    Σ::AbstractMatrix, 
    Sblocks::BlockDiagonal, 
    groups::Vector{Int},
    method::Symbol=:equi;
    kwargs...)
    # check for error first
    all(x -> x ≈ 1, diag(Σ)) || error("Σ must be scaled to a correlation matrix first.")
    for block in Sblocks.blocks
        all(x -> x ≈ 1, diag(block)) || 
            error("Sblocks must be scaled to a correlation matrix first.")
    end
    # solve optimization problem
    if method == :equi
        S, γs = solve_group_equi(Σ, Sblocks)
    elseif method == :sdp
        S, γs = solve_group_SDP(Σ, Sblocks)
    else
        error("Method can only be :equi or :sdp, but was $method")
    end
    return S, γs
end

"""
    modelX_gaussian_group_knockoffs(X, groups, method, Σ, μ)
    modelX_gaussian_group_knockoffs(X, groups, method; [covariance_approximator])

Constructs Gaussian model-X group knockoffs. If the covariance `Σ` and mean `μ` 
are not specified, they will be estimated from data, i.e. we will make second-order
group knockoffs. To incorporate group structure, the (true or estimated) covariance 
matrix is block-diagonalized according to `groups` membership to solve a relaxed 
optimization problem. See reference paper and Knockoffs.jl docs for more details. 

# Inputs
+ `X`: A `n × p` design matrix. Each row is a sample, each column is a feature.
+ `groups`: Vector of group membership
+ `method`: Method for constructing knockoffs. Options are `:equi` or `:sdp`
+ `Σ`: A `p × p` covariance matrix for columns of `X`
+ `μ`: A length `p` vector storing the true column means of `X`
+ `covariance_approximator`: A covariance estimator, defaults to 
    `LinearShrinkage(DiagonalUnequalVariance(), :lw)`. See CovarianceEstimation.jl 
    for more options.

# Reference
Dai & Barber 2016, The knockoff filter for FDR control in group-sparse and multitask regression
"""
function modelX_gaussian_group_knockoffs(
    X::Matrix, 
    groups::AbstractVector{Int},
    method::Symbol;
    covariance_approximator=LinearShrinkage(DiagonalUnequalVariance(), :lw),
    kwargs...
    )
    # first check errors
    length(groups) == size(X, 2) || 
        error("Expected length(groups) == size(X, 2). Each variable in X needs a group membership.")
    issorted(groups) || 
        error("groups not sorted. Currently group memberships must be non-overlapping and contiguous")
    # approximate covariance matrix
    Σapprox = cov(covariance_approximator, X)
    # mean component is just column means
    μ = vec(mean(X, dims=1))
    return modelX_gaussian_group_knockoffs(X, groups, method, Σapprox, μ)
end

function modelX_gaussian_group_knockoffs(
    X::Matrix, 
    groups::AbstractVector{Int},
    method::Symbol,
    Σ::AbstractMatrix,
    μ::AbstractVector;
    kwargs...
    )
    # first check errors
    length(groups) == size(X, 2) || 
        error("Expected length(groups) == size(X, 2). Each variable in X needs a group membership.")
    issorted(groups) || 
        error("groups not sorted. Currently group memberships must be non-overlapping and contiguous")
    # Scale covariance to correlation matrix
    σs = sqrt.(diag(Σ))
    iscor = all(x -> x ≈ 1, σs)
    Σcor = iscor ? Σ : StatsBase.cov2cor!(Matrix(Σ), σs)
    # define group-blocks
    Sblocks = Matrix{eltype(X)}[]
    for g in unique(groups)
        idx = findall(x -> x == g, groups)
        push!(Sblocks, Σcor[idx, idx])
    end
    Sblocks = BlockDiagonal(Sblocks)
    # compute block diagonal S matrix using the specified knockoff method
    S, γs = solve_s_group(Σcor, Sblocks, groups, method; kwargs...)
    # rescale S back to the result for a covariance matrix   
    iscor || StatsBase.cor2cov!(S, σs)
    # generate knockoffs
    X̃ = condition(X, μ, inv(Σ), S)
    return GaussianGroupKnockoff(X, X̃, S, γs, Symmetric(Σ), method)
end

# every `windowsize` SNPs form a group
function partition_group(snp_idx; windowsize=10)
    p = length(snp_idx)
    windows = floor(Int, p / windowsize)
    remainder = p - windows * windowsize
    groups = zeros(Int, p)
    for window in 1:windows
        groups[(window - 1)*windowsize + 1:window * windowsize] .= window
    end
    groups[p-remainder+1:p] .= windows + 1
    return groups
end

function modelX_gaussian_group_knockoffs(
    xdata::SnpData, 
    method::Symbol;
    T::Type{T} = Float32,
    covariance_approximator=LinearShrinkage(DiagonalUnequalVariance(), :lw),
    kwargs...
    )
    # first check errors
    n, p = size(xdata)
    any(x -> iszero(x), maf(xdata.snparray)) || 
        error("Detected monomorphic SNPs. Please make sure QC is done properly.")
    # estimate rough memory requirement
    chromosomes = xdata.snp_info[!, :chromosome]
    unique_chr = unique(chromosomes)
    chromosome_length = [count(x -> x == chr, chromosomes) for chr in unique_chr]
    max_chr_len = maximum(chromosome_length)
    @info "This routine requires roughly $((4max_chr_len^2 + 4n*max_chr_len) / 10^9) GB of RAM"
    # preallocated arrays
    Sblocks = Matrix{T}[]
    Σapprox = Matrix{T}[]
    genotype_storage = Matrix{T}(undef, n, max_chr_len)
    # construct group knockoffs chromosome by chromosome
    for chr in unique_chr
        snp_idx = findall(x -> x == chr, chromosomes)
        groups = get_group(snp_idx)
        # copy genotypes to numeric matrix
        copyto!(genotype_storage, @view(xdata.snparray[:, snp_idx]), impute=true)
        X = @view(genotype_storage[:, 1:length(snp_idx)])
        # approximate covariance matrix and scale it to correlation matrix
        @time Σapprox = cov(covariance_approximator, X)
        σs = sqrt.(diag(Σapprox))
        Σcor = StatsBase.cov2cor!(Matrix(Σapprox), σs)
        # define group-blocks
        for g in unique(groups)
            idx = findall(x -> x == g, groups)
            push!(Sblocks, Symmetric(Σcor[idx, idx]))
        end
    end
end

"""
    modelX_gaussian_group_knockoffs(xdata::SnpData, method)

Generates (model-X Gaussian second-order) group knockoffs for
a single chromosome stored in PLINK formatted data. 

# todo 
1. Use single precision for Σ and X, once this issue is resolved
    https://github.com/mateuszbaran/CovarianceEstimation.jl/issues/82
2. Handle PLINK files with multiple chromosomes and multiple plink
    files each storing a chromosome
"""
function modelX_gaussian_group_knockoffs(
    x::SnpArray, # assumes only have 1 chromosome, allows missing data
    method::Symbol;
    T::Type{T} = Float64,
    covariance_approximator=LinearShrinkage(DiagonalUnequalVariance(), :lw),
    outfile = Union{String, UndefInitializer} = UndefInitializer
    )
    # estimate rough memory requirement (need Σ which is p*p and X which is n*p)
    n, p = size(x)
    @info "This routine requires at least $((T.size * p^2 + T.size * n*p) / 10^9) GB of RAM"
    # import genotypes into numeric array
    @time X = convert(Matrix{T}, x, impute=true)
    any(x -> iszero(x), std(X, dims=1)) || 
        error("Detected monomorphic SNPs. Please make sure QC is done properly.")
    #
    #### todo: need to split large chromosomes into blocks, and redefine groups accordingly
    #
    # approximate covariance matrix and scale it to correlation matrix
    @time Σapprox = cov(covariance_approximator, X) # ~2 min
    σs = sqrt.(diag(Σapprox))
    Σcor = StatsBase.cov2cor!(Matrix(Σapprox), σs)
    # define group-blocks
    groups = partition_group(1:p; windowsize=10) # todo: redefine groups if chr is split into blocks
    group_ranges = Vector{Vector{Int}}[]
    Sblocks = Matrix{T}[]
    for g in unique(groups)
        idx = findall(x -> x == g, groups)
        push!(Sblocks, @view(Σcor[idx, idx]))
        push!(group_ranges, idx)
    end
    Sblocks = BlockDiagonal(Sblocks)
    # compute block diagonal S matrix using the specified knockoff method
    @time S, γs = solve_s_group(Σcor, Sblocks, groups, method) # 1061.229520 seconds (this step requires more memory allocation, need to analyze)
    # rescale S back to the result for a covariance matrix   
    for (i, idx) in enumerate(group_ranges)
        StatsBase.cor2cov!(S.blocks[i], @view(σs[idx]))
    end
    # generate knockoffs
    μ = vec(mean(X, dims=1))
    @time invΣ = inv(Σapprox) # 
    @time X̃ = condition(X, μ, invΣ, S) # 
    # finally, round values of X̃ to integers
    round.(X̃)
    clamp!.(X̃, 0, 2)
    # copy result into SnpArray before returning
    X̃snparray = SnpArray(outfile, n, p)
    for j in 1:p, i in 1:n
        X̃snparray[i, j] = iszero(X̃[i, j]) ? 0x00 : 
            isone(iszero(X̃[i, j])) ? 0x02 : 0x03
    end
    return X̃snparray
end
