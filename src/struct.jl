"""
A `Knockoff` is an `AbstractMatrix`, essentially the matrix [X X̃] of
concatenating X̃ to X. It behaves like a regular matrix and can be inputted
into any function that supports inputs of type `AbstractMatrix`. Basic
operations like @view(A[:, 1]) are supported. 
"""
struct Knockoff{T} <: AbstractMatrix{T}
    X::Matrix{T} # original design matrix
    X̃::Matrix{T} # knockoff of X
    s::Vector{T} # diagonal(s) and 2Σ - diagonal(s) are both psd
    Σ::Matrix{T} # X'X
    Σinv::Matrix{T} # inv(X'X)
end

Base.size(A::Knockoff) = size(A.X, 1), 2size(A.X, 2)
Base.eltype(A::Knockoff) = eltype(A.X)
function Base.getindex(A::Knockoff, i::Int)
    n, p = size(A.X)
    i ≤ n * p ? getindex(A.X, i) : getindex(A.X̃, i - n * p)
end
function Base.getindex(A::Knockoff, i::Int, j::Int)
    n, p = size(A.X)
    j ≤ p ? getindex(A.X, i, j) : getindex(A.X̃, i, j - p)
end
function LinearAlgebra.mul!(C::AbstractMatrix, A::Knockoff, B::AbstractMatrix)
    p = size(A.X, 2)
    mul!(C, A.X, @view(B[1:p, :]))
    mul!(C, A.X̃, @view(B[p+1:end, :]), 1.0, 1.0)
end
function LinearAlgebra.mul!(c::AbstractVector, A::Knockoff, b::AbstractVector)
    p = size(A.X, 2)
    mul!(c, A.X, @view(b[1:p]))
    mul!(c, A.X̃, @view(b[p+1:end]), 1.0, 1.0)
end

"""
    modelX_gaussian(X::Matrix{T})

Creates model-free multivariate normal knockoffs by sequentially sampling from 
conditional multivariate normal distributions.

# Inputs
+ `X`: A `n × p` numeric matrix. Each row is a sample, and each column is standardized
    to mean 0 variance 1. 

# Reference: 
"Panning for Gold: Model-X Knockoffs for High-dimensional Controlled
Variable Selection" by Candes, Fan, Janson, and Lv (2018)
"""
function modelX_gaussian(X::Matrix{T}, method::Symbol=:sdp;
    μ::Vector{T}=mean(X, dims=2)) where T <: AbstractFloat
    n, p = size(X)
    # compute gram matrix using full svd
    U, σ, V = svd(X, full=true)
    Σ = V * Diagonal(σ)^2 * V'
    Σinv = V * inv(Diagonal(σ)^2) * V'
    # compute s vector using the specified method
    if method == :equi
        λmin = typemax(T)
        for σi in σ
            σi^2 < λmin && (λmin = σi^2)
        end
        s = min(1, 2λmin) .* ones(size(Σ, 1))
    elseif method == :sdp
        svar = Variable(p)
        problem = maximize(sum(svar), svar ≥ 0, 1 ≥ svar, 2Σ - Diagonal(svar) == Semidefinite(p))
        solve!(problem, () -> SCS.Optimizer(verbose=false))
        s = clamp.(vec(svar.value), 0, 1) # for numeric stability
    elseif method==:asdp
        # todo
        error("ASDP not supported yet! sorry!")
    else
        error("modelX_gaussian: method can only be :equi, or :sdp, or :asdp")
    end
    X̃ = Matrix{T}(undef, n, p)
    for i in 1:p
        x̃ = condition(@view(X[i, :]), μ, Σinv, Diagonal(s)) #todo efficiency
        copyto!(@view(X̃[i, :]), x̃)
    end
    return Knockoff(X, X̃, s, Σ, Σinv)
end

"""
    condition(x::AbstractVector, μ::AbstractVector, Σinv::AbstractMatrix, D::AbstractMatrix)

Samples a knockoff x̃ from Gaussian x using conditional distribution formulas:

If (x, x̃) ~ N((μ, μ), G) where G = [Σ  Σ - D; Σ - D  Σ], then we sample x̃ from 
x̃|x = N(x - D*inv(Σ)(x - μ), 2D - D*inv(Σ)*D)
"""
function condition(x::AbstractVector, μ::AbstractVector, Σinv::AbstractMatrix, D::AbstractMatrix)
    DinvΣ = D * Σinv
    new_μ = x - DinvΣ * (x - μ)
    new_V = 2D - DinvΣ * D
    return rand(MvNormal(new_μ, new_V))
end

"""
    fixed_knockoffs(X::Matrix{T}; method=:sdp)

Creates fixed knockoffs based on equation (2.2)-(2.4) of 
"Controlling the false discovery rate via Knockoffs" by Barber and Candes (2015)

# Inputs
+ `X`: A `n × p` numeric matrix, each row is a sample, and each column is normalized
    to mean 0 variance 1 with unit norm. 

# Optional inputs
+ `method`: can be :equi for equi-distant knockoffs (eq 2.3) or :sdp for SDP
    knockoffs (eq 2.4)

# Output
+ `Knockoff`: A struct containing the original `X` and its knockoff `X̃`, 
    in addition to other variables (e.g. `s`)
"""
function fixed_knockoffs(X::Matrix{T}; method::Symbol=:sdp) where T <: AbstractFloat
    n, p = size(X)
    n ≥ 2p || error("fixed_knockoffs: only works for n ≥ 2p case! sorry!")
    # compute gram matrix using full svd
    U, σ, V = svd(X, full=true)
    Σ = V * Diagonal(σ)^2 * V'
    Σinv = V * inv(Diagonal(σ)^2) * V'
    # compute s vector using the specified method
    if method == :equi
        λmin = typemax(T)
        for σi in σ
            σi^2 < λmin && (λmin = σi^2)
        end
        s = min(1, 2λmin) .* ones(size(Σ, 1))
    elseif method == :sdp
        svar = Variable(p)
        problem = maximize(sum(svar), svar ≥ 0, 1 ≥ svar, 2Σ - Diagonal(svar) == Semidefinite(p))
        solve!(problem, () -> SCS.Optimizer(verbose=false))
        s = clamp.(vec(svar.value), 0, 1) # for numeric stability
    else
        error("fixed_knockoffs: method can only be :equi or :sdp")
    end
    # compute Ũ such that Ũ'X = 0
    Ũ = U[:, p+1:2p]
    # compute C such that C'C = 2D - D*inv(Σ)*D via eigendecomposition (cholesky not stable)
    D = Diagonal(s)
    γ, P = eigen(2D - D*Σinv*D)
    clamp!(γ, 0, typemax(T)) # numerical stability
    C = Diagonal(sqrt.(γ)) * P
    # compute knockoffs
    X̃ = X * (I - Σinv*D) + Ũ * C
    return Knockoff(X, X̃, s, Σ, Σinv)
end

function normalize_col!(X::AbstractMatrix)
    n, p = size(X)
    @inbounds for x in eachcol(X)
        μi = mean(x)
        xnorm = norm(x)
        @simd for i in eachindex(x)
            x[i] = (x[i] - μi) / xnorm
        end
    end
    return X
end
