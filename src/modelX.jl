"""
    modelX_gaussian_knockoffs(X::Matrix{T})

Creates model-free multivariate normal knockoffs by sequentially sampling from 
conditional multivariate normal distributions.

# Inputs
+ `X`: A `n × p` numeric matrix. Each row is a sample, and each column is standardized
to mean 0 variance 1. 

# Reference: 
"Panning for Gold: Model-X Knockoffs for High-dimensional Controlled
Variable Selection" by Candes, Fan, Janson, and Lv (2018)
"""
function modelX_gaussian_knockoffs(X::Matrix{T}, method::Symbol, μ::Vector{T}) where T <: AbstractFloat
    n, p = size(X)
    full_svd = n > p ? true : false
    # compute gram matrix using full svd
    U, σ, V = svd(X, full=full_svd)
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
        problem = maximize(sum(svar), svar ≥ 0, 1 ≥ svar, 2Σ - Diagonal(svar) in :SDP)
        solve!(problem, () -> SCS.Optimizer(verbose=false))
        s = clamp.(evaluate(svar), 0, 1) # for numeric stability
    elseif method==:asdp
        # todo
        error("ASDP not supported yet! sorry!")
    else
        error("modelX_gaussian: method can only be :equi, or :sdp, or :asdp")
    end
    X̃ = condition(X, μ, Σinv, Diagonal(s))
    return Knockoff(X, X̃, s, Σ, Σinv)
end

"""
    condition(x::AbstractVector, μ::AbstractVector, Σinv::AbstractMatrix, D::AbstractMatrix)

Samples a knockoff x̃ from Gaussian x using conditional distribution formulas:

If (x, x̃) ~ N((μ, μ), G) where G = [Σ  Σ - D; Σ - D  Σ], then we sample x̃ from 
x̃|x = N(x - D*inv(Σ)(x - μ), 2D - D*inv(Σ)*D)

# todo: efficiency
"""
function condition(X::AbstractMatrix, μ::AbstractVector, Σinv::AbstractMatrix, D::AbstractMatrix)
    n, p = size(X)
    ΣinvD = Σinv * D
    new_V = Symmetric(2D - D * ΣinvD + 0.00001I) # small perturbation ensures positive eigvals
    L = cholesky(new_V).L
    return X - (X .- μ') * ΣinvD + randn(n, p) * L
end
