"""
    ipad(X::Matrix; [r_method], [m])

Generates knockoffs based on intertwined probabilitistic factors decoupling (IPAD).
This assumes that `X` can be factored as `X = FΛ' + E` where `F` is a `n × r`
random matrix of latent factors, `Λ` are factor loadings, and `E` are residual
errors. When this assumption is met, FDR can be controlled with no power loss
when applying the knockoff procedure. Internally, we need to compute an
eigenfactorization for a `n × n` matrix. This is often faster than standard
model-X knockoffs which requires solving `p`-dimensional convex optimization
problem.

# Inputs
+ `X`: A `n × p` numeric matrix, each row is a sample, and each column is covariate.
+ `r_method`: Method used for estimating `r`, the number of latent factors. 
    Choices include `:er` (default), `:gr`, or `:ve`
+ `m`: Number of (simultaneous) knockoffs per variable to generate, default `m=1`

# References
+ Fan, Y., Lv, J., Sharifvaghefi, M. and Uematsu, Y., 2020. IPAD: stable interpretable forecasting with knockoffs inference. Journal of the American Statistical Association, 115(532), pp.1822-1834.
+ Bai, J., 2003. Inferential theory for factor models of large dimensions. Econometrica, 71(1), pp.135-171.
+ Ahn, S.C. and Horenstein, A.R., 2013. Eigenvalue ratio test for the number of factors. Econometrica, 81(3), pp.1203-1227.
"""
function ipad(X::AbstractMatrix{T}; r_method = :er, m::Number = 1) where T
    n, p = size(X)
    # estimate r
    XXt = X * X'
    evals, evecs = eigen(XXt)
    evecs = evecs[:, sortperm(evals)]
    reverse!(evals)
    r = 0
    # kmax = min(n, p)
    kmax = count(x -> x > 0, evals) - 2
    if r_method == :er
        r = [evals[i] / evals[i + 1] for i in 1:kmax-1] |> argmax
    elseif r_method == :gr
        r = [gr(evals, k) for k in 1:kmax] |> argmax
    elseif r_method == :ve
        r = findfirst(x -> x > 0.9, cumsum(evals) ./ sum(evals))
    else
        error("r_method can only be :er, :gr but was $r_method")
    end
    # compute Ĉ and E
    F̂ = sqrt(p) * evecs[:, 1:r]
    Ĉ = F̂ * (F̂' * X) / p
    E = X - Ĉ
    σ2 = var(E, dims=1) |> vec |> Diagonal
    # sample knockoffs
    Xk = Matrix{T}[]
    for i in 1:m
        Ek = rand(MvNormal(σ2), n)' |> Matrix
        push!(Xk, Ĉ + Ek)
    end
    X̃ = hcat(Xk...)
    return IPADKnockoff(X, X̃, Int(m), r, r_method)
end

v(evals, k::Int) = sum(@views(evals[k+1:end]))
gr(evals, k::Int) = log(1 + evals[k] / v(evals, k)) / log(1 + evals[k+1] / v(evals, k+1))
