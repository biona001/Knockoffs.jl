"""
    fit_lasso(y, X, method=:mvr, ...)

Generates model-X knockoffs with `method`, runs Lasso, 
then applies the knockoff-filter.

# Inputs
+ `y`: Response vector
+ `X`: Design matrix
+ `method`: Method for knockoff generation (defaults to `:mvr`)
+ `d`: Distribution of response. Defaults `Normal()`, for binary response
    (logistic regression) use `Binomial()`.
+ `fdrs`: Target FDRs, defaults to `[0.01, 0.05, 0.1, 0.25, 0.5]`
+ `filter_method`: Choices are `:knockoff` (default) or `:knockoff_plus`
+ `debias`: Defines how the selected coefficients are debiased. Specify `:ls` 
    for least squares (default) or `:lasso` for Lasso (only running on the 
    support). To not debias, specify `debias=nothing`
+ `kwargs`: Additional arguments to input into `glmnetcv` and `glmnet`
"""
function fit_lasso(
    y::AbstractVector{T},
    X::AbstractMatrix{T};
    method::Symbol = :mvr,
    d::Distribution=Normal(),
    fdrs::Vector{Float64}=[0.01, 0.05, 0.1, 0.25, 0.5],
    groups::Union{Nothing, AbstractVector{Int}} = nothing,
    filter_method::Symbol = :knockoff,
    debias::Union{Nothing, Symbol} = :ls,
    kwargs..., # arguments for glmnetcv
    ) where T
    ko = modelX_gaussian_knockoffs(X, method)
    return fit_lasso(y, X, ko.X̃, d=d, fdrs=fdrs, groups=groups, 
        filter_method=filter_method, debias=debias; kwargs...)
end

function fit_lasso(
    y::AbstractVector{T},
    X::AbstractMatrix{T}, 
    X̃::AbstractMatrix{T};
    d::Distribution=Normal(),
    fdrs::Vector{Float64}=[0.01, 0.05, 0.1, 0.25, 0.5],
    groups::Union{Nothing, AbstractVector{Int}} = nothing,
    filter_method::Symbol = :knockoff, # `:knockoff` or `:knockoff_plus`
    debias::Union{Nothing, Symbol} = :ls,
    kwargs..., # arguments for glmnetcv
    ) where T <: AbstractFloat
    isnothing(groups) || error("groups keyword not supported yet! Sorry!")
    ytmp = d == Binomial() ? form_glmnet_logistic_y(y) : y
    # fit lasso (note: need to interleaves X with X̃)
    XX̃, original, knockoff = merge_knockoffs_with_original(X, X̃)
    knockoff_cv = glmnetcv(XX̃, ytmp, d; kwargs...)
    βestim = GLMNet.coef(knockoff_cv)
    a0 = knockoff_cv.path.a0[argmin(knockoff_cv.meanloss)]
    # compute feature importance statistics and allocate necessary knockoff-filter variables
    W = isnothing(groups) ? coefficient_diff(βestim, original, knockoff) : 
        coefficient_diff(βestim, groups, original, knockoff)
    βs, a0s, τs = Vector{T}[], T[], T[]
    for fdr in fdrs
        # apply knockoff-filter based on target fdr
        β_filtered, _, τ = isnothing(groups) ? 
            extract_beta(βestim, fdr, original, knockoff, filter_method, W) : 
            extract_beta(βestim, fdr, groups, original, knockoff, filter_method, W)
        # debias the estimates if requested
        isnothing(debias) || (a0 = debias!(β_filtered, X, y; method=debias, d=d, kwargs...))
        # save knockoff statistics
        push!(βs, β_filtered)
        push!(τs, τ)
        push!(a0s, a0)
    end
    return KnockoffFilter(y, X, X̃, W, βs, a0s, τs, fdrs, d, debias)
end

function debias!(
    β̂::AbstractVector{T},
    x::AbstractMatrix{T},
    y::AbstractVector{T};
    method=:ls, # :ls or :lasso
    d::Distribution=Normal(),
    kwargs... # extra arguments for glmnetcv
    ) where T
    zero_idx = β̂ .== 0
    if method == :lasso
        # Give infinite penalty to indices of zeros
        penalty_factor = ones(T, length(β̂))
        @view(penalty_factor[zero_idx]) .= typemax(T)
        # run cross validated lasso
        cv = glmnetcv(x, y, penalty_factor=penalty_factor; kwargs...)
        # refit lasso on best performing lambda and extract resulting beta/intercept
        λbest = cv.lambda[argmin(cv.meanloss)]
        best_fit = glmnet(x, y, lambda=[λbest], penalty_factor=penalty_factor)
        copyto!(β̂, best_fit.betas)
        intercept = best_fit.a0[1]
    elseif method == :ls
        nonzero_idx = findall(!iszero, β̂)
        Xsubset = [ones(T, size(x, 1)) x[:, nonzero_idx]]
        model = glm(Xsubset, y, d)
        β_debiased = GLM.coef(model)
        intercept = β_debiased[1]
        β̂[nonzero_idx] .= @view(β_debiased[2:end])
    else
        error("method should be :ls or :lasso but was $method")
    end
    all(x -> x ≈ zero(T), @view(β̂[zero_idx])) ||
        error("Debiasing error: a zero index has non-zero coefficient")
    return intercept
end

# According to GLMNet.jl documentation https://github.com/JuliaStats/GLMNet.jl
# y needs to be a m by 2 matrix, where the first column is the count of negative responses for each row in X and the second column is the count of positive responses.
function form_glmnet_logistic_y(y::AbstractVector{T}) where T
    sort!(unique(y)) == [zero(T), one(T)] || error("y should have values 0 and 1 only")
    glmnet_y = [y .== 0 y .== 1] |> Matrix{T}
    return glmnet_y
end
