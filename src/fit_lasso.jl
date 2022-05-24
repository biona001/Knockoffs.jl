"""
    fit_lasso()

Generates knockoffs, runs Lasso with GLMNet, then applies the knockoff-filter.

# Inputs
+ method: `:knockoff` or `:knockoff_plus`
"""
function fit_lasso(
    y::AbstractVector{T},
    X::AbstractMatrix{T}, 
    X̃::AbstractMatrix{T};
    d::Distribution=Normal(),
    fdrs::Vector{Float64}=[0.01, 0.05, 0.1, 0.25, 0.5],
    groups::Union{Nothing, AbstractVector{Int}} = nothing,
    filter_method::Symbol = :knockoff, # `:knockoff` or `:knockoff_plus`
    debias::Bool = true,
    kwargs..., # arguments for glmnetcv
    ) where T <: AbstractFloat
    # fit lasso (note: need to interleaves X with X̃)
    XX̃, original, knockoff = merge_knockoffs_with_original(X, X̃)
    knockoff_cv = glmnetcv(XX̃, y, d, kwargs...)
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
        debias && (a0 = debias!(β_filtered, X, y))
        # save knockoff statistics
        push!(βs, β_filtered)
        push!(τs, τ)
        push!(a0s, a0)
    end
    return KnockoffFilter(XX̃, original, knockoff, W, βs, a0s, τs, fdrs, debias)
end

function debias!(
    β̂::AbstractVector{T},
    x::AbstractMatrix,
    y::AbstractVector;
    kwargs... # extra arguments for glmnetcv
    ) where T
    # for debiasing, lasso can only have non-0 entries on the support of β̂
    zero_idx = β̂ .== 0
    penalty_factor = ones(T, length(β̂))
    @view(penalty_factor[zero_idx]) .= typemax(T)
    # run cross validated lasso
    cv = glmnetcv(x, y, penalty_factor=penalty_factor, kwargs...)
    # refit lasso on best performing lambda and extract resulting beta/intercept
    λbest = cv.lambda[argmin(cv.meanloss)]
    best_fit = glmnet(x, y, lambda=[λbest], penalty_factor=penalty_factor)
    β̂ .= best_fit.betas
    intercept = best_fit.a0[1]
    # β̂ .= cv.path.betas[:, argmin(cv.meanloss)]
    # intercept = cv.path.a0[argmin(cv.meanloss)]
    sum(@view(β̂[zero_idx])) ≈ zero(T) || 
        error("Debiasing error: a zero index has non-zero coefficient")
    return intercept
end

# function debias(x::AbstractMatrix, y::AbstractVector, method::Symbol=:ls)
#     if method == :ls
#         return x \ y # intercept must be in x already
#     elseif method == :lasso
#         cv = glmnetcv(x, y, alpha=1.0) # 0.0 = ridge regression, 1.0 = lasso, in between = elastic net
#         β_lasso = copy(cv.path.betas[:, argmin(cv.meanloss)])
#         lasso_intercept = cv.path.a0[argmin(cv.meanloss)]
#     elseif method == :ridge
#         cv = glmnetcv(x, y, alpha=0.0) # 0.0 = ridge regression, 1.0 = lasso, in between = elastic net
#         β_ridge = copy(cv.path.betas[:, argmin(cv.meanloss)])
#         ridge_intercept = cv.path.a0[argmin(cv.meanloss)]
#     elseif method == :elastic_net
#         cv = glmnetcv(x, y, alpha=0.5) # 0.0 = ridge regression, 1.0 = lasso, in between = elastic net
#         β_elastic = copy(cv.path.betas[:, argmin(cv.meanloss)])
#         elastic_intercept = cv.path.a0[argmin(cv.meanloss)]
#     else
#         error("method should be :ls, :lasso, :ridge, :elastic_net, but got $method")
#     end
# end
