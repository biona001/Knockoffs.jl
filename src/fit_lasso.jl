"""
    fit_lasso(y, X, method=:maxent, ...)
    fit_lasso(y, X, μ, Σ, method=:maxent, ...)

Generates model-X knockoffs with `method`, runs Lasso, then applies the 
knockoff-filter. If `μ` and `Σ` are not provided, they will be estimated from
data. 

# Inputs
+ `y`: A `n × 1` response vector
+ `X`: A `n × p` numeric matrix, each row is a sample, and each column is covariate.
+ `method`: Method for knockoff generation (defaults to `:maxent`)
+ `μ`: A `p × 1` vector of column mean of `X`. If not provided, defaults to column mean.
+ `Σ`: A `p × p` covariance matrix of `X`. If not provided, it will be estimated 
    based on a shrinked empirical covariance matrix, see [`modelX_gaussian_knockoffs`](@ref)
+ `d`: Distribution of response. Defaults `Normal()`, for binary response
    (logistic regression) use `Binomial()`.
+ `m`: Number of simultaneous knockoffs to generate, defaults to `m=1`
+ `fdrs`: Target FDRs, defaults to `[0.01, 0.05, 0.1, 0.25, 0.5]`
+ `filter_method`: Choices are `:knockoff` or `:knockoff_plus` (default) 
+ `debias`: Defines how the selected coefficients are debiased. Specify `:ls` 
    for least squares or `:lasso` for Lasso (only running on the 
    support). To not debias, specify `debias=nothing` (default).
+ `kwargs`: Additional arguments to input into `glmnetcv` and `glmnet`
"""
function fit_lasso(
    y::AbstractVector{T},
    X::AbstractMatrix{T};
    method::Symbol = :maxent,
    d::Distribution=Normal(),
    m::Int = 1,
    fdrs::Vector{Float64}=[0.01, 0.05, 0.1, 0.25, 0.5],
    groups::Union{Nothing, AbstractVector{Int}} = nothing,
    filter_method::Symbol = :knockoff_plus,
    debias::Union{Nothing, Symbol} = nothing,
    kwargs..., # arguments for glmnetcv
    ) where T
    ko = isnothing(groups) ? modelX_gaussian_knockoffs(X, method, m=m) : 
        modelX_gaussian_group_knockoffs(X, method, groups, m=m)
    return fit_lasso(y, X, ko, d=d, fdrs=fdrs, 
        filter_method=filter_method, debias=debias; kwargs...)
end

function fit_lasso(
    y::AbstractVector{T},
    X::AbstractMatrix{T},
    μ::AbstractVector{T},
    Σ::AbstractMatrix{T};
    method::Symbol = :maxent,
    d::Distribution=Normal(),
    m::Int = 1,
    fdrs::Vector{Float64}=[0.01, 0.05, 0.1, 0.25, 0.5],
    groups::Union{Nothing, AbstractVector{Int}} = nothing,
    filter_method::Symbol = :knockoff_plus,
    debias::Union{Nothing, Symbol} = :ls,
    kwargs..., # arguments for glmnetcv
    ) where T
    ko = isnothing(groups) ? modelX_gaussian_knockoffs(X, method, μ, Σ, m=m) : 
        modelX_gaussian_group_knockoffs(X, method, groups, μ, Σ; m=m)
    return fit_lasso(y, X, ko, d=d, fdrs=fdrs, 
        filter_method=filter_method, debias=debias; kwargs...)
end

function fit_lasso(
    y::AbstractVector{T},
    X::AbstractMatrix{T}, 
    ko::Knockoff;
    d::Distribution=Normal(),
    fdrs::Vector{Float64}=[0.01, 0.05, 0.1, 0.25, 0.5],
    filter_method::Symbol = :knockoff_plus, # `:knockoff` or `:knockoff_plus`
    debias::Union{Nothing, Symbol} = nothing,
    stringent::Bool = false,
    kwargs..., # arguments for glmnetcv
    ) where T <: AbstractFloat
    ytmp = d == Binomial() ? form_glmnet_logistic_y(y) : y
    X̃ = ko.X̃
    m = Int(size(X̃, 2) / size(X, 2)) # number of knockoffs per feature
    # merge X with its knockoffs X̃ and shuffle around the indices
    merged_ko = merge_knockoffs_with_original(X, X̃)
    # cross validate for λ, then refit Lasso with best λ
    knockoff_cv = glmnetcv(merged_ko.XX̃, ytmp, d; kwargs...)
    λbest = knockoff_cv.lambda[argmin(knockoff_cv.meanloss)]
    best_fit = glmnet(merged_ko.XX̃, y, lambda=[λbest])
    βestim = vec(best_fit.betas) |> Vector{T}
    a0 = best_fit.a0[1]
    # check if groups exist (todo: do I really need groups_full defined)
    groups = nothing
    if hasproperty(ko, :groups)
        groups = ko.groups # group membership of original variables
        groups_full = repeat(groups, inner=m+1) # since X and X̃ is interleaved, each group length is repeated m times
    end
    # compute feature importance statistics and allocate necessary knockoff-filter variables
    βs, a0s = Vector{T}[], T[]
    for fdr in fdrs
        # apply knockoff-filter based on target fdr
        β_filtered = isnothing(groups) ? 
            extract_beta(βestim, fdr, merged_ko.original, merged_ko.knockoff, filter_method) : 
            extract_beta(βestim, fdr, groups_full, merged_ko.original, merged_ko.knockoff, filter_method)
        # debias the estimates if requested
        if !isnothing(debias) && count(!iszero, β_filtered) > 0
            a0 = isnothing(groups) ? 
                debias!(β_filtered, X, y; method=debias, d=d, kwargs...) : 
                debias!(β_filtered, X, y, groups; method=debias, d=d, stringent=stringent, kwargs...)
        end
        # save beta and intercept
        push!(βs, β_filtered)
        push!(a0s, a0)
    end
    return KnockoffFilter(y, X, ko, merged_ko, m, βs, a0s, fdrs, d, debias)
end

# for group representative variant method
function fit_lasso(
    y::AbstractVector{T},
    X::AbstractMatrix{T}, 
    ko::GaussianRepGroupKnockoff;
    d::Distribution=Normal(),
    fdrs::Vector{Float64}=[0.01, 0.05, 0.1, 0.25, 0.5],
    filter_method::Symbol = :knockoff_plus, # `:knockoff` or `:knockoff_plus`
    debias::Union{Nothing, Symbol} = nothing,
    kwargs..., # arguments for glmnetcv
    ) where T <: AbstractFloat
    ytmp = d == Binomial() ? form_glmnet_logistic_y(y) : y
    X̃ = ko.ko.X̃
    m = Int(size(X̃, 2) / length(ko.groups_reps)) # number of knockoffs per feature
    # merge X with its knockoffs X̃ and shuffle around the indices
    Xrep = @view(X[:, ko.groups_reps])
    merged_ko = merge_knockoffs_with_original(Xrep, X̃)
    # cross validate for λ, then refit Lasso with best λ
    knockoff_cv = glmnetcv(merged_ko.XX̃, ytmp, d; kwargs...)
    λbest = knockoff_cv.lambda[argmin(knockoff_cv.meanloss)]
    best_fit = glmnet(merged_ko.XX̃, y, lambda=[λbest])
    βestim = vec(best_fit.betas) |> Vector{T}
    a0 = best_fit.a0[1]
    # compute feature importance statistics and allocate necessary knockoff-filter variables
    βs, a0s = Vector{T}[], T[]
    for fdr in fdrs
        # apply knockoff-filter based on target fdr
        β_filtered = extract_beta(βestim, fdr, merged_ko.original, merged_ko.knockoff, filter_method)
        # debias the estimates if requested
        if !isnothing(debias) && count(!iszero, β_filtered) > 0
            a0 = debias!(β_filtered, ko.ko.X, y; method=debias, d=d, kwargs...)
        end
        # save beta and intercept
        β_filtered_full = zeros(T, size(X, 2))
        β_filtered_full[ko.groups_reps] .= β_filtered
        push!(βs, β_filtered_full)
        push!(a0s, a0)
    end
    return KnockoffFilter(y, X, ko, merged_ko, m, βs, a0s, fdrs, d, debias)
end

function debias!(
    β̂::AbstractVector{T},
    x::AbstractMatrix{T},
    y::AbstractVector{T};
    method=:ls, # :ls or :lasso
    d::Distribution=Normal(),
    kwargs... # extra arguments for glmnetcv
    ) where T
    count(!iszero, β̂) == 0 && error("β̂ is all zeros! Nothing to debias!")
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

function debias!(
    β̂::AbstractVector{T},
    x::AbstractMatrix{T},
    y::AbstractVector{T},
    groups::AbstractVector{Int};
    method=:ls, # :ls or :lasso
    stringent::Bool=false,
    d::Distribution=Normal(),
    kwargs... # extra arguments for glmnetcv
    ) where T
    p = length(β̂)
    p == length(groups) || error("check vector length") # note GLMNet.jl does not include intercept in β̂
    count(!iszero, β̂) == 0 && error("β̂ is all zeros! Nothing to debias!")
    # first find active groups
    if stringent
        # active variables can only be non-zero variables within active group
        active_vars = findall(!iszero, β̂)
    else
        # active variables can be every variables within active group
        active_vars = Int[]
        active_groups = unique(groups[findall(!iszero, β̂)])
        for g in active_groups
            vars = findall(x -> x == g, groups)
            for var in vars
                push!(active_vars, var)
            end
        end
    end
    # now debias on the group level
    zero_idx = setdiff(1:p, active_vars)
    if method == :lasso
        Xsubset = x[:, active_vars]
        # run cross validated lasso
        cv = glmnetcv(Xsubset, y; kwargs...)
        # refit lasso on best performing lambda and extract resulting beta/intercept
        λbest = cv.lambda[argmin(cv.meanloss)]
        best_fit = glmnet(Xsubset, y, lambda=[λbest])
        β̂[active_vars] .= best_fit.betas[:]
        intercept = best_fit.a0[1]
    elseif method == :ls
        Xsubset = [ones(T, size(x, 1)) x[:, active_vars]]
        model = glm(Xsubset, y, d)
        β_debiased = GLM.coef(model)
        intercept = β_debiased[1]
        β̂[active_vars] .= @view(β_debiased[2:end])
    else
        error("method should be :ls or :lasso but was $method")
    end
    all(x -> x ≈ zero(T), @view(β̂[zero_idx])) ||
        error("Group debiasing error: a zero index has non-zero coefficient")
    return intercept
end

# According to GLMNet.jl documentation https://github.com/JuliaStats/GLMNet.jl
# y needs to be a m by 2 matrix, where the first column is the count of negative responses for each row in X and the second column is the count of positive responses.
function form_glmnet_logistic_y(y::AbstractVector{T}) where T
    sort!(unique(y)) == [zero(T), one(T)] || error("y should have values 0 and 1 only")
    glmnet_y = [y .== 0 y .== 1] |> Matrix{T}
    return glmnet_y
end
