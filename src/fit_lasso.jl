"""
    fit_lasso(y, X, [method], [d], [m], [groups], [filter_method], 
        [debias], [kwargs...])
    fit_lasso(y, X, μ, Σ, [method], [d], [m], [groups], [filter_method], 
        [debias], [kwargs...])

Generates model-X knockoffs with `method`, runs Lasso, then applies the 
knockoff filter. If `μ` and `Σ` are not provided, they will be estimated from
data. Feature selection is done on demand via q-values with
`select_variables(model, q)` and `selected_coefficients(model, q)`.

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
+ `groups`: Vector of group membership. If not supplied, we generate regular knockoffs.
    If supplied, we run group knockoffs.
+ `filter_method`: Choices are `:knockoff` or `:knockoff_plus` (default) 
+ `debias`: Defines how the selected coefficients are debiased. Specify `:ls` 
    for least squares or `:lasso` for Lasso (only running on the 
    support). To not debias, specify `debias=nothing` (default).
+ `kwargs`: Additional arguments to input into `glmnetcv` and `glmnet`
"""
function fit_lasso(
    y::AbstractVecOrMat{T},
    X::AbstractMatrix{T};
    method::Union{Symbol,String} = :maxent,
    d::Distribution=Normal(),
    m::Number = 1,
    groups::Union{Nothing, AbstractVector{Int}} = nothing,
    filter_method::Symbol = :knockoff_plus,
    debias::Union{Nothing, Symbol} = nothing,
    kwargs..., # arguments for glmnetcv
    ) where T
    ko = isnothing(groups) ? modelX_gaussian_knockoffs(X, method, m=m) : 
        modelX_gaussian_group_knockoffs(X, method, groups, m=m)
    return fit_lasso(y, ko, d=d,
        filter_method=filter_method, debias=debias; kwargs...)
end

function fit_lasso(
    y::AbstractVecOrMat{T},
    X::AbstractMatrix{T},
    μ::AbstractVector{T},
    Σ::AbstractMatrix{T};
    method::Union{Symbol,String} = :maxent,
    d::Distribution=Normal(),
    m::Number = 1,
    groups::Union{Nothing, AbstractVector{Int}} = nothing,
    filter_method::Symbol = :knockoff_plus,
    debias::Union{Nothing, Symbol} = :ls,
    kwargs..., # arguments for glmnetcv
    ) where T
    ko = isnothing(groups) ? modelX_gaussian_knockoffs(X, method, μ, Σ, m=m) : 
        modelX_gaussian_group_knockoffs(X, method, groups, μ, Σ; m=m)
    return fit_lasso(y, ko, d=d,
        filter_method=filter_method, debias=debias; kwargs...)
end

function fit_lasso(
    y::AbstractVecOrMat{T},
    ko::Knockoff;
    d::Distribution=Normal(),
    filter_method::Symbol = :knockoff_plus, # `:knockoff` or `:knockoff_plus`
    debias::Union{Nothing, Symbol} = nothing,
    stringent::Bool = false,
    kwargs..., # arguments for glmnetcv
    ) where T <: AbstractFloat
    typeof(y) <: AbstractMatrix && (y = vec(y))
    ytmp = d == Binomial() ? form_glmnet_logistic_y(y) : y
    X = ko.X
    X̃ = ko.Xko
    m = ko.m
    p = size(X, 2)
    # cross validate for λ, then refit Lasso with best λ
    XX̃ = hcat(X, X̃)
    perm = collect(1:(m+1)*p) |> shuffle!
    XX̃ .= @view(XX̃[:, perm]) # permute columns of XX̃ so there's no ordering bias
    knockoff_cv = glmnetcv(XX̃, ytmp, d; kwargs...)
    λbest = knockoff_cv.lambda[argmin(knockoff_cv.meanloss)]
    best_fit = glmnet(XX̃, y, lambda=[λbest])
    βestim = vec(best_fit.betas) |> Vector{T}
    βestim .= @view(βestim[invperm(perm)]) # permute beta back
    a0 = best_fit.a0[1]
    # feature importance statistics
    T0 = βestim[1:p]
    Tk = m > 1 ? [βestim[k*p+1:(k+1)*p] for k in 1:m] : βestim[p+1:end]
    group_labels_for_stats = nothing
    if hasproperty(ko, :groups)
        groups = ko.groups
        unique_groups = unique(groups)
        group_labels_for_stats = unique_groups
        T0_group = T[]
        Tk_group = m > 1 ? [T[] for k in 1:m] : T[]
        for g in unique_groups
            idx = findall(x -> x == g, groups)
            push!(T0_group, sum(abs, @view(T0[idx])))# / length(idx))
            if m > 1
                for k in 1:m
                    push!(Tk_group[k], sum(abs, @view(Tk[k][idx])))# / length(idx))
                end
            else
                push!(Tk_group, sum(abs, @view(Tk[idx])))# / length(idx))
            end
        end
        if m > 1
            κ, τ, W = MK_statistics(T0_group, Tk_group)
        else
            W = MK_statistics(T0_group, Tk_group)
        end
    else # no groups, compute regular knockoff statistics
        if m > 1
            κ, τ, W = MK_statistics(T0, Tk)
        else
            W = MK_statistics(T0, Tk)
        end
    end
    qvalues = if m > 1
        isnothing(group_labels_for_stats) ? get_knockoff_qvalue(κ, τ, m) :
            get_knockoff_qvalue(κ, τ, m; groups=group_labels_for_stats)
    else
        get_knockoff_qvalue(W; method=filter_method)
    end
    return LassoKnockoffFilter(
        y, X, ko, Int(m), T0, a0, W, qvalues, group_labels_for_stats, d, debias, stringent)
end

"""
    select_variables(model::KnockoffFilter, q::Real)

Select statistics whose q-value is at most `q`.
"""
function select_variables(model::KnockoffFilter, q::Real)
    0 ≤ q ≤ 1 || error("Target FDR should be between 0 and 1 but got $q")
    return findall(x -> x ≤ q, model.qvalues)
end

"""
    select_groups(model::KnockoffFilter, q::Real)

Return selected groups and variable indices in each selected group.

# Output
+ `selected_groups`: Group labels selected at q-value level `q`
+ `selected_group_vars`: `selected_group_vars[i]` stores variable indices in
    `selected_groups[i]`
"""
function select_groups(model::KnockoffFilter, q::Real)
    hasproperty(model.ko, :groups) ||
        error("select_groups requires a grouped knockoff model.")
    selected_stats = select_variables(model, q)
    selected_groups = isnothing(model.stat_groups) ? selected_stats :
        model.stat_groups[selected_stats]
    selected_groups = unique(selected_groups)
    selected_group_vars = [findall(isequal(g), model.ko.groups) for g in selected_groups]
    return selected_groups, selected_group_vars
end

Base.@deprecate selected_variables(model::KnockoffFilter, q::Real) select_variables(model, q)

"""
    selected_coefficients(model::LassoKnockoffFilter, q::Real; [debias], [kwargs...])

Return thresholded coefficients/intercept corresponding to q-value level `q`.
"""
function selected_coefficients(
    model::LassoKnockoffFilter{T},
    q::Real;
    debias::Union{Nothing, Symbol}=model.debias,
    kwargs...,
    ) where T
    β_filtered = zeros(T, length(model.beta))
    selected_stats = select_variables(model, q)
    if hasproperty(model.ko, :groups)
        groups = model.ko.groups
        selected_groups = isnothing(model.stat_groups) ? selected_stats :
            model.stat_groups[selected_stats]
        active_vars = Int[]
        for g in selected_groups
            append!(active_vars, findall(isequal(g), groups))
        end
        β_filtered[active_vars] .= model.beta[active_vars]
    else
        β_filtered[selected_stats] .= model.beta[selected_stats]
    end
    a0 = model.a0
    if !isnothing(debias) && count(!iszero, β_filtered) > 0
        if hasproperty(model.ko, :groups)
            a0 = debias!(β_filtered, model.X, model.y, model.ko.groups;
                method=debias, d=model.d, stringent=model.stringent, kwargs...)
        else
            a0 = debias!(β_filtered, model.X, model.y; method=debias, d=model.d, kwargs...)
        end
    end
    return β_filtered, a0
end

"""
    fit_marginal(y, X, method=:maxent, ...)
    fit_marginal(y, X, μ, Σ, method=:maxent, ...)

Generates model-X knockoffs with `method` and computes feature importance statistics
based on squared marginal Z score: abs2(x[:, i]^t*y) / n. If `μ` and `Σ` are not
provided, they will be estimated from data. 

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
+ `groups`: Vector of group membership. If not supplied, we generate regular knockoffs.
    If supplied, we run group knockoffs.
+ `filter_method`: Choices are `:knockoff` or `:knockoff_plus` (default) 
+ `debias`: Defines how the selected coefficients are debiased. Specify `:ls` 
    for least squares or `:lasso` for Lasso (only running on the 
    support). To not debias, specify `debias=nothing` (default).
+ `kwargs`: Additional arguments to input into `glmnetcv` and `glmnet`
"""
function fit_marginal(
    y::AbstractVecOrMat{T},
    X::AbstractMatrix{T};
    method::Union{Symbol,String} = :maxent,
    d::Distribution=Normal(),
    m::Number = 1,
    groups::Union{Nothing, AbstractVector{Int}} = nothing,
    filter_method::Symbol = :knockoff_plus,
    kwargs..., # arguments for glmnetcv
    ) where T
    ko = isnothing(groups) ? modelX_gaussian_knockoffs(X, method, m=m) : 
        modelX_gaussian_group_knockoffs(X, method, groups, m=m)
    return fit_marginal(y, ko, d=d,
        filter_method=filter_method; kwargs...)
end

function fit_marginal(
    y::AbstractVecOrMat{T},
    ko::Knockoff;
    d::Distribution=Normal(),
    filter_method::Symbol = :knockoff_plus, # `:knockoff` or `:knockoff_plus`
    ) where T <: AbstractFloat
    typeof(y) <: AbstractMatrix && (y = vec(y))
    X = ko.X
    X̃ = ko.Xko
    m = ko.m
    n, p = size(X)
    # compute feature importance statistics (squared marginal Z score abs2.(X'y)/n)
    y_std = zscore(y, mean(y), std(y))
    X_std = zscore(X, mean(X, dims=1), std(X, dims=1))
    X̃_std = zscore(X̃, mean(X̃, dims=1), std(X̃, dims=1))
    T0 = abs2.(X_std' * y_std) ./ n
    if m > 1
        Tk = [abs2.(Transpose(@view(X̃_std[:, (k-1)*p+1:k*p]))*y_std) ./ n for k in 1:m]
    else
        Tk = abs2.(X̃_std'*y_std) ./ n
    end
    group_labels_for_stats = nothing
    if hasproperty(ko, :groups)
        groups = ko.groups
        unique_groups = unique(groups)
        group_labels_for_stats = unique_groups
        T0_group = T[]
        Tk_group = m > 1 ? [T[] for k in 1:m] : T[]
        for g in unique_groups
            idx = findall(x -> x == g, groups)
            push!(T0_group, sum(abs, @view(T0[idx])))# / length(idx))
            if m > 1
                for k in 1:m
                    push!(Tk_group[k], sum(abs, @view(Tk[k][idx])))# / length(idx))
                end
            else
                push!(Tk_group, sum(abs, @view(Tk[idx])))# / length(idx))
            end
        end
        if m > 1
            κ, τ, W = MK_statistics(T0_group, Tk_group)
        else
            W = MK_statistics(T0_group, Tk_group)
        end
    else # no groups, compute regular knockoff statistics
        if m > 1
            κ, τ, W = MK_statistics(T0, Tk)
        else
            W = MK_statistics(T0, Tk)
        end
    end
    qvalues = if m > 1
        isnothing(group_labels_for_stats) ? get_knockoff_qvalue(κ, τ, m) :
            get_knockoff_qvalue(κ, τ, m; groups=group_labels_for_stats)
    else
        get_knockoff_qvalue(W; method=filter_method)
    end
    return MarginalKnockoffFilter(y, X, ko, W, qvalues, Int(m), group_labels_for_stats, d)
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
function form_glmnet_logistic_y(y::AbstractVecOrMat{T}) where T
    sort!(unique(y)) == [zero(T), one(T)] || error("y should have values 0 and 1 only")
    glmnet_y = [y .== 0 y .== 1] |> Matrix{T}
    return glmnet_y
end
