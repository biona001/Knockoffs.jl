function debias(x::AbstractMatrix, y::AbstractVector, method::Symbol=:ls)
    if method == :ls
        return x \ y # intercept must be in x already
    elseif method == :lasso
        cv = glmnetcv(x, y, alpha=1.0) # 0.0 = ridge regression, 1.0 = lasso, in between = elastic net
        β_lasso = copy(cv.path.betas[:, argmin(cv.meanloss)])
        lasso_intercept = cv.path.a0[argmin(cv.meanloss)]
    elseif method == :ridge
        cv = glmnetcv(x, y, alpha=0.0) # 0.0 = ridge regression, 1.0 = lasso, in between = elastic net
        β_ridge = copy(cv.path.betas[:, argmin(cv.meanloss)])
        ridge_intercept = cv.path.a0[argmin(cv.meanloss)]
    elseif method == :elastic_net
        cv = glmnetcv(x, y, alpha=0.5) # 0.0 = ridge regression, 1.0 = lasso, in between = elastic net
        β_elastic = copy(cv.path.betas[:, argmin(cv.meanloss)])
        elastic_intercept = cv.path.a0[argmin(cv.meanloss)]
    else
        error("method should be :ls, :lasso, :ridge, :elastic_net, but got $method")
    end
end
