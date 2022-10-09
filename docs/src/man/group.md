
# Group Knockoffs

This tutorial generates group (model-X) knockoffs, which is useful when predictors are highly correlated. The methodology is described in the following paper

> Dai R, Barber R. The knockoff filter for FDR control in group-sparse and multitask regression. InInternational conference on machine learning 2016 Jun 11 (pp. 1851-1859). PMLR.


!!! note

    In the original paper, Dai and Barber only describes how to construct a suboptimal equi-correlated group knockoffs. Here we implement fully generalized alternatives.
    
Currently available options for group knockoffs:
+ `:mvr`: Fully general minimum variance-based reconstructability (MVR) group knockoff, based on coordinate descent.
+ `:maxent`: Fully general maximum entropy (maxent) group knockoff, based on coordinate descent.
+ `:equi`: This implements the equi-correlated idea proposed in [Barber and Dai](https://proceedings.mlr.press/v48/daia16.html), which lets $S_j = \gamma \Sigma_{(G_j, G_j)}$ where $\Sigma_{(G_j, G_j)}$ is the block of $\Sigma$ containing variables in the $j$th group. Thus, instead of optimizing over all variables in $S$, we optimize a scalar $\gamma$. Conveniently, there a simple closed form solution for $\gamma$. For `mvr` and `maxent` group knockoffs, we initialize $S$ using this construction. 
+ `:SDP`: This generalizes the equi-correlated group knockoff idea by having $S_j = \gamma_j \Sigma_{(G_j, G_j)}$. Instead of optimizing over all variables in $S$, we optimize over a vector $\gamma_1,...,\gamma_G$. 



```julia
# load packages for this tutorial
using Revise
using Knockoffs
using LinearAlgebra
using Random
using StatsBase
using Statistics
using ToeplitzMatrices

# some helper functions to compute power and empirical FDR
function TP(correct_groups, signif_groups)
    return length(signif_groups ∩ correct_groups) / length(correct_groups)
end
function TP(correct_groups, β̂, groups)
    signif_groups = get_signif_groups(β̂, groups)
    return TP(correct_groups, signif_groups)
end
function FDR(correct_groups, signif_groups)
    FP = length(signif_groups) - length(signif_groups ∩ correct_groups) # number of false positives
    FDR = FP / max(1, length(signif_groups))
    return FDR
end
function FDR(correct_groups, β̂, groups)
    signif_groups = get_signif_groups(β̂, groups)
    return FDR(correct_groups, signif_groups)
end
function get_signif_groups(β, groups)
    correct_groups = Int[]
    for i in findall(!iszero, β)
        g = groups[i]
        g ∈ correct_groups || push!(correct_groups, g)
    end
    return correct_groups
end
```




    get_signif_groups (generic function with 1 method)



# Constructing group knockoffs

First, let's simulate data and generate equi-correlated knockoffs. Our true covariance matrix looks like

```math
\begin{aligned}
\Sigma = 
\begin{pmatrix}
    1 & \rho & \rho^2 & ... & \rho^p\\
    \rho & 1 & & ... & \rho^{p-1}\\
    \vdots & & & 1 & \vdots \\
    \rho^p & \cdots & & & 1
\end{pmatrix}, \quad \rho = 0.9
\end{aligned}
```

Because variables are highly correlated with its neighbors ($\rho = 0.9$), it becomes difficult to distinguish which variables among a group are truly causal. Thus, group knockoffs which test whether a *group* of variables have any signal should have better power than standard (single-variable) knockoffs. 

For simplicity, let simulate data where every 5 variables form a group:


```julia
# simulate data
Random.seed!(2022)
n = 1000 # sample size
p = 200  # number of covariates
k = 10   # number of true predictors
Σ = Matrix(SymmetricToeplitz(0.4.^(0:(p-1)))) # true covariance matrix
groupsizes = [5 for i in 1:div(p, 5)] # each group has 5 variables
groups = vcat([i*ones(g) for (i, g) in enumerate(groupsizes)]...) |> Vector{Int}
true_mu = zeros(p)
L = cholesky(Σ).L
X = randn(n, p) * L
zscore!(X, mean(X, dims=1), std(X, dims=1)); # standardize columns of X
```

Generate group knockoffs with the exported function `modelX_gaussian_group_knockoffs`. Similar to non-group knockoffs, group knockoff accepts keyword arguments `m`, `tol`, `niter`, and `verbose` which controls the algorithm's behavior. 


```julia
Gme = modelX_gaussian_group_knockoffs(
    X, :maxent, groups, true_mu, Σ, 
    m = 1,          # number of knockoffs per variable to generate
    tol = 0.0001,   # convergence tolerance
    niter = 100,    # max number of coordinate descent iterations
    verbose=true);  # whether to print informative intermediate results
```

    Iter 1: δ = 0.35041635762964374, t1 = 0.07, t2 = 0.01, t3 = 0.0
    Iter 2: δ = 0.06174971923894666, t1 = 0.07, t2 = 0.02, t3 = 0.0
    Iter 3: δ = 0.020679713293120027, t1 = 0.08, t2 = 0.04, t3 = 0.0
    Iter 4: δ = 0.010800522507824841, t1 = 0.09, t2 = 0.05, t3 = 0.0
    Iter 5: δ = 0.006051906091741958, t1 = 0.09, t2 = 0.06, t3 = 0.0
    Iter 6: δ = 0.003458575128440409, t1 = 0.1, t2 = 0.07, t3 = 0.0
    Iter 7: δ = 0.0016285523099343307, t1 = 0.1, t2 = 0.09, t3 = 0.0
    Iter 8: δ = 0.0007940739052498399, t1 = 0.11, t2 = 0.1, t3 = 0.0
    Iter 9: δ = 0.0003768922472274655, t1 = 0.11, t2 = 0.11, t3 = 0.0
    Iter 10: δ = 0.00015544193550259136, t1 = 0.12, t2 = 0.12, t3 = 0.0
    Iter 11: δ = 0.00014451264472101048, t1 = 0.12, t2 = 0.14, t3 = 0.0
    Iter 12: δ = 3.6816774372557906e-5, t1 = 0.12, t2 = 0.15, t3 = 0.0


Note $t1, t2, t3$ are timers which corresponds to (1) updating cholesky factors, (2) solving forward-backward equations, and (3) solving off-diagonal 1D optimization problems using Brent's method. As we can see, the computational bottleneck in (2), which we dispatch to efficient LAPACK libraries. 

The output is a struct with the following fields
```julia
struct GaussianGroupKnockoff{T<:AbstractFloat, BD<:AbstractMatrix, S<:Symmetric} <: Knockoff
    X::Matrix{T} # n × p design matrix
    X̃::Matrix{T} # n × p knockoff of X
    groups::Vector{Int} # p × 1 vector of group membership
    S::BD # p × p block-diagonal matrix of the same size as Σ. S and 2Σ - S are both psd
    γs::Vector{T} # scalars chosen so that 2Σ - S is positive definite where S_i = γ_i * Σ_i
    Σ::S # p × p symmetric covariance matrix. 
    method::Symbol # method for solving s
end
```
Given this result, lets do a sanity check: is $2\Sigma - S$ positive semi-definite?


```julia
# compute minimum eigenvalues of 2Σ - S
eigmin(2Gme.Σ - Gme.S)
```




    0.417481414774321



## Second order group knockoffs

In practice, we often do not have the true covariance matrix $\Sigma$ and the true means $\mu$. In that case, we can generate second order group knockoffs via the 3 argument function


```julia
Gme_second_order = modelX_gaussian_group_knockoffs(X, :maxent, groups);
```

This will estimate the covariance matrix via a shrinkage estimator, see documentation API for more details. 

## Lasso Example

Lets see the empirical power and FDR group knockoffs over 10 simulations when the targer FDR is 10%. Here power and FDR is defined at the group level. 


```julia
target_fdr = 0.1
group_powers, group_fdrs, group_times, group_s = Float64[], Float64[], Float64[], Float64[]

Random.seed!(2022)
for sim in 1:10
    # simulate X
    Random.seed!(sim)
    n = 1000 # sample size
    p = 200  # number of covariates
    k = 10   # number of true predictors
    Σ = Matrix(SymmetricToeplitz(0.9.^(0:(p-1)))) # true covariance matrix
    groupsizes = [5 for i in 1:div(p, 5)] # each group has 5 variables
    groups = vcat([i*ones(g) for (i, g) in enumerate(groupsizes)]...) |> Vector{Int}
    true_mu = zeros(p)
    L = cholesky(Σ).L
    X = randn(n, p) * L
    zscore!(X, mean(X, dims=1), std(X, dims=1)); # standardize columns of X

    # simulate y
    βtrue = zeros(p)
    βtrue[1:k] .= rand(-1:2:1, k) .* 0.1
    shuffle!(βtrue)
    correct_groups = get_signif_groups(βtrue, groups)
    ϵ = randn(n)
    y = X * βtrue + ϵ;

    # group MVR knockoffs
    t = @elapsed ko_filter = fit_lasso(y, X, method=:maxent, groups=groups)
    power = round(TP(correct_groups, ko_filter.βs[idx], groups), digits=3)
    fdr = round(FDR(correct_groups, ko_filter.βs[idx], groups), digits=3)
    println("Sim $sim group-knockoff power = $power, FDR = $fdr, time=$t")
    push!(group_powers, power); push!(group_fdrs, fdr); push!(group_times, t)
    GC.gc();GC.gc();GC.gc();
end

println("\nME group knockoffs have average group power $(mean(group_powers))")
println("ME group knockoffs have average group FDR $(mean(group_fdrs))")
println("ME group knockoffs took average $(mean(group_times)) seconds");
```

    Sim 1 group-knockoff power = 0.0, FDR = 0.0, time=5.356796708
    Sim 2 group-knockoff power = 0.1, FDR = 0.0, time=3.846174709
    Sim 3 group-knockoff power = 0.222, FDR = 0.0, time=2.418540375
    Sim 4 group-knockoff power = 0.4, FDR = 0.2, time=3.774003875
    Sim 5 group-knockoff power = 0.4, FDR = 0.0, time=2.337522042
    Sim 6 group-knockoff power = 0.0, FDR = 0.0, time=3.913260458
    Sim 7 group-knockoff power = 0.222, FDR = 0.333, time=2.531130667
    Sim 8 group-knockoff power = 0.0, FDR = 0.0, time=3.927257125
    Sim 9 group-knockoff power = 0.0, FDR = 0.0, time=3.9803245
    Sim 10 group-knockoff power = 0.1, FDR = 0.0, time=2.300255291
    
    ME group knockoffs have average group power 0.1444
    ME group knockoffs have average group FDR 0.0533
    ME group knockoffs took average 3.438526575 seconds


## Conclusion

+ When variables are highly correlated so that one cannot find exact discoveries, group knockoffs may be useful as it identifies whether a group of variables are non-null without having to pinpoint the exact discovery.
+ Group knockoffs control the group FDR to be below the target FDR level. 
+ Groups do not have to be contiguous

