# Group Knockoffs

This tutorial generates group (model-X) knockoffs, which is useful when predictors are highly correlated. The methodology is described in the following paper

> Dai R, Barber R. The knockoff filter for FDR control in group-sparse and multitask regression. InInternational conference on machine learning 2016 Jun 11 (pp. 1851-1859). PMLR.


!!! note

    In the original paper, Dai and Barber only describes how to construct equi-correlated group knockoffs, but the same idea can be generalized to SDP group knockoffs, which we also implement here. 


```julia
# load packages for this tutorial
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



# Equi-correlated group knockoffs

+ Given $p \times p$ positive definite matrix $\Sigma$, partition the $p$ features into $m$ groups $G_1,...,G_m$. We want to optimize the following problem
```math
\begin{aligned}
    \min_{S} & \ Tr(|\Sigma - S|)\\
    \text{such that } & S \succeq 0 \text{ and } 2\Sigma - S \succeq 0.
\end{aligned}
```
+ Here $S$ is a group-block-diagonal matrix of the form $S = diag(S_1,...,S_m)$ where each $S_j$ is a positive definite matrix that has dimension $|G_j| \times |G_j|$
+ The equi-correlated idea proposed in [Barber and Dai](https://proceedings.mlr.press/v48/daia16.html) is to let $S_j = \gamma \Sigma_{(G_j, G_j)}$ where $\Sigma_{(G_j, G_j)}$ is the block of $\Sigma$ containing variables in the $j$th group. Thus, instead of optimizing over all variables in $S$, we optimize a scalar $\gamma$. Conveniently, there a simple closed form solution.

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
p = 100  # number of covariates
k = 10   # number of true predictors
Σ = Matrix(SymmetricToeplitz(0.9.^(0:(p-1)))) # true covariance matrix
groupsizes = [5 for i in 1:20] # each group has 5 variables
groups = vcat([i*ones(g) for (i, g) in enumerate(groupsizes)]...) |> Vector{Int}
true_mu = zeros(p)
L = cholesky(Σ).L
X = randn(n, p) * L
zscore!(X, mean(X, dims=1), std(X, dims=1)); # standardize columns of X
```

Generate group knockoffs as such:


```julia
ko_equi = modelX_gaussian_group_knockoffs(X, groups, :equi, Σ, true_mu);
```

Lets do a sanity check: is $2\Sigma - S$ positive semi-definite?


```julia
# compute minimum eigenvalues of 2Σ - S
eigmin(2ko_equi.Σ - ko_equi.S)
```




    6.152687425537049e-16



The min eigenvalue is $\approx 0$ up to numerical precision, so the knockoff structure indeed satisfies the PSD constraint. 

## SDP group knockoffs


+ This extends the equi-correlated construction of [Barber and Dai](https://proceedings.mlr.press/v48/daia16.html)
+ The idea is to choose $S_j = \gamma_j \Sigma_{(G_j, G_j)}$. Note that the difference with the equi-correlated construction is that $\gamma$ is potentially allowed to vary in each group. If $\Sigma$ has unit variance, we optimize the following problem

```math
\begin{aligned}
    \min_{\gamma_1,...,\gamma_m} & Tr(|\Sigma - S|)\\
    \text{such that } & 0 \le \gamma_j \le 1 \text{ for all } j \text{ and }\\
    & 2\Sigma - 
    \begin{pmatrix}
        \gamma_1\Sigma_{(G_1, G_1)} & & 0\\
        & \ddots & \\
        0 & & \gamma_m \Sigma_{(G_m, G_m)}
    \end{pmatrix} \succeq 0
\end{aligned}
```

Now lets generate SDP group knockoffs


```julia
@time ko_sdp = modelX_gaussian_group_knockoffs(X, groups, :sdp, Σ, true_mu);
```

      0.379894 seconds (96.31 k allocations: 18.528 MiB)


We can also do a sanity check to see if the SDP knockoffs satisfy the PSD constraint


```julia
# compute minimum eigenvalues of 2Σ - S
eigmin(2ko_sdp.Σ - ko_sdp.S)
```




    -6.413307528979749e-8



## Second order group knockoffs

In practice, we often do not have the true covariance matrix $\Sigma$ and the true means $\mu$. In that case, we can generate second order group knockoffs via the 3 argument function


```julia
ko_equi = modelX_gaussian_group_knockoffs(X, groups, :equi);
```

This will estimate the covariance matrix, see documentation API for more details. 

## Power and FDR comparison

Lets compare empirical power and FDR for equi and SDP group knockoffs when the targer FDR is 10%.


```julia
target_fdr = 0.1
equi_powers, equi_fdrs, equi_times = Float64[], Float64[], Float64[]
sdp_powers, sdp_fdrs, sdp_times = Float64[], Float64[], Float64[]

Random.seed!(2022)
for sim in 1:10
    # simulate y
    βtrue = zeros(p)
    βtrue[1:k] .= rand(-1:2:1, k) .* 0.1
    shuffle!(βtrue)
    correct_groups = get_signif_groups(βtrue, groups)
    ϵ = randn(n)
    y = X * βtrue + ϵ;

    # equi-group knockoffs
    t = @elapsed ko_filter = fit_lasso(y, X, method=:equi, groups=groups)
    idx = findfirst(x -> x == target_fdr, ko_filter.fdr_target)
    power = round(TP(correct_groups, ko_filter.βs[idx], groups), digits=3)
    fdr = round(FDR(correct_groups, ko_filter.βs[idx], groups), digits=3)
    println("Simulation $sim equi-group knockoffs power = $power, FDR = $fdr, time=$t")
    push!(equi_powers, power)
    push!(equi_fdrs, fdr)
    push!(equi_times, t)
    
    # SDP-group knockoffs
    t = @elapsed ko_filter = fit_lasso(y, X, method=:sdp, groups=groups)
    power = round(TP(correct_groups, ko_filter.βs[idx], groups), digits=3)
    fdr = round(FDR(correct_groups, ko_filter.βs[idx], groups), digits=3)
    println("Simulation $sim SDP-group knockoffs power = $power, FDR = $fdr, time=$t")
    push!(sdp_powers, power)
    push!(sdp_fdrs, fdr)
    push!(sdp_times, t)
end

println("\nEqui-correlated group knockoffs have average group power $(mean(equi_powers))")
println("Equi-correlated group knockoffs have average group FDR $(mean(equi_fdrs))");
println("Equi-correlated group knockoffs took average $(mean(equi_times)) seconds");

println("\nSDP group knockoffs have average group power $(mean(sdp_powers))")
println("SDP group knockoffs have average group FDR $(mean(sdp_fdrs))");
println("SDP group knockoffs took average $(mean(sdp_times)) seconds");
```

    Simulation 1 equi-group knockoffs power = 0.125, FDR = 0.0, time=2.053749326
    Simulation 1 SDP-group knockoffs power = 0.0, FDR = 0.0, time=6.45688248
    Simulation 2 equi-group knockoffs power = 0.0, FDR = 0.0, time=1.777067223
    Simulation 2 SDP-group knockoffs power = 0.125, FDR = 0.5, time=2.843801253
    Simulation 3 equi-group knockoffs power = 0.333, FDR = 0.0, time=3.566960454
    Simulation 3 SDP-group knockoffs power = 0.0, FDR = 0.0, time=2.373457417
    Simulation 4 equi-group knockoffs power = 0.333, FDR = 0.0, time=1.670845043
    Simulation 4 SDP-group knockoffs power = 0.333, FDR = 0.0, time=2.79045689
    Simulation 5 equi-group knockoffs power = 0.25, FDR = 0.0, time=1.317231961
    Simulation 5 SDP-group knockoffs power = 0.375, FDR = 0.0, time=6.14065572
    Simulation 6 equi-group knockoffs power = 0.333, FDR = 0.0, time=1.737344252
    Simulation 6 SDP-group knockoffs power = 0.667, FDR = 0.333, time=3.045115028
    Simulation 7 equi-group knockoffs power = 0.125, FDR = 0.0, time=1.873825217
    Simulation 7 SDP-group knockoffs power = 0.5, FDR = 0.0, time=4.890016678
    Simulation 8 equi-group knockoffs power = 0.111, FDR = 0.0, time=1.986784703
    Simulation 8 SDP-group knockoffs power = 0.667, FDR = 0.143, time=2.524734588
    Simulation 9 equi-group knockoffs power = 0.556, FDR = 0.0, time=1.600709261
    Simulation 9 SDP-group knockoffs power = 0.444, FDR = 0.0, time=2.742993123
    Simulation 10 equi-group knockoffs power = 0.5, FDR = 0.429, time=1.599380885
    Simulation 10 SDP-group knockoffs power = 0.25, FDR = 0.0, time=2.205388288
    
    Equi-correlated group knockoffs have average group power 0.2666
    Equi-correlated group knockoffs have average group FDR 0.0429
    Equi-correlated group knockoffs took average 1.9183898324999997 seconds
    
    SDP group knockoffs have average group power 0.33609999999999995
    SDP group knockoffs have average group FDR 0.09759999999999999
    SDP group knockoffs took average 3.6013501465000006 seconds


## Conclusion

+ Both equicorrelated and SDP group knockoffs control the group FDR to be below the target FDR level. 
+ SDP group knockoffs have slightly better power than equi-correlated group knockoffs
+ Equi-correlated knockoffs are ~2x faster to construct than group-SDP (for $p=100$ covariates and 20 groups). On a separate test with 200 groups and 5 features per group ($p = 1000$), SDP construction were ~45x slower. 

