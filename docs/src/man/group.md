# Group Knockoffs

This tutorial generates group (model-X) knockoffs, which is useful when predictors are highly correlated. The methodology is described in the following paper

> Chu BB, Gu J, Chen Z, Morrison T, Candes E, He Z, Sabatti C. Second-order group knockoffs with applications to GWAS. arXiv preprint arXiv:2310.15069. 2023 Oct 23.

> Dai R, Barber R. The knockoff filter for FDR control in group-sparse and multitask regression. In International conference on machine learning 2016 Jun 11 (pp. 1851-1859). PMLR.

Currently available options for group knockoffs:
+ `:maxent`: Fully general maximum entropy (maxent) group knockoff, based on coordinate descent.
+ `:mvr`: Fully general minimum variance-based reconstructability (MVR) group knockoff, based on coordinate descent.
+ `:sdp`: Fully general SDP group knockoffs, based on coordinate descent. In general MVR/ME knockoffs tends to perform better than SDP in terms of power, and SDP generally converges slower. 
+ `:equi`: This implements the equi-correlated idea proposed in [Barber and Dai](https://proceedings.mlr.press/v48/daia16.html), which lets $S_j = \gamma \Sigma_{(G_j, G_j)}$ where $\Sigma_{(G_j, G_j)}$ is the block of $\Sigma$ containing variables in the $j$th group. Thus, instead of optimizing over all variables in $S$, we optimize a scalar $\gamma$. Conveniently, there a simple closed form solution for $\gamma$. For `mvr` and `maxent` group knockoffs, we initialize $S$ using this construction. 


```julia
# load packages for this tutorial
using Knockoffs
using LinearAlgebra
using Random
using StatsKit
using ToeplitzMatrices
using Distributions
```

## Gaussian model-X group knockoffs with known mean and covariance

To illustrate, lets simulate data $\mathbf{X}$ with covariance $\Sigma$ and mean $\mu$. Our model is
```math
\begin{aligned}
    X_{p \times 1} \sim N(\mathbf{0}_p, \Sigma)
\end{aligned}
```
where
```math
\begin{aligned}
\Sigma = 
\begin{pmatrix}
    1 & \rho & \rho^2 & ... & \rho^p\\
    \rho & 1 & & ... & \rho^{p-1}\\
    \vdots & & & 1 & \vdots \\
    \rho^p & \cdots & & & 1
\end{pmatrix}
\end{aligned}
```
Given $n$ iid samples from the above distribution, we will generate knockoffs according to 
```math
\begin{aligned}
(X, \tilde{X}) \sim N
\left(0, \ 
\begin{pmatrix}
    \Sigma & \Sigma - S\\
    \Sigma - S & \Sigma
\end{pmatrix}
\right)
\end{aligned}
```
where $S$ is a block-diagonal matrix satisfying $S \succeq 0$ and $2\Sigma - S \succeq 0$. 

Because variables are highly correlated with its neighbors ($\rho = 0.9$), it becomes difficult to distinguish which among a bunch of highly correlated variables are truly causal. Thus, group knockoffs test whether a *group* of variables have any signal should have better power than standard (single-variable) knockoffs. 

First, lets simulate some data


```julia
# simulate data
Random.seed!(2023)
n = 250 # sample size
p = 500 # number of features
k = 10  # number of causal variables
Σ = Matrix(SymmetricToeplitz(0.9.^(0:(p-1))))
# Σ = simulate_AR1(p, a=3, b=1)
# Σ = simulate_block_covariance(groups, 0.75, 0.25)
μ = zeros(p)
L = cholesky(Σ).L
X = randn(n, p) * L # design matrix
zscore!(X, mean(X, dims=1), std(X, dims=1)); # standardize columns of X
```

## Define group memberships

To generate group knockoffs, we need to vector specifying group membership. One can define this vector manually, or use the built-in functions [`hc_partition_groups`](https://biona001.github.io/Knockoffs.jl/dev/man/api/#Knockoffs.hc_partition_groups) which runs hierarchical clustering. 


```julia
groups = hc_partition_groups(X, cutoff = 0.5)
```




    500-element Vector{Int64}:
      1
      1
      1
      2
      2
      2
      2
      3
      3
      3
      3
      3
      4
      ⋮
     93
     93
     93
     93
     94
     94
     94
     95
     95
     96
     96
     96



## Generating group knockoffs

Generate group knockoffs with the exported function [`modelX_gaussian_group_knockoffs`](https://biona001.github.io/Knockoffs.jl/dev/man/api/#Knockoffs.modelX_gaussian_group_knockoffs). Similar to non-group knockoffs, group knockoff accepts keyword arguments `m`, `tol`, `method`, and `verbose` which controls the algorithm's behavior. 


```julia
@time me = modelX_gaussian_group_knockoffs(
    X, :maxent, groups, μ, Σ, 
    m = 5,              # number of knockoffs per variable to generate
    tol = 0.001,        # convergence tolerance
    inner_ccd_iter = 1, # optimize every entry of S exactly 1 time before moving on to PCA updates
    inner_pca_iter = 1, # optimize S with respect to pre-computed eigenvectors 1 time before going to CCA updates
    verbose=true);      # whether to print informative intermediate results
```

    Maxent initial obj = -12356.342528382938
    Iter 1 (PCA): obj = -8052.261406317259, δ = 0.08590802942739165, t1 = 0.02, t2 = 0.06
    Iter 2 (CCD): obj = -7773.418142140313, δ = 0.02528913358815642, t1 = 0.07, t2 = 0.22, t3 = 0.0
    Iter 3 (PCA): obj = -7500.564395607506, δ = 0.06150639965621555, t1 = 0.09, t2 = 0.28
    Iter 4 (CCD): obj = -7421.717268605351, δ = 0.013407344086563865, t1 = 0.13, t2 = 0.45, t3 = 0.01
    Iter 5 (PCA): obj = -7301.791785651631, δ = 0.05273855674419927, t1 = 0.14, t2 = 0.51
    Iter 6 (CCD): obj = -7261.496828481993, δ = 0.010742521958646422, t1 = 0.18, t2 = 0.68, t3 = 0.02
    Iter 7 (PCA): obj = -7189.736001319752, δ = 0.040084930695873995, t1 = 0.19, t2 = 0.74
    Iter 8 (CCD): obj = -7163.703260068862, δ = 0.009365146534480048, t1 = 0.23, t2 = 0.9, t3 = 0.02
    Iter 9 (PCA): obj = -7118.729388501863, δ = 0.03205741062856066, t1 = 0.25, t2 = 0.96
    Iter 10 (CCD): obj = -7099.35685440108, δ = 0.008319119029470615, t1 = 0.28, t2 = 1.13, t3 = 0.03
    Iter 11 (PCA): obj = -7069.4337783928995, δ = 0.03469906748947143, t1 = 0.3, t2 = 1.19
    Iter 12 (CCD): obj = -7053.809089037512, δ = 0.007363938980155588, t1 = 0.34, t2 = 1.35, t3 = 0.04
    Iter 13 (PCA): obj = -7032.680815573552, δ = 0.03513730957983818, t1 = 0.36, t2 = 1.41
    Iter 14 (CCD): obj = -7019.633183231447, δ = 0.006669152777008879, t1 = 0.4, t2 = 1.58, t3 = 0.04
    Iter 15 (PCA): obj = -7003.904253625472, δ = 0.03282563041856268, t1 = 0.42, t2 = 1.64
    Iter 16 (CCD): obj = -6992.830434732079, δ = 0.00608605978016194, t1 = 0.45, t2 = 1.8, t3 = 0.05
    Iter 17 (PCA): obj = -6980.615690052075, δ = 0.029470438897636054, t1 = 0.47, t2 = 1.86
    Iter 18 (CCD): obj = -6971.102297622322, δ = 0.005585492126193696, t1 = 0.51, t2 = 2.03, t3 = 0.05
    Iter 19 (PCA): obj = -6961.353115061036, δ = 0.025869490219870676, t1 = 0.52, t2 = 2.09
    Iter 20 (CCD): obj = -6953.078832706372, δ = 0.005130902988737138, t1 = 0.56, t2 = 2.26, t3 = 0.06
    Iter 21 (PCA): obj = -6945.114438554987, δ = 0.022715986357968256, t1 = 0.58, t2 = 2.32
    Iter 22 (CCD): obj = -6937.861762990844, δ = 0.004759993786595553, t1 = 0.62, t2 = 2.48, t3 = 0.06
    Iter 23 (PCA): obj = -6931.238155868997, δ = 0.019895600860486345, t1 = 0.64, t2 = 2.54
    Iter 24 (CCD): obj = -6924.81827242897, δ = 0.004424342317802936, t1 = 0.67, t2 = 2.71, t3 = 0.07
     14.845494 seconds (56.15 M allocations: 2.765 GiB, 4.55% gc time, 101.36% compilation time: 25% of which was recompilation)


+ Here CCD corresponds to optimization each entry ``S_{ij}`` independently, while PCA is a faster update that updates ``S_{new} = S + \delta vv'``. 
+ Users can modify the default behavior by supplying the arguments `inner_pca_iter` and `inner_ccd_iter`. For instance, we can turn off `inner_ccd_iter` to achieve much faster convergence at the sacrifice small accuracy. 
+ ``t_1, t_2, t_3`` are timers, which reveals that the computational bottleneck is in (2), which we dispatch to efficient LAPACK libraries, so the overall performance of our algorithm cannot really be improved. 
    1. ``t_1``: updating cholesky factors
    2. ``t_2``: solving forward-backward equations
    3. ``t_3``: solving off-diagonal 1D optimization problems using Brent's method

The output is a struct with the following fields
```julia
struct GaussianGroupKnockoff{T<:AbstractFloat, BD<:AbstractMatrix, S<:Symmetric} <: Knockoff
    X::Matrix{T} # n × p design matrix
    Xko::Matrix{T} # n × mp matrix storing knockoffs of X
    groups::Vector{Int} # p × 1 vector of group membership
    S::BD # p × p block-diagonal matrix of the same size as Sigma. S and (m+1)/m*Sigma - S are both psd
    gammas::Vector{T} # for suboptimal group construction only. These are scalars chosen so that S_i = γ_i * Sigma_i
    m::Int # number of knockoffs per feature generated
    Sigma::S # p × p symmetric covariance matrix. 
    method::Symbol # method for solving s
    obj::T # final objective value of group knockoff
end
```
Given this result, lets do a sanity check: is $(m+1)/m\Sigma - S$ positive semi-definite?


```julia
m = 5
eigmin((m+1)/m*me.Sigma - me.S)
```




    0.007198090465447198



## Second order group knockoffs

In practice, we often do not have the true covariance matrix $\Sigma$ and the true means $\mu$. In that case, we can generate second order group knockoffs via the 3 argument function


```julia
me_second_order = modelX_gaussian_group_knockoffs(X, :maxent, groups);
```

This will estimate the covariance matrix via a shrinkage estimator, see documentation API for more details. 

## Group knockoffs based on conditional independence assumption

One can choose a few representatives from each group and generate *representative* group knockoffs via [`modelX_gaussian_rep_group_knockoffs`](https://biona001.github.io/Knockoffs.jl/dev/man/api/#Knockoffs.modelX_gaussian_rep_group_knockoffs). Its advantages include:

+ Dramatically improved computational efficiency, since the group-knockoff optimization problem only needs to be carried out on the representative variables.
+ Improved power over standard group knockoffs, since the exchangeability have to be satisfied for less variables, so the resulting knockoffs are more "flexible"

This model assume that conditional on the group representatives, remaining variables are independent by groups. Although this assumption is not always met, we find that group-FDR is never really violated when `rep_threshold = 0.5` in our experiments with real or simulated data. 


```julia
@time rME = modelX_gaussian_rep_group_knockoffs(
    X, :maxent, groups, μ, Σ, 
    m = 5,               # number of knockoffs per variable to generate
    tol = 0.0001,        # convergence tolerance
    rep_threshold = 0.5, # R2 threshold for choosing representatives
    verbose=true);       # whether to print informative intermediate results
```

    96 representatives for 500 variables, 96 optimization variables
    Iter 1: δ = 0.17097649194255435
    Iter 2: δ = 0.21816588319812047
    Iter 3: δ = 0.015193710385058123
    Iter 4: δ = 0.005264630389642366
    Iter 5: δ = 0.0007711357354073245
    Iter 6: δ = 0.00015192767130323137
    Iter 7: δ = 2.084227499749014e-5
      3.425291 seconds (26.90 M allocations: 1.352 GiB, 14.15% gc time, 178.83% compilation time: 50% of which was recompilation)


Observe the 96/500 variables were selected as representatives, resulting in $>10$ times speedup compared to standard group knockoffs. 

Also, the resulting knockoffs is still $n \times mp$, so we do sample knockoffs for each variable even though the optimization was only carried out on a subset.


```julia
rME.Xko
```




    250×2500 Matrix{Float64}:
     -1.36277   -1.68917    -0.916809   …   2.82242    2.72732     1.76236
     -0.352257   0.210195    0.170385      -0.681593  -0.12002    -0.803533
     -2.5285    -2.17058    -2.44769       -0.703286  -0.639158    0.0138371
      1.44446    1.14063     1.24941        0.575184   1.39981     1.4122
     -0.248072   0.0936132   0.234051      -0.196387  -0.0474507   0.460242
     -0.470258  -0.917885   -0.93198    …   0.40301    0.509177    0.623285
      0.731273   0.522623   -0.020438      -0.646962  -1.78139    -2.78991
     -1.85267   -1.94916    -1.60076        0.188866  -0.0538957  -0.757954
     -3.14544   -3.2571     -3.64745        1.03654    0.922437    0.84392
      1.07965    0.899967    0.774725       0.364908   0.892139   -0.0277094
      0.928806   1.04353     0.503417   …   0.131817  -0.448856   -0.138799
      1.08034    0.947275    0.851498      -0.295635  -0.629569   -1.32304
     -0.247422  -0.042002   -0.482177      -1.21204   -0.0821259   0.0820233
      ⋮                                 ⋱                         
     -1.04164   -1.64549    -1.31717        1.24481    1.67012     1.42037
      0.497597   0.474561    0.577012       1.17042    0.730953    0.130879
     -0.705371  -1.09896    -1.39721    …   0.539932   0.138708    0.801311
     -0.372699  -0.314788    0.0229731     -1.73432   -1.12943    -1.50057
     -1.07809    0.245403    1.09644       -1.12278   -1.03152    -0.782376
     -0.679771  -0.311122   -0.392692       1.72593    2.52517     1.59474
     -0.237038  -0.517252   -1.05072       -0.602533  -0.36173     0.0850561
     -0.404256   0.238769    0.966127   …  -1.28687   -0.956438   -0.567431
     -0.326905  -0.0642425   0.691291       0.439435   0.680286    0.772022
     -0.855096  -0.42782    -0.24862       -0.892996  -1.48545    -1.45224
     -0.735913  -0.325394    0.262669       0.340387   0.63169     0.246588
     -0.923137  -1.24118    -0.583673       1.18556    1.1271      0.252369



## Lasso Example

Lets see the empirical power and FDR group knockoffs over 10 simulations when
+ the target FDR is 10% (i.e. `fdr_target[3] = 0.1`)
+ we generate $m=5$ knockoffs per feature
+ ``\beta_j \sim \pm 0.25`` for 10 causal ``j``s

Note power and FDR is defined at the group level


```julia
group_powers, group_fdrs, group_times, group_s = Float64[], Float64[], Float64[], Float64[]

for sim in 1:10
    # simulate X
    n = 1000 # sample size
    p = 200  # number of covariates
    k = 10   # number of true predictors
    Σ = Matrix(SymmetricToeplitz(0.9.^(0:(p-1)))) # true covariance matrix
    μ = zeros(p)
    L = cholesky(Σ).L
    X = randn(n, p) * L
    zscore!(X, mean(X, dims=1), std(X, dims=1)); # standardize columns of X

    # define groups
    groups = hc_partition_groups(X, cutoff=0.5)
    
    # simulate y
    βtrue = zeros(p)
    βtrue[1:k] .= rand(-1:2:1, k) .* 0.25
    shuffle!(βtrue)
    correct_groups = groups[findall(!iszero, βtrue)] |> unique
    ϵ = randn(n)
    y = X * βtrue + ϵ;

    # group ME knockoffs
    t = @elapsed ko_filter = fit_lasso(y, X, μ, Symmetric(Σ), method=:maxent, groups=groups, m=5)
    selected_groups, _ = select_groups(ko_filter, 0.1) # select groups at 10% FDR
    power = length(intersect(correct_groups, selected_groups)) / length(correct_groups)
    fdr = length(setdiff(selected_groups, correct_groups)) / max(1, length(selected_groups))
    println("Sim $sim group-knockoff power = $power, FDR = $fdr, time=$t")
    push!(group_powers, power); push!(group_fdrs, fdr); push!(group_times, t)
    GC.gc();GC.gc();GC.gc();
end

println("\nME group knockoffs have average group power $(mean(group_powers))")
println("ME group knockoffs have average group FDR $(mean(group_fdrs))")
println("ME group knockoffs took average $(mean(group_times)) seconds");
```

    Sim 1 group-knockoff power = 0.8888888888888888, FDR = 0.1111111111111111, time=7.961366334
    Sim 2 group-knockoff power = 1.0, FDR = 0.0, time=7.594480167
    Sim 3 group-knockoff power = 0.5555555555555556, FDR = 0.0, time=7.619353625
    Sim 4 group-knockoff power = 0.75, FDR = 0.0, time=10.109215166
    Sim 5 group-knockoff power = 1.0, FDR = 0.09090909090909091, time=9.392233958
    Sim 6 group-knockoff power = 1.0, FDR = 0.18181818181818182, time=6.870906584
    Sim 7 group-knockoff power = 0.8888888888888888, FDR = 0.1111111111111111, time=9.787938917
    Sim 8 group-knockoff power = 0.875, FDR = 0.0, time=6.637393125
    Sim 9 group-knockoff power = 1.0, FDR = 0.0, time=9.742192583
    Sim 10 group-knockoff power = 0.7777777777777778, FDR = 0.0, time=13.692710584
    
    ME group knockoffs have average group power 0.8736111111111112
    ME group knockoffs have average group FDR 0.049494949494949494
    ME group knockoffs took average 8.940779104299999 seconds


For comparison, lets try the same simulation but we generate regular (non-grouped) knockoffs


```julia
regular_powers, regular_fdrs, regular_times = Float64[], Float64[], Float64[]

Random.seed!(2026)
for sim in 1:10
    # simulate X
    n = 1000 # sample size
    p = 200  # number of covariates
    k = 10   # number of true predictors
    Σ = Matrix(SymmetricToeplitz(0.9.^(0:(p-1)))) # true covariance matrix
    μ = zeros(p)
    L = cholesky(Σ).L
    X = randn(n, p) * L
    zscore!(X, mean(X, dims=1), std(X, dims=1)); # standardize columns of X
    
    # simulate y
    βtrue = zeros(p)
    βtrue[1:k] .= rand(-1:2:1, k) .* 0.25
    shuffle!(βtrue)
    correct_snps = findall(!iszero, βtrue)
    ϵ = randn(n)
    y = X * βtrue + ϵ;

    # group ME knockoffs
    t = @elapsed ko_filter = fit_lasso(y, X, μ, Symmetric(Σ), method=:maxent, m=5)
    selected = select_variables(ko_filter, 0.1) # select variables at 10% FDR
    power = length(intersect(correct_snps, selected)) / length(correct_snps)
    fdr = length(setdiff(selected, correct_snps)) / max(1, length(selected))
    println("Sim $sim nongroup-knockoff power = $power, FDR = $fdr, time=$t")
    push!(regular_powers, power); push!(regular_fdrs, fdr); push!(regular_times, t)
    GC.gc();GC.gc();GC.gc();
end

println("\nME (standard) knockoffs have average group power $(mean(regular_powers))")
println("ME (standard) knockoffs have average group FDR $(mean(regular_fdrs))")
println("ME (standard) knockoffs took average $(mean(regular_times)) seconds");
```

    Sim 1 nongroup-knockoff power = 0.4, FDR = 0.0, time=9.094616916
    Sim 2 nongroup-knockoff power = 0.7, FDR = 0.3, time=6.695011417
    Sim 3 nongroup-knockoff power = 0.4, FDR = 0.0, time=6.605352125
    Sim 4 nongroup-knockoff power = 0.0, FDR = 0.0, time=4.543773167
    Sim 5 nongroup-knockoff power = 0.7, FDR = 0.125, time=8.372022291
    Sim 6 nongroup-knockoff power = 0.0, FDR = 0.0, time=13.595817792
    Sim 7 nongroup-knockoff power = 0.3, FDR = 0.0, time=8.254227083
    Sim 8 nongroup-knockoff power = 0.6, FDR = 0.14285714285714285, time=7.075842125
    Sim 9 nongroup-knockoff power = 0.8, FDR = 0.0, time=8.694660792
    Sim 10 nongroup-knockoff power = 0.4, FDR = 0.0, time=5.203888084
    
    ME (standard) knockoffs have average group power 0.43000000000000005
    ME (standard) knockoffs have average group FDR 0.05678571428571429
    ME (standard) knockoffs took average 7.813521179199998 seconds


## Conclusion

+ When variables are highly correlated so that one cannot find exact discoveries, group knockoffs may be useful for improving power as it identifies whether a group of variables are non-null without having to pinpoint the exact discovery. It trades resolution to discover more causal signals.
+ Group knockoffs control the group FDR to be below the target FDR level. 
+ Groups do not have to be contiguous
+ With modest group sizes, group knockoff's compute time is roughly equivalent to standard (non-grouped) knockoffs
+ When $p$ is too large or group sizes are too large, one can employ representative group knockoff strategy. Empirically it has better power and much faster compute times. 
