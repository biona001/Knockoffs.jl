
# Group Knockoffs

This tutorial generates group (model-X) knockoffs, which is useful when predictors are highly correlated. The methodology is described in the following paper

> Chu BB, Gu J, Chen Z, Morrison T, Candes E, He Z, Sabatti C. Second-order group knockoffs with applications to GWAS. arXiv preprint arXiv:2310.15069. 2023 Oct 23.

> Dai R, Barber R. The knockoff filter for FDR control in group-sparse and multitask regression. In International conference on machine learning 2016 Jun 11 (pp. 1851-1859). PMLR.

Currently available options for group knockoffs:
+ `:maxent`: Fully general maximum entropy (maxent) group knockoff, based on coordinate descent.
+ `:mvr`: Fully general minimum variance-based reconstructability (MVR) group knockoff, based on coordinate descent.
+ `:sdp`: Fully general SDP group knockoffs, based on coordinate descent. In general MVR/ME knockoffs tends to perform better than SDP in terms of power, and SDP generally converges slower. 
+ `:equi`: This implements the equi-correlated idea proposed in [Barber and Dai](https://proceedings.mlr.press/v48/daia16.html), which lets $S_j = \gamma \Sigma_{(G_j, G_j)}$ where $\Sigma_{(G_j, G_j)}$ is the block of $\Sigma$ containing variables in the $j$th group. Thus, instead of optimizing over all variables in $S$, we optimize a scalar $\gamma$. Conveniently, there a simple closed form solution for $\gamma$. For `mvr` and `maxent` group knockoffs, we initialize $S$ using this construction. 
+ `:sdp_subopt`: This generalizes the equi-correlated group knockoff idea by having $S_j = \gamma_j \Sigma_{(G_j, G_j)}$. Instead of optimizing over all variables in $S$, we optimize over a vector $\gamma_1,...,\gamma_G$. Note this functionality is mainly provided for testing purposes. 


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

To generate group knockoffs, we need to vector specifying group membership. One can define this vector manually, or use the built-in functions [`hc_partition_groups`](https://biona001.github.io/Knockoffs.jl/dev/man/api/#Knockoffs.hc_partition_groups) or [`id_partition_groups`](https://biona001.github.io/Knockoffs.jl/dev/man/api/#Knockoffs.id_partition_groups). 


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
    Iter 1 (PCA): obj = -8052.261406317261, δ = 0.08590802942739098, t1 = 0.11, t2 = 0.06
    Iter 2 (CCD): obj = -7794.572748302039, δ = 0.021168745196021146, t1 = 0.17, t2 = 0.18, t3 = 0.0
    Iter 3 (PCA): obj = -7511.581367389803, δ = 0.051611315519312195, t1 = 0.28, t2 = 0.24
    Iter 4 (CCD): obj = -7461.192156067141, δ = 0.012741504093028691, t1 = 0.32, t2 = 0.37, t3 = 0.0
    Iter 5 (PCA): obj = -7338.128509045188, δ = 0.047892393422037396, t1 = 0.39, t2 = 0.43
    Iter 6 (CCD): obj = -7308.932874229332, δ = 0.01053604459600144, t1 = 0.43, t2 = 0.55, t3 = 0.0
    Iter 7 (PCA): obj = -7229.540018465837, δ = 0.036888999559287136, t1 = 0.51, t2 = 0.61
    Iter 8 (CCD): obj = -7208.67612681877, δ = 0.009320045724745799, t1 = 0.55, t2 = 0.74, t3 = 0.0
    Iter 9 (PCA): obj = -7154.410518552644, δ = 0.030821712350118435, t1 = 0.62, t2 = 0.79
    Iter 10 (CCD): obj = -7137.772110915237, δ = 0.00853792672155702, t1 = 0.66, t2 = 0.93, t3 = 0.0
    Iter 11 (PCA): obj = -7099.32785445378, δ = 0.03136109572323209, t1 = 0.77, t2 = 0.98
    Iter 12 (CCD): obj = -7085.247398507887, δ = 0.007935922476527225, t1 = 0.81, t2 = 1.11, t3 = 0.0
    Iter 13 (PCA): obj = -7057.1404838054295, δ = 0.03291205436687801, t1 = 0.92, t2 = 1.17
    Iter 14 (CCD): obj = -7044.876172687204, δ = 0.007314871106654346, t1 = 0.96, t2 = 1.3, t3 = 0.01
    Iter 15 (PCA): obj = -7023.629304760585, δ = 0.03250775080424084, t1 = 1.06, t2 = 1.36
    Iter 16 (CCD): obj = -7012.850120947562, δ = 0.006854890650307599, t1 = 1.11, t2 = 1.49, t3 = 0.01
    Iter 17 (PCA): obj = -6996.271858853715, δ = 0.030738229078674354, t1 = 1.21, t2 = 1.55
    Iter 18 (CCD): obj = -6986.792631541152, δ = 0.006430704916053182, t1 = 1.25, t2 = 1.68, t3 = 0.01
    Iter 19 (PCA): obj = -6973.531686161982, δ = 0.028115057674247407, t1 = 1.37, t2 = 1.74
    Iter 20 (CCD): obj = -6965.156942284159, δ = 0.006029270847763139, t1 = 1.41, t2 = 1.87, t3 = 0.01
    Iter 21 (PCA): obj = -6954.364877631323, δ = 0.02578275449508116, t1 = 1.48, t2 = 1.93
    Iter 22 (CCD): obj = -6946.941292530288, δ = 0.005644470245583583, t1 = 1.52, t2 = 2.06, t3 = 0.01
    Iter 23 (PCA): obj = -6938.0502511322275, δ = 0.023339824851938414, t1 = 1.58, t2 = 2.11
    Iter 24 (CCD): obj = -6931.429620290818, δ = 0.005272863979625453, t1 = 1.62, t2 = 2.24, t3 = 0.01
    Iter 25 (PCA): obj = -6924.0383562435045, δ = 0.02083871047261553, t1 = 1.67, t2 = 2.29
    Iter 26 (CCD): obj = -6918.112876107218, δ = 0.004918165386397036, t1 = 1.71, t2 = 2.42, t3 = 0.01
    Iter 27 (PCA): obj = -6911.905927691649, δ = 0.018590008447367395, t1 = 1.78, t2 = 2.48
    Iter 28 (CCD): obj = -6906.577950472131, δ = 0.00457709162696512, t1 = 1.83, t2 = 2.61, t3 = 0.01
      4.830540 seconds (42.09 k allocations: 236.493 MiB, 0.18% gc time)


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




    0.007406506932664684



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
    Iter 1: δ = 0.1495328321789994
    Iter 2: δ = 0.19916106242709308
    Iter 3: δ = 0.019321694755150942
    Iter 4: δ = 0.005068052646868659
    Iter 5: δ = 0.0009112831450813208
    Iter 6: δ = 0.00012302743183395526
    Iter 7: δ = 2.467589395582781e-5
      0.341659 seconds (15.35 k allocations: 214.558 MiB, 5.78% gc time)


Observe the 96/500 variables were selected as representatives, resulting in $>10$ times speedup compared to standard group knockoffs. 

Also, the resulting knockoffs is still $n \times mp$, so we do sample knockoffs for each variable even though the optimization was only carried out on a subset.


```julia
rME.Xko
```




    250×2500 Matrix{Float64}:
     -1.65214    -1.2264     -0.90357    …   2.37346      2.19603     1.85045
      1.04509     0.742908    0.276858      -0.19256     -0.127092   -0.590027
      0.202973   -1.17446    -0.66758       -0.493992    -0.240993    0.115518
      0.718775    0.878739    0.538289       1.44572      1.66234     1.60899
     -0.432394    0.0979436   1.42076       -0.401493    -0.235617   -0.273724
     -1.06394    -1.1612     -0.668314   …   1.68086      0.518035    0.473119
      1.51455     1.30363     0.229759       0.00686854   0.035861   -0.201092
      0.0646647  -1.05129    -0.120897      -0.621302    -0.287705    0.248626
     -1.91251    -1.13833    -2.38578        1.17047      0.782406    0.256628
      0.202989   -0.707357   -0.793931      -0.538658    -0.940356   -0.54821
     -0.143222   -0.617772   -0.667247   …  -0.582162     0.0561363  -0.0149074
      1.99224     1.03271     0.248894      -0.101577    -0.766011   -1.01534
     -0.586636   -0.585003    0.0514114     -1.2039      -0.702172   -0.708949
      ⋮                                  ⋱                           
     -0.643646   -0.0710437  -0.380527      -0.395322    -0.667184   -0.0555226
     -0.36097    -0.174519    1.15659        0.810179     0.23772     0.279265
      1.00372     0.525153    0.509426   …   0.893199     0.391243    0.0832597
      0.237773   -0.0549151  -1.04705       -1.05313     -1.19608    -0.863732
      0.419572    0.611245    1.18161       -0.831543    -0.680479   -1.24888
     -1.82841    -1.42881    -1.34797        2.41993      2.69197     2.47547
     -1.0607     -0.798633   -0.798233      -0.578312    -0.510215   -0.0882358
     -0.622852   -0.0162713   0.743521   …  -0.132991    -0.514501   -0.962194
     -0.165246    1.11386     0.689408      -0.14881     -0.6484     -0.456806
     -0.500523   -0.421848   -0.481629       1.01061      0.982484    0.529933
      2.10645     2.79504     2.12414        0.824754     0.530422    0.777875
     -0.391608   -0.45099    -0.856056       0.461328    -0.396119   -0.599234



## Lasso Example

Lets see the empirical power and FDR group knockoffs over 10 simulations when
+ the targer FDR is 10%
+ we generate $m=5$ knockoffs per feature
+ ``\beta_j \sim \pm 0.25`` for 10 causal ``j``s

Note power and FDR is defined at the group level


```julia
group_powers, group_fdrs, group_times, group_s = Float64[], Float64[], Float64[], Float64[]

Random.seed!(2022)
for sim in 1:10
    # simulate X
    Random.seed!(sim)
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
    t = @elapsed ko_filter = fit_lasso(y, X, method=:maxent, groups=groups, m=5)
    selected = ko_filter.selected[3]
    power = length(intersect(correct_groups, selected)) / length(correct_groups)
    fdr = length(setdiff(selected, correct_groups)) / max(1, length(selected))
    println("Sim $sim group-knockoff power = $power, FDR = $fdr, time=$t")
    push!(group_powers, power); push!(group_fdrs, fdr); push!(group_times, t)
    GC.gc();GC.gc();GC.gc();
end

println("\nME group knockoffs have average group power $(mean(group_powers))")
println("ME group knockoffs have average group FDR $(mean(group_fdrs))")
println("ME group knockoffs took average $(mean(group_times)) seconds");
```

    Sim 1 group-knockoff power = 1.0, FDR = 0.1, time=9.395477167
    Sim 2 group-knockoff power = 0.7777777777777778, FDR = 0.0, time=8.08905475
    Sim 3 group-knockoff power = 0.8888888888888888, FDR = 0.1111111111111111, time=6.093907333
    Sim 4 group-knockoff power = 0.8, FDR = 0.0, time=8.676211084
    Sim 5 group-knockoff power = 0.7, FDR = 0.0, time=10.33491675
    Sim 6 group-knockoff power = 0.5, FDR = 0.0, time=10.055918625
    Sim 7 group-knockoff power = 1.0, FDR = 0.0, time=6.909068458
    Sim 8 group-knockoff power = 0.4444444444444444, FDR = 0.0, time=9.819233042
    Sim 9 group-knockoff power = 0.7, FDR = 0.0, time=11.155753209
    Sim 10 group-knockoff power = 0.5555555555555556, FDR = 0.0, time=7.340749875
    
    ME group knockoffs have average group power 0.7366666666666667
    ME group knockoffs have average group FDR 0.021111111111111112
    ME group knockoffs took average 8.7870290293 seconds


For comparison, lets try the same simulation but we generate regular (non-grouped) knockoffs


```julia
regular_powers, regular_fdrs, regular_times = Float64[], Float64[], Float64[]

Random.seed!(2022)
for sim in 1:10
    # simulate X
    Random.seed!(sim)
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
    t = @elapsed ko_filter = fit_lasso(y, X, method=:maxent, m=5)
    selected = ko_filter.selected[3]
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

    Sim 1 nongroup-knockoff power = 0.7, FDR = 0.2222222222222222, time=7.151643042
    Sim 2 nongroup-knockoff power = 0.7, FDR = 0.0, time=7.163531958
    Sim 3 nongroup-knockoff power = 0.2, FDR = 0.0, time=5.438854459
    Sim 4 nongroup-knockoff power = 0.0, FDR = 0.0, time=7.861218583
    Sim 5 nongroup-knockoff power = 0.2, FDR = 0.0, time=9.57650625
    Sim 6 nongroup-knockoff power = 0.0, FDR = 0.0, time=8.987028709
    Sim 7 nongroup-knockoff power = 0.0, FDR = 0.0, time=5.27945125
    Sim 8 nongroup-knockoff power = 0.0, FDR = 0.0, time=9.898184792
    Sim 9 nongroup-knockoff power = 0.4, FDR = 0.0, time=10.721144208
    Sim 10 nongroup-knockoff power = 0.5, FDR = 0.0, time=6.266258084
    
    ME (standard) knockoffs have average group power 0.26999999999999996
    ME (standard) knockoffs have average group FDR 0.02222222222222222
    ME (standard) knockoffs took average 7.8343821335 seconds


## Conclusion

+ When variables are highly correlated so that one cannot find exact discoveries, group knockoffs may be useful for improving power as it identifies whether a group of variables are non-null without having to pinpoint the exact discovery. It trades resolution to discover more causal signals.
+ Group knockoffs control the group FDR to be below the target FDR level. 
+ Groups do not have to be contiguous
+ With modest group sizes, group knockoff's compute time is roughly equivalent to standard (non-grouped) knockoffs
+ When $p$ is too large or group sizes are too large, one can employ representative group knockoff strategy. Empirically it has better power and much faster compute times. 
