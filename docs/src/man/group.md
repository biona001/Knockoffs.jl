
# Group Knockoffs

This tutorial generates group (model-X) knockoffs, which is useful when predictors are highly correlated. The methodology is described in the following paper

> Dai R, Barber R. The knockoff filter for FDR control in group-sparse and multitask regression. InInternational conference on machine learning 2016 Jun 11 (pp. 1851-1859). PMLR.


!!! note

    In the original paper, Dai and Barber only describes how to construct a suboptimal equi-correlated group knockoffs. Here we implement fully generalized alternatives.
    
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

    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mPrecompiling StatsKit [2cb19f9e-ec4d-5c53-8573-a4542a68d3f0]
    [32mMinimizing 2 	 Time: 0:00:00 (88.81 ms/it)[39m[K
    [32mMinimizing 57 	 Time: 0:00:00 ( 4.06 ms/it)[39m[K
    [32mMinimizing 34 	 Time: 0:00:00 ( 2.97 ms/it)[39m[K
    [32mMinimizing 119 	 Time: 0:00:00 ( 1.72 ms/it)[39m[K
    [32mMinimizing 203 	 Time: 0:00:00 ( 1.51 ms/it)[39m[K
    [32mMinimizing 222 	 Time: 0:00:00 ( 1.48 ms/it)[39m[K
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mPrecompiling ToeplitzMatrices [c751599d-da0a-543b-9d20-d0a503d91d24]


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
    \Sigma & \Sigma - diag(s)\\
    \Sigma - diag(s) & \Sigma
\end{pmatrix}
\right)
\end{aligned}
```

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

    Maxent initial obj = -12356.34252838294
    Iter 1 (PCA): obj = -8052.261406317257, δ = 0.08590802942739141, t1 = 0.02, t2 = 0.06
    Iter 2 (CCD): obj = -7794.572748302033, δ = 0.021168745196021455, t1 = 0.06, t2 = 0.22, t3 = 0.0
    Iter 3 (PCA): obj = -7511.5813673898, δ = 0.05161131551931233, t1 = 0.08, t2 = 0.28
    Iter 4 (CCD): obj = -7461.192156067138, δ = 0.012741504093028197, t1 = 0.11, t2 = 0.44, t3 = 0.0
    Iter 5 (PCA): obj = -7338.12850904518, δ = 0.04789239342203643, t1 = 0.13, t2 = 0.5
    Iter 6 (CCD): obj = -7308.932874229322, δ = 0.010536044596001392, t1 = 0.17, t2 = 0.73, t3 = 0.0
    Iter 7 (PCA): obj = -7229.540018465826, δ = 0.036888999559287, t1 = 0.32, t2 = 0.81
    Iter 8 (CCD): obj = -7208.676126818758, δ = 0.0093200457247459, t1 = 0.36, t2 = 0.99, t3 = 0.0
    Iter 9 (PCA): obj = -7154.410518552632, δ = 0.03082171235011954, t1 = 0.38, t2 = 1.05
    Iter 10 (CCD): obj = -7137.772110915227, δ = 0.008537926721557176, t1 = 0.41, t2 = 1.21, t3 = 0.01
    Iter 11 (PCA): obj = -7099.327854453767, δ = 0.031361095723232424, t1 = 0.43, t2 = 1.27
    Iter 12 (CCD): obj = -7085.247398507873, δ = 0.007935922476527326, t1 = 0.46, t2 = 1.43, t3 = 0.01
    Iter 13 (PCA): obj = -7057.140483805416, δ = 0.0329120543668779, t1 = 0.48, t2 = 1.49
    Iter 14 (CCD): obj = -7044.876172687189, δ = 0.0073148711066542684, t1 = 0.51, t2 = 1.65, t3 = 0.01
    Iter 15 (PCA): obj = -7023.629304760571, δ = 0.03250775080424135, t1 = 0.53, t2 = 1.72
    Iter 16 (CCD): obj = -7012.850120947548, δ = 0.006854890650307496, t1 = 0.56, t2 = 1.88, t3 = 0.01
    Iter 17 (PCA): obj = -6996.2718588537, δ = 0.03073822907867497, t1 = 0.58, t2 = 1.94
    Iter 18 (CCD): obj = -6986.792631541138, δ = 0.00643070491605296, t1 = 0.61, t2 = 2.1, t3 = 0.01
    Iter 19 (PCA): obj = -6973.531686161966, δ = 0.028115057674247466, t1 = 0.63, t2 = 2.16
    Iter 20 (CCD): obj = -6965.156942284143, δ = 0.006029270847764364, t1 = 0.66, t2 = 2.33, t3 = 0.01
    Iter 21 (PCA): obj = -6954.364877631307, δ = 0.025782754495081224, t1 = 0.68, t2 = 2.38
    Iter 22 (CCD): obj = -6946.941292530272, δ = 0.0056444702455843545, t1 = 0.71, t2 = 2.55, t3 = 0.01
    Iter 23 (PCA): obj = -6938.050251132212, δ = 0.023339824851938744, t1 = 0.74, t2 = 2.61
    Iter 24 (CCD): obj = -6931.429620290803, δ = 0.005272863979624524, t1 = 0.77, t2 = 2.77, t3 = 0.02
    Iter 25 (PCA): obj = -6924.038356243489, δ = 0.020838710472615027, t1 = 0.78, t2 = 2.83
    Iter 26 (CCD): obj = -6918.112876107202, δ = 0.004918165386397138, t1 = 0.81, t2 = 2.99, t3 = 0.02
    Iter 27 (PCA): obj = -6911.905927691633, δ = 0.01859000844736733, t1 = 0.83, t2 = 3.05
    Iter 28 (CCD): obj = -6906.577950472116, δ = 0.004577091626964406, t1 = 0.86, t2 = 3.21, t3 = 0.02
     19.578308 seconds (65.69 M allocations: 3.483 GiB, 4.39% gc time, 77.76% compilation time)


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




    0.007406506932664477



## Second order group knockoffs

In practice, we often do not have the true covariance matrix $\Sigma$ and the true means $\mu$. In that case, we can generate second order group knockoffs via the 3 argument function


```julia
me_second_order = modelX_gaussian_group_knockoffs(X, :maxent, groups);
```

This will estimate the covariance matrix via a shrinkage estimator, see documentation API for more details. 

## Representative group knockoffs

One can choose a few representatives from each group and generate *representative* group knockoffs, with the following advantage:

+ Dramatically improved computational efficiency, since the group-knockoff optimization problem only needs to be carried out on the representative variables.
+ Improved power over standard group knockoffs, since the exchangeability have to be satisfied for less variables, so the resulting knockoffs are more "flexible"

This model assume that conditional on the group representatives, remaining variables are independent by groups. Although this assumption is not always met, we find that group-FDR is never really violated in our experiments with real or simulated data. 


```julia
@time rME = modelX_gaussian_rep_group_knockoffs(
    X, :maxent, groups, μ, Σ, 
    m = 5,               # number of knockoffs per variable to generate
    tol = 0.0001,        # convergence tolerance
    rep_threshold = 0.8, # R2 threshold for choosing representatives
    verbose=true);       # whether to print informative intermediate results
```

    100 representatives for 500 variables, 108 optimization variables
    Maxent initial obj = -1251.9504482429606
    Iter 1 (PCA): obj = -779.3679715355898, δ = 0.396729400414175, t1 = 0.0, t2 = 0.0
    Iter 2 (CCD): obj = -775.060406275815, δ = 0.0732451631869636, t1 = 0.0, t2 = 0.0, t3 = 0.0
    Iter 3 (PCA): obj = -742.0320684018326, δ = 0.35854568776631224, t1 = 0.0, t2 = 0.0
    Iter 4 (CCD): obj = -742.0051586067349, δ = 0.005820330294768541, t1 = 0.0, t2 = 0.0, t3 = 0.0
    Iter 5 (PCA): obj = -740.2381969308991, δ = 0.06132916410588748, t1 = 0.0, t2 = 0.0
    Iter 6 (CCD): obj = -740.2377890912246, δ = 0.0007984668584872479, t1 = 0.0, t2 = 0.0, t3 = 0.0
    Iter 7 (PCA): obj = -739.5557857700379, δ = 0.027095277708548027, t1 = 0.0, t2 = 0.0
    Iter 8 (CCD): obj = -739.5554840270346, δ = 0.0006939533604367249, t1 = 0.0, t2 = 0.0, t3 = 0.0
    Iter 9 (PCA): obj = -739.364132913859, δ = 0.014219568405133007, t1 = 0.0, t2 = 0.0
    Iter 10 (CCD): obj = -739.3640794172708, δ = 0.00026751205521428727, t1 = 0.0, t2 = 0.0, t3 = 0.0
    Iter 11 (PCA): obj = -739.3004784935146, δ = 0.00699480271009976, t1 = 0.0, t2 = 0.0
    Iter 12 (CCD): obj = -739.3004578732453, δ = 0.00021513370749886738, t1 = 0.0, t2 = 0.0, t3 = 0.0
      0.255370 seconds (20.49 k allocations: 219.477 MiB)


Note that the resulting knockoffs are still $n \times mp$, so we do sample knockoffs for each variable even though the optimization was only carried out on a subset.


```julia
rME.Xko
```




    250×2500 Matrix{Float64}:
     -2.02635    -1.7701      -1.46708     …   1.14117    1.3035      1.51521
     -0.473473   -0.00353028  -0.446125       -1.43538   -1.61465    -1.0251
     -0.434336   -1.23647     -1.25906        -0.6267    -0.272594   -0.431613
      1.56359     0.278573     0.671124        0.630377   0.579558    1.0876
      0.387161    0.856616     1.44796        -0.760866  -0.748318   -0.495847
     -0.791019   -0.98869     -0.396516    …   1.6927     0.214687   -0.125589
      1.26676     0.361408     0.194175       -0.823221  -0.99416    -0.529763
      0.0446295  -0.0775493   -0.0545377      -0.143951  -0.243873    0.138882
     -1.43764    -1.35191     -1.47821         0.372513   0.147667    0.263357
     -0.445685   -0.409486    -0.117022       -0.539456  -0.979681   -1.13224
     -1.12771    -0.895357    -0.875399    …  -0.428835  -0.722677   -0.0963992
      3.24501     1.87553      1.4093          0.225577  -0.482896   -0.58531
     -0.406158   -0.104083    -0.208398       -1.58035   -1.28124    -1.07925
      ⋮                                    ⋱                         
      0.0744866  -0.12148     -0.506796        2.27877    1.59647     1.08393
     -0.639735   -0.0899262   -0.682674        2.26979    2.36776     1.23767
     -0.743036   -0.078543     0.190691    …   0.280561   0.683024    0.604584
      0.216802   -0.235232    -0.583439       -1.14196   -0.837435   -1.26028
      1.39833     0.642316     0.752464       -2.25555   -2.05632    -1.58866
     -1.35121    -1.18921     -1.48731         1.83777    1.85815     1.37793
      0.0421617   0.0257475    0.00332313      0.120749  -0.126425   -0.245098
     -0.257331   -0.0124215    0.259443    …  -0.228297  -0.260701   -0.571025
      2.36186     1.48677      1.10764         0.599617   0.888141    0.566555
     -2.95162    -1.58974     -1.50044         0.488967   0.510731   -0.0708063
      1.8735      1.90576      1.54649         0.263684  -0.0619415   0.573829
     -1.31383    -0.977912    -2.02039         1.34478    0.644186    0.77451



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
