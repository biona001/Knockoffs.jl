
# Group Knockoffs

This tutorial generates group (model-X) knockoffs, which is useful when predictors are highly correlated. The methodology is described in the following paper

> Dai R, Barber R. The knockoff filter for FDR control in group-sparse and multitask regression. InInternational conference on machine learning 2016 Jun 11 (pp. 1851-1859). PMLR.


!!! note

    In the original paper, Dai and Barber only describes how to construct a suboptimal equi-correlated group knockoffs. Here we implement fully generalized alternatives.
    
Currently available options for group knockoffs:
+ `:maxent`: Fully general maximum entropy (maxent) group knockoff, based on coordinate descent.
+ `:mvr`: Fully general minimum variance-based reconstructability (MVR) group knockoff, based on coordinate descent.
+ `:sdp`: This generalizes the equi-correlated group knockoff idea by having $S_j = \gamma_j \Sigma_{(G_j, G_j)}$. Instead of optimizing over all variables in $S$, we optimize over a vector $\gamma_1,...,\gamma_G$. 
+ `:equi`: This implements the equi-correlated idea proposed in [Barber and Dai](https://proceedings.mlr.press/v48/daia16.html), which lets $S_j = \gamma \Sigma_{(G_j, G_j)}$ where $\Sigma_{(G_j, G_j)}$ is the block of $\Sigma$ containing variables in the $j$th group. Thus, instead of optimizing over all variables in $S$, we optimize a scalar $\gamma$. Conveniently, there a simple closed form solution for $\gamma$. For `mvr` and `maxent` group knockoffs, we initialize $S$ using this construction. 



```julia
# load packages for this tutorial
using Revise
using Knockoffs
using LinearAlgebra
using Random
using StatsKit
using ToeplitzMatrices
using Distributions
```

# Data simulation

## Gaussian model-X knockoffs with known mean and covariance

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

# Define group memberships

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
@time Gme = modelX_gaussian_group_knockoffs(
    X, :maxent, groups, μ, Σ, 
    m = 5,              # number of knockoffs per variable to generate
    tol = 0.0001,       # convergence tolerance
    inner_ccd_iter = 1, # optimize every entry of S exactly 1 time before moving on to PCA updates
    inner_pca_iter = 1, # optimize S with respect to pre-computed eigenvectors 1 time before going to CCA updates
    verbose=true);      # whether to print informative intermediate results
```

    Maxent initial obj = -10748.931182611366
    Iter 1 (PCA): obj = -8237.523985375447, δ = 0.16612980949819264, t1 = 0.03, t2 = 0.07
    Iter 2 (CCD): obj = -7700.4180043840925, δ = 0.03731459026023631, t1 = 0.1, t2 = 0.23, t3 = 0.0
    Iter 3 (PCA): obj = -7425.308463255121, δ = 0.060823604038161845, t1 = 0.14, t2 = 0.3
    Iter 4 (CCD): obj = -7308.074499758944, δ = 0.019936923421413913, t1 = 0.18, t2 = 0.47, t3 = 0.0
    Iter 5 (PCA): obj = -7229.089072737786, δ = 0.036631251375332345, t1 = 0.21, t2 = 0.53
    Iter 6 (CCD): obj = -7182.085494609535, δ = 0.009753360860322807, t1 = 0.24, t2 = 0.7, t3 = 0.0
    Iter 7 (PCA): obj = -7142.003455889968, δ = 0.02924379554922749, t1 = 0.27, t2 = 0.75
    Iter 8 (CCD): obj = -7116.141211182261, δ = 0.008709415093997086, t1 = 0.3, t2 = 0.92, t3 = 0.01
    Iter 9 (PCA): obj = -7088.633103966935, δ = 0.021931911964248568, t1 = 0.33, t2 = 0.98
    Iter 10 (CCD): obj = -7071.026287747464, δ = 0.007971315641216871, t1 = 0.36, t2 = 1.15, t3 = 0.01
    Iter 11 (PCA): obj = -7049.550647201161, δ = 0.018181164682472842, t1 = 0.39, t2 = 1.21
    Iter 12 (CCD): obj = -7035.956937271431, δ = 0.007331376400204198, t1 = 0.43, t2 = 1.37, t3 = 0.01
    Iter 13 (PCA): obj = -7018.249208462705, δ = 0.017763228347353627, t1 = 0.45, t2 = 1.44
    Iter 14 (CCD): obj = -7007.066219165877, δ = 0.006763792193718139, t1 = 0.48, t2 = 1.6, t3 = 0.01
    Iter 15 (PCA): obj = -6992.312356584215, δ = 0.017052323184397946, t1 = 0.51, t2 = 1.66
    Iter 16 (CCD): obj = -6982.801633080217, δ = 0.006238045378787146, t1 = 0.54, t2 = 1.83, t3 = 0.01
    Iter 17 (PCA): obj = -6970.503897017125, δ = 0.0169181936791644, t1 = 0.58, t2 = 1.89
    Iter 18 (CCD): obj = -6962.256075073076, δ = 0.005754704016835743, t1 = 0.61, t2 = 2.06, t3 = 0.01
    Iter 19 (PCA): obj = -6951.98847526622, δ = 0.017495516481736216, t1 = 0.64, t2 = 2.12
    Iter 20 (CCD): obj = -6944.75205618899, δ = 0.005307117648802748, t1 = 0.67, t2 = 2.28, t3 = 0.02
    Iter 21 (PCA): obj = -6936.16334351868, δ = 0.01803786656173507, t1 = 0.71, t2 = 2.34
    Iter 22 (CCD): obj = -6929.739879775674, δ = 0.004890636323370091, t1 = 0.75, t2 = 2.51, t3 = 0.02
    Iter 23 (PCA): obj = -6922.524913590562, δ = 0.017576002827777423, t1 = 0.77, t2 = 2.57
    Iter 24 (CCD): obj = -6916.783196930395, δ = 0.00450512978065051, t1 = 0.8, t2 = 2.74, t3 = 0.02
    Iter 25 (PCA): obj = -6910.701374793149, δ = 0.01648855402262321, t1 = 0.83, t2 = 2.8
    Iter 26 (CCD): obj = -6905.531718965983, δ = 0.004141478762581323, t1 = 0.86, t2 = 2.96, t3 = 0.02
    Iter 27 (PCA): obj = -6900.35669742446, δ = 0.015157719648886831, t1 = 0.88, t2 = 3.02
    Iter 28 (CCD): obj = -6895.6834422341135, δ = 0.0037992078852924147, t1 = 0.92, t2 = 3.19, t3 = 0.02
    Iter 29 (PCA): obj = -6891.254461129407, δ = 0.01392451146460574, t1 = 0.94, t2 = 3.25
    Iter 30 (CCD): obj = -6887.0118391955575, δ = 0.0034762384975926385, t1 = 0.97, t2 = 3.42, t3 = 0.02
    Iter 31 (PCA): obj = -6883.190632395787, δ = 0.012848883381779091, t1 = 1.0, t2 = 3.48
    Iter 32 (CCD): obj = -6879.328073344284, δ = 0.0032005500692677477, t1 = 1.03, t2 = 3.64, t3 = 0.02
    Iter 33 (PCA): obj = -6876.01003814804, δ = 0.011952533693101768, t1 = 1.05, t2 = 3.7
    Iter 34 (CCD): obj = -6872.478354404191, δ = 0.0029074018026914923, t1 = 1.08, t2 = 3.87, t3 = 0.03
    Iter 35 (PCA): obj = -6869.5744362024125, δ = 0.01121455390103929, t1 = 1.11, t2 = 3.93
    Iter 36 (CCD): obj = -6866.3336891099325, δ = 0.0026315953771842587, t1 = 1.14, t2 = 4.09, t3 = 0.03
    Iter 37 (PCA): obj = -6863.769768030753, δ = 0.0107275159524292, t1 = 1.19, t2 = 4.15
    Iter 38 (CCD): obj = -6860.79269413788, δ = 0.0024740941046095376, t1 = 1.22, t2 = 4.33, t3 = 0.03
    Iter 39 (PCA): obj = -6858.51594212795, δ = 0.010298664583133318, t1 = 1.25, t2 = 4.39
    Iter 40 (CCD): obj = -6855.771264206441, δ = 0.0024064146309562212, t1 = 1.28, t2 = 4.55, t3 = 0.03
    Iter 41 (PCA): obj = -6853.735446664343, δ = 0.009915572984871833, t1 = 1.3, t2 = 4.61
    Iter 42 (CCD): obj = -6851.2074750216725, δ = 0.0023357786291544956, t1 = 1.33, t2 = 4.77, t3 = 0.03
    Iter 43 (PCA): obj = -6849.379593784872, δ = 0.009608287999421476, t1 = 1.36, t2 = 4.83
    Iter 44 (CCD): obj = -6847.045770099706, δ = 0.0022650709455480852, t1 = 1.39, t2 = 5.0, t3 = 0.03
    Iter 45 (PCA): obj = -6845.399072151567, δ = 0.00935618001100869, t1 = 1.42, t2 = 5.06
    Iter 46 (CCD): obj = -6843.24183819163, δ = 0.0021953758961676087, t1 = 1.45, t2 = 5.22, t3 = 0.03
    Iter 47 (PCA): obj = -6841.753367578095, δ = 0.009027268125129754, t1 = 1.47, t2 = 5.28
    Iter 48 (CCD): obj = -6839.759267726732, δ = 0.0021548981851473624, t1 = 1.5, t2 = 5.44, t3 = 0.04
    Iter 49 (PCA): obj = -6838.408794182777, δ = 0.008816217716468795, t1 = 1.53, t2 = 5.5
    Iter 50 (CCD): obj = -6836.561658397979, δ = 0.002085650424855929, t1 = 1.56, t2 = 5.66, t3 = 0.04
    Iter 51 (PCA): obj = -6835.338059571331, δ = 0.008583998864277846, t1 = 1.58, t2 = 5.72
    Iter 52 (CCD): obj = -6833.621680052428, δ = 0.0020193622588125446, t1 = 1.61, t2 = 5.88, t3 = 0.04
    Iter 53 (PCA): obj = -6832.507705021808, δ = 0.008357659672822983, t1 = 1.64, t2 = 5.94
    Iter 54 (CCD): obj = -6830.913171864168, δ = 0.0019596122853489293, t1 = 1.67, t2 = 6.11, t3 = 0.04
    Iter 55 (PCA): obj = -6829.897889026071, δ = 0.008166006257203585, t1 = 1.7, t2 = 6.17
    Iter 56 (CCD): obj = -6828.413002054003, δ = 0.0018945753621293698, t1 = 1.73, t2 = 6.33, t3 = 0.04
    Iter 57 (PCA): obj = -6827.484840513088, δ = 0.007944022932777064, t1 = 1.75, t2 = 6.39
    Iter 58 (CCD): obj = -6826.10062438168, δ = 0.0018359149332439267, t1 = 1.78, t2 = 6.55, t3 = 0.04
    Iter 59 (PCA): obj = -6825.2494332900915, δ = 0.007782413558754493, t1 = 1.8, t2 = 6.62
    Iter 60 (CCD): obj = -6823.956997438751, δ = 0.0017780526622721631, t1 = 1.83, t2 = 6.79, t3 = 0.04
    Iter 61 (PCA): obj = -6823.17494882124, δ = 0.007545428624149307, t1 = 1.86, t2 = 6.84
    Iter 62 (CCD): obj = -6821.965768790584, δ = 0.0017207312595902264, t1 = 1.89, t2 = 7.01, t3 = 0.05
    Iter 63 (PCA): obj = -6821.246264870465, δ = 0.007434174308049364, t1 = 1.91, t2 = 7.07
    Iter 64 (CCD): obj = -6820.113139892253, δ = 0.0016625779896528147, t1 = 1.94, t2 = 7.24, t3 = 0.05
    Iter 65 (PCA): obj = -6819.451373736169, δ = 0.0072737103797947756, t1 = 1.97, t2 = 7.3
    Iter 66 (CCD): obj = -6818.387283413891, δ = 0.0015918188550377254, t1 = 2.0, t2 = 7.47, t3 = 0.05
    Iter 67 (PCA): obj = -6817.774865622847, δ = 0.00711334427118358, t1 = 2.02, t2 = 7.53
    Iter 68 (CCD): obj = -6816.773377927246, δ = 0.0015526989102086811, t1 = 2.05, t2 = 7.69, t3 = 0.05
    Iter 69 (PCA): obj = -6816.205085751263, δ = 0.006988029537547295, t1 = 2.08, t2 = 7.75
    Iter 70 (CCD): obj = -6815.261423062906, δ = 0.001482617801898897, t1 = 2.11, t2 = 7.91, t3 = 0.05
    Iter 71 (PCA): obj = -6814.733758473422, δ = 0.006851280378816475, t1 = 2.14, t2 = 7.97
    Iter 72 (CCD): obj = -6813.844204400972, δ = 0.0014295027056425783, t1 = 2.17, t2 = 8.14, t3 = 0.05
    Iter 73 (PCA): obj = -6813.352492935046, δ = 0.006708844719793019, t1 = 2.19, t2 = 8.2
    Iter 74 (CCD): obj = -6812.512641423066, δ = 0.0013762433710736894, t1 = 2.22, t2 = 8.37, t3 = 0.05
    Iter 75 (PCA): obj = -6812.053537854734, δ = 0.0065632627924375, t1 = 2.24, t2 = 8.42
    Iter 76 (CCD): obj = -6811.259372541207, δ = 0.001342038854451493, t1 = 2.27, t2 = 8.59, t3 = 0.06
    Iter 77 (PCA): obj = -6810.8300945867395, δ = 0.0064501287333486295, t1 = 2.29, t2 = 8.65
    Iter 78 (CCD): obj = -6810.079085892393, δ = 0.001299214537577804, t1 = 2.32, t2 = 8.81, t3 = 0.06
    Iter 79 (PCA): obj = -6809.676743437872, δ = 0.006344899636897539, t1 = 2.35, t2 = 8.87
    Iter 80 (CCD): obj = -6808.965593049298, δ = 0.0012766626080034616, t1 = 2.37, t2 = 9.03, t3 = 0.06
    Iter 81 (PCA): obj = -6808.586676794652, δ = 0.00623872268311287, t1 = 2.4, t2 = 9.09
    Iter 82 (CCD): obj = -6807.912558663124, δ = 0.0012551259521810696, t1 = 2.42, t2 = 9.26, t3 = 0.06
     11.951811 seconds (108.59 k allocations: 235.689 MiB)


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
    X̃::Matrix{T} # n × mp matrix storing knockoffs of X
    groups::Vector{Int} # p × 1 vector of group membership
    S::BD # p × p block-diagonal matrix of the same size as Σ. S and (m+1)/m*Σ - S are both psd
    γs::Vector{T} # for suboptimal group construction only. These are scalars chosen so that S_i = γ_i * Σ_i
    m::Int # number of knockoffs per feature generated
    Σ::S # p × p symmetric covariance matrix. 
    method::Symbol # method for solving s
    obj::T # final objective value of group knockoff
end
```
Given this result, lets do a sanity check: is $(m+1)/m\Sigma - S$ positive semi-definite?


```julia
eigmin((m+1)/m*Gme.Σ - Gme.S)
```




    0.05110858707177174



## Second order group knockoffs

In practice, we often do not have the true covariance matrix $\Sigma$ and the true means $\mu$. In that case, we can generate second order group knockoffs via the 3 argument function


```julia
Gme_second_order = modelX_gaussian_group_knockoffs(X, :maxent, groups);
```

This will estimate the covariance matrix via a shrinkage estimator, see documentation API for more details. 

## Representative group knockoffs

One can choose a few representatives from each group and generate *representative* group knockoffs, with the following advantage:

+ Dramatically improved computational efficiency, since the group-knockoff optimization problem only needs to be carried out on the representative variables.
+ Improved power over standard group knockoffs, since the exchangeability have to be satisfied for less variables, so the resulting knockoffs are more "flexible"

This model assume that conditional on the group representatives, remaining variables are independent by groups. Although this assumption is not always met, we find that group-FDR is never really violated in our experiments with real or simulated data. 


```julia
@time rME = modelX_gaussian_rep_group_knockoffs(
    X, :maxent, μ, Σ, groups,
    m = 5,          # number of knockoffs per variable to generate
    tol = 0.0001,   # convergence tolerance
    verbose=true);  # whether to print informative intermediate results
```

    96 representatives for 500 variables, 96 optimization variables
    Iter 1: δ = 0.14953283217899976
    Iter 2: δ = 0.1991610624266248
    Iter 3: δ = 0.01932169475512019
    Iter 4: δ = 0.005068052646704513
    Iter 5: δ = 0.0009112831450636683
    Iter 6: δ = 0.0001230274318336222
    Iter 7: δ = 2.4675893956049855e-5
      0.250789 seconds (15.34 k allocations: 214.701 MiB, 14.04% gc time)


Note that the resulting knockoffs are still $n \times mp$


```julia
rME.X̃
```




    250×2500 Matrix{Float64}:
     -0.747196   -1.2342    -0.702566    …   1.66098    1.79071     1.40427
      0.600782    0.109681  -1.2857         -0.81733   -0.876087   -0.539925
     -1.4536     -1.53914   -1.76241        -1.07475   -0.982822   -0.364055
     -1.24691    -0.878209  -0.122253        0.882058   0.698461    1.27731
      0.669513    0.478596   0.718306       -1.11829   -0.958759    0.00439087
     -1.04199    -0.784127  -1.82756     …   0.905066   0.748423    0.339189
     -0.754254   -0.33635    0.442443        0.24998   -0.0987811   0.0899613
     -2.35308    -1.81752   -2.28223        -0.164806  -0.104967   -0.447325
     -2.20415    -2.76933   -2.59485        -0.905337  -0.745101    0.237391
      2.20236     2.1198     1.64855        -1.02309   -1.22663    -0.745322
     -1.46614    -0.198733  -0.508032    …  -1.75448   -2.04408    -1.30121
      0.0185783  -0.123839  -0.524711       -0.169963  -0.0599242   0.0216337
      0.222628    0.110846  -0.0438031      -1.09253   -1.05077    -1.2574
      ⋮                                  ⋱                         
     -0.112625    0.305465  -0.00886701      1.04541    1.33245     0.956846
      0.804336    1.04927    0.665463        1.24162    0.889683    0.759236
     -0.732725   -0.719744  -0.423113    …   0.32596    0.521656    0.352721
     -1.94595    -0.630124  -0.136644       -2.19721   -2.19188    -1.10755
      2.188       1.77627    0.80627        -0.649521  -0.493163   -0.953951
     -1.44836    -1.32895   -1.02944         2.1705     1.53445     1.73658
     -1.72554    -1.06335   -1.27667         0.702605   0.719975    0.858033
      0.173359    0.713088   0.969075    …  -0.541989  -0.382434   -0.508017
      1.68208     1.19315    0.725897       -0.490734  -0.324412   -0.0368795
     -1.22018    -1.25161   -0.824222        0.543467   0.401156    0.398281
      1.58907     1.72442    2.18569         0.363076   0.644025    0.349402
     -0.90697    -1.35714   -2.32267         0.526776   0.0533524   0.447866



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

    Sim 1 group-knockoff power = 1.0, FDR = 0.1, time=6.71429675
    Sim 2 group-knockoff power = 0.7777777777777778, FDR = 0.0, time=7.203738083
    Sim 3 group-knockoff power = 0.8888888888888888, FDR = 0.1111111111111111, time=5.199876167
    Sim 4 group-knockoff power = 0.8, FDR = 0.0, time=7.725970875
    Sim 5 group-knockoff power = 0.7, FDR = 0.0, time=8.715202042
    Sim 6 group-knockoff power = 0.5, FDR = 0.0, time=8.797519166
    Sim 7 group-knockoff power = 1.0, FDR = 0.0, time=6.052631459
    Sim 8 group-knockoff power = 0.4444444444444444, FDR = 0.0, time=8.221799459
    Sim 9 group-knockoff power = 0.7, FDR = 0.0, time=9.489696541
    Sim 10 group-knockoff power = 0.5555555555555556, FDR = 0.0, time=6.281113166
    
    ME group knockoffs have average group power 0.7366666666666667
    ME group knockoffs have average group FDR 0.021111111111111112
    ME group knockoffs took average 7.4401843708 seconds


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

    Sim 1 nongroup-knockoff power = 0.7, FDR = 0.2222222222222222, time=5.165706875
    Sim 2 nongroup-knockoff power = 0.7, FDR = 0.0, time=5.707978708
    Sim 3 nongroup-knockoff power = 0.2, FDR = 0.0, time=4.334730542
    Sim 4 nongroup-knockoff power = 0.0, FDR = 0.0, time=6.279638458
    Sim 5 nongroup-knockoff power = 0.2, FDR = 0.0, time=7.839875459
    Sim 6 nongroup-knockoff power = 0.0, FDR = 0.0, time=7.261292667
    Sim 7 nongroup-knockoff power = 0.0, FDR = 0.0, time=4.292064292
    Sim 8 nongroup-knockoff power = 0.0, FDR = 0.0, time=7.985766
    Sim 9 nongroup-knockoff power = 0.4, FDR = 0.0, time=8.667096167
    Sim 10 nongroup-knockoff power = 0.5, FDR = 0.0, time=5.635861
    
    ME (standard) knockoffs have average group power 0.26999999999999996
    ME (standard) knockoffs have average group FDR 0.02222222222222222
    ME (standard) knockoffs took average 6.3170010168 seconds


## Conclusion

+ When variables are highly correlated so that one cannot find exact discoveries, group knockoffs may be useful for improving power as it identifies whether a group of variables are non-null without having to pinpoint the exact discovery.
+ Group knockoffs control the group FDR to be below the target FDR level. 
+ Groups do not have to be contiguous
