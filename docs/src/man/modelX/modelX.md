
# Model-X knockoffs

This tutorial is for generating model-X knockoffs, which handles cases where covariates outnumber sample size ($p > n$). The methodology is described in the following paper

> Candes E, Fan Y, Janson L, Lv J. *Panning for gold:‘model‐X’knockoffs for high dimensional controlled variable selection.* Journal of the Royal Statistical Society: Series B (Statistical Methodology). 2018 Jun;80(3):551-77.


```julia
# load packages needed for this tutorial
using Revise
using Knockoffs
using Plots
using Random
using GLMNet
using Distributions
using LinearAlgebra
using ToeplitzMatrices
using StatsBase
gr(fmt=:png);
```

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
where $s$ is solved so that $0 \le s_j$ for all $j$ and $2Σ - diag(s)$ is PSD


```julia
Random.seed!(2022)
n = 100 # sample size
p = 500 # number of covariates
ρ = 0.4
Σ = Matrix(SymmetricToeplitz(ρ.^(0:(p-1)))) # true covariance matrix
μ = zeros(p) # true mean parameters
L = cholesky(Σ).L
X = randn(n, p) * L # var(X) = L var(N(0, 1)) L' = var(Σ)
```




    100×500 Matrix{Float64}:
     -0.877527   -1.30346    -0.682964   …  -0.350442   -2.05031    -1.19774
      1.94981     0.625748   -0.157094       0.328944   -0.50083    -1.77715
     -0.372347   -0.0577255  -0.545009       2.04892     0.447051   -0.172137
      0.281239   -1.2314     -2.25883       -0.863995   -1.29658    -1.69292
     -1.94197    -0.514252   -0.126406       1.58385    -0.508209   -0.576327
      0.32584     1.29406    -0.944499   …   0.336041    0.834595    1.01313
      0.312506   -0.892288   -1.28543        0.656042   -0.727716   -0.062318
     -0.130798    0.080975   -1.00406        0.140716   -0.0171299   0.0520695
     -1.37595    -1.7559     -0.965395       0.475523    0.476843    0.0988572
      1.17982     1.1271      1.29655       -1.16738    -1.72623    -0.848002
      0.328697    0.674708    1.08453    …   0.886085   -0.405945   -0.190064
     -0.155799    0.740727    0.0747548      0.694317    0.483984    0.791628
     -2.03164    -0.90797    -1.57553        2.3821      1.89785     0.318261
      ⋮                                  ⋱                          
     -0.950099   -1.38457     0.158824      -0.187195    1.70564     1.20939
      0.236699   -0.22054    -0.852487       0.44721    -1.06464     1.19481
     -0.409184    0.393978    0.894276   …  -0.603513   -0.0926733  -0.979548
     -0.20095    -0.200403   -1.76704       -0.151618    0.0987953   1.35385
     -0.366858    0.719476    0.234908      -0.245508   -0.321876   -0.420004
     -0.232295    0.305845   -0.494038       0.910901    1.67494    -0.114834
      0.712566    0.86745     1.08866       -0.161063   -1.05527    -0.604559
     -0.413006    0.322623    0.152769   …  -0.438893    0.731315   -0.398558
      0.0977592  -0.301898    0.0328142      1.17656     0.45638     0.232298
      0.988324   -0.698384   -0.436609      -0.466774   -0.0569571  -0.499042
      0.132829    0.26532     1.60743        0.955361   -0.626877   -0.242588
     -0.525434   -0.854175    0.170717       0.0667342  -2.16955    -0.63145



To generate knockoffs, the 4 argument function [modelX\_gaussian\_knockoffs](https://biona001.github.io/Knockoffs.jl/dev/man/api/#Knockoffs.modelX_gaussian_knockoffs) will generate exact model-X knockoffs. The 2nd argument specifies the method to generate knockoffs. We generally recommend `:mvr` or `:maxent` because they are [more efficient to compute and tend to be more powerful than the SDP construction](https://projecteuclid.org/journals/annals-of-statistics/volume-50/issue-1/Powerful-knockoffs-via-minimizing-reconstructability/10.1214/21-AOS2104.short). The 3rd and 4th argument supplies the true mean and covariance of features.


```julia
@time mvr = modelX_gaussian_knockoffs(X, :mvr, μ, Σ);
```

     11.227334 seconds (32.85 M allocations: 1.709 GiB, 4.59% gc time, 96.61% compilation time)


The return type is a `GaussianKnockoff` struct, which contains the following fields

```julia
struct GaussianKnockoff{T<:AbstractFloat, M<:AbstractMatrix, S <: Symmetric} <: Knockoff
    X::M # n × p design matrix
    X̃::Matrix{T} # n × p knockoff of X
    s::Vector{T} # p × 1 vector. Diagonal(s) and 2Σ - Diagonal(s) are both psd
    Σ::S # p × p symmetric covariance matrix. 
    method::Symbol # method for solving s
end
```

Thus, to access these fields, one can do e.g.


```julia
s = mvr.s
```




    500-element Vector{Float64}:
     0.7055837242604757
     0.5505996431325556
     0.5579633308207881
     0.5578990459757973
     0.5578830358745575
     0.5578842779646903
     0.557884239188223
     0.5578842383629335
     0.557884238524227
     0.557884238515753
     0.5578842385158744
     0.5578842385158905
     0.5578842385158891
     ⋮
     0.5578842369066757
     0.5578842348979727
     0.5578842398872366
     0.5578842817258646
     0.5578843495474252
     0.5578842071395304
     0.5578843562366873
     0.5578830621807334
     0.5578990896765453
     0.5579633702735247
     0.5505996786150797
     0.7055838005026088



## Second order knockoffs

The 2 argument [modelX\_gaussian\_knockoffs](https://biona001.github.io/Knockoffs.jl/dev/man/api/#Knockoffs.modelX_gaussian_knockoffs) will estimate the mean and covariance of `X` and use them to generate model-X knockoffs


```julia
# make 2nd order knockoffs
@time mvr_2nd_order = modelX_gaussian_knockoffs(X, :mvr);
```

      1.023831 seconds (3.42 M allocations: 204.294 MiB, 3.40% gc time, 70.10% compilation time)


## Approximate construction for speed

Generating model-X knockoffs scales as $\mathcal{O}(p^3)$ with coordinate descent (e.g. `sdp_fast`, `mvr`, `maxent`), which becomes prohibitively slow for large $p$ (e.g. $p = 5000$). 

Sometimes one expects that covariates are only correlated with its nearby neighbors. Then, we can approximate the covariance matrix as a block diagonal structure with block size `windowsize`, and solve each block independently as smaller problems. This is implemented as [approx\_modelX\_gaussian\_knockoffs](https://biona001.github.io/Knockoffs.jl/dev/man/api/#Knockoffs.approx_modelX_gaussian_knockoffs)


```julia
@time mvr_approx = approx_modelX_gaussian_knockoffs(X, :mvr, windowsize=100);
```

      6.919185 seconds (30.79 M allocations: 1.702 GiB, 4.76% gc time, 98.03% compilation time)


## Multiple knockoffs

[Gimenez et al](http://proceedings.mlr.press/v89/gimenez19b.html) suggested multiple simultaneous knockoffs, which can give a boost in power when the target FDR or the number of variables to select are low. 

If one generated $m$ knockoffs for each of the $p$ variables, the convex optimization problem in solving for diagonal $S$ matrix is equally efficient as in the single-knockoff case, but the subsequent model selection would have $(m + 1) * p$ columns as opposed to $2p$ columns in the single-knockoff case. Thus, both computational speed and memory demand scales roughly linearly in $m$. 


```julia
m = 3
@time mvr_multiple = modelX_gaussian_knockoffs(X, :mvr, μ, Σ, m=m);
```

      0.698913 seconds (2.81 k allocations: 80.742 MiB, 2.11% gc time, 0.96% compilation time)


As a sanity check, lets make sure the modified SDP constraint is satisfied


```julia
eigmin((m+1)/m * Σ - Diagonal(mvr_multiple.s))
```




    0.0714938929162792



Finally, we can compare the `s` vector estimated from all 4 methods.


```julia
[mvr.s mvr_2nd_order.s mvr_approx.s mvr_multiple.s]
```




    500×4 Matrix{Float64}:
     0.705584  0.765939  0.663288  0.659554
     0.5506    0.809084  0.697905  0.490815
     0.557963  1.04366   0.921062  0.499175
     0.557899  0.86139   0.761527  0.499988
     0.557883  0.911251  0.79024   0.499943
     0.557884  0.978194  0.861529  0.499939
     0.557884  0.919028  0.801033  0.499939
     0.557884  0.738538  0.667032  0.499939
     0.557884  0.867094  0.766879  0.499939
     0.557884  0.85405   0.741398  0.499939
     0.557884  0.719779  0.643368  0.499939
     0.557884  1.17217   1.0488    0.499939
     0.557884  0.827193  0.71336   0.499939
     ⋮                             
     0.557884  0.817115  0.692749  0.499939
     0.557884  0.991195  0.855441  0.499939
     0.557884  0.604032  0.534393  0.499939
     0.557884  0.67311   0.593851  0.499939
     0.557884  0.867172  0.765961  0.49994
     0.557884  0.877053  0.784296  0.499939
     0.557884  0.962095  0.843415  0.499939
     0.557883  1.09807   0.955454  0.499943
     0.557899  0.929814  0.821244  0.499988
     0.557963  0.961874  0.833571  0.499174
     0.5506    0.868214  0.779083  0.490815
     0.705584  0.685257  0.598158  0.659554



In this example, they are quite different.

## LASSO example

Let us apply the generated knockoffs to the model selection problem

> Given response $\mathbf{y}_{n \times 1}$, design matrix $\mathbf{X}_{n \times p}$, we want to select a subset $S \subset \{1,...,p\}$ of variables that are truly causal for $\mathbf{y}$. 

### Simulate data

We will simulate 

$$\mathbf{y} \sim N(\mathbf{X}\mathbf{\beta}, \mathbf{\epsilon}), \quad \mathbf{\epsilon} \sim N(0, 1)$$

where $k=50$ positions of $\mathbf{\beta}$ is non-zero with effect size $\beta_j \sim N(0, 1)$. The goal is to recover those 50 positions using LASSO.


```julia
# set seed for reproducibility
Random.seed!(123)

# simulate true beta
n, p = size(X)
k = 50
βtrue = zeros(p)
βtrue[1:k] .= randn(k)
shuffle!(βtrue)

# find true causal variables
correct_position = findall(!iszero, βtrue)

# simulate y
y = X * βtrue + randn(n)
```




    100-element Vector{Float64}:
      10.430304834810705
       9.430844234150015
       2.411065680091229
      14.925207979666721
      -8.743596474193755
       5.775257576204116
      -1.587996940979867
      13.605140626431009
     -18.781349632593464
       7.855239413947385
      -5.7381170081617405
      -4.631197619265106
     -11.89784179331142
       ⋮
     -10.887325590058026
      13.949139924153947
      -2.54163219529571
       3.1895938123917973
       1.0637262185056753
      -7.626490623432161
       1.8563101020426065
       1.066601624640122
      -7.242249824299234
      -3.2289570492580513
       4.014480647244052
       4.832304427120103



### Standard LASSO

Lets try running standard LASSO. We use LASSO solver in [GLMNet.jl](https://github.com/JuliaStats/GLMNet.jl) package, which is just a Julia wrapper for the GLMnet Fortran code. 

How does it perform in power and FDR?


```julia
# run 10-fold cross validation to find best λ minimizing MSE
lasso_cv = glmnetcv(X, y)
λbest = lasso_cv.lambda[argmin(lasso_cv.meanloss)]

# use λbest to fit LASSO on full data
βlasso = glmnet(X, y, lambda=[λbest]).betas[:, 1]

# check power and false discovery rate
power = length(findall(!iszero, βlasso) ∩ correct_position) / k
FDR = length(setdiff(findall(!iszero, βlasso), correct_position)) / count(!iszero, βlasso)
println("Lasso power = $power, FDR = $FDR")
```

    Lasso power = 0.54, FDR = 0.7


More than half of all Lasso discoveries are false positives. 

### Knockoff+LASSO

Now lets try applying the knockoff methodology. Recall that consists of a few steps 

1. Run LASSO on $[\mathbf{X} \mathbf{\tilde{X}}]$
2. Compare feature importance score $W_j = \text{score}(x_j) - \text{score}(\tilde{x}_j)$ for each $j = 1,...,p$. Here we use $W_j = |\beta_j| - |\tilde{\beta}_{j}|$
3. Choose target FDR $q \in [0, 1]$ and compute 
$$\tau = min_{t}\left\{t > 0: \frac{{\{\#j: W_j ≤ -t}\}}{max(1, {\{\#j: W_j ≥ t}\})} \le q\right\}$$


!!! note
    
    In step 1, $[\mathbf{X} \mathbf{\tilde{X}}]$ is written for notational convenience. In practice one must interleave knockoffs with the original variables, where either the knockoff come first or the original genotype come first with equal probability. This is due to the inherent bias of LASSO solvers: when the original and knockoff variable are equally valid, the one listed first will be selected. We export a convenient function `merge_knockoffs_with_original` that performs this operation. 

The `fit_lasso` function generates knockoffs, run Lasso on $[\mathbf{X} \mathbf{\tilde{X}}]$, and apply knockoff filter:


```julia
@time knockoff_filter = fit_lasso(y, X, method=:mvr);
```

      0.524779 seconds (102.90 k allocations: 60.830 MiB, 16.13% compilation time)


The return type is now a `KnockoffFilter`, which contains the following information

```julia
struct KnockoffFilter{T}
    y :: Vector{T} # n × 1 response vector
    X :: Matrix{T} # n × p matrix of original X and its knockoff interleaved randomly
    X̃ :: Matrix{T} # n × p matrix of X knockoff
    m :: Int # number of knockoffs per feature generated
    βs :: Vector{Vector{T}} # βs[i] is the p × 1 vector of effect sizes corresponding to fdr level fdr_target[i]
    a0 :: Vector{T}   # intercepts for each model in βs
    fdr_target :: Vector{T} # target FDR level for each τs and βs
    d :: UnivariateDistribution # distribution of y
    debias :: Union{Nothing, Symbol} # how βs and a0 have been debiased (`nothing` for not debiased)
end
```

Given these information, we can e.g. visualize power and FDR trade-off:


```julia
FDR = knockoff_filter.fdr_target
empirical_power = Float64[]
empirical_fdr = Float64[]
for i in eachindex(FDR)
    # extract beta for current fdr
    βknockoff = knockoff_filter.βs[i]
    
    # compute power and false discovery proportion
    power = length(findall(!iszero, βknockoff) ∩ correct_position) / k
    fdp = length(setdiff(findall(!iszero, βknockoff), correct_position)) / max(count(!iszero, βknockoff), 1)
    push!(empirical_power, power)
    push!(empirical_fdr, fdp)
end

# visualize FDR and power
power_plot = plot(FDR, empirical_power, xlabel="Target FDR", ylabel="Empirical power", legend=false, w=2)
fdr_plot = plot(FDR, empirical_fdr, xlabel="Target FDR", ylabel="Empirical FDR", legend=false, w=2)
Plots.abline!(fdr_plot, 1, 0, line=:dash)
plot(power_plot, fdr_plot)
```




![png](output_28_0.png)



### Multiple Knockoffs+LASSO

To improve selection stability, we can generate multiple simultaneous knockoffs. This is more computationally intensive but tends to boost power. 


```julia
@time knockoff_filter = fit_lasso(y, X, method=:mvr, m=3);
```

      0.667028 seconds (13.69 k allocations: 140.866 MiB, 1.09% gc time)



```julia
FDR = knockoff_filter.fdr_target
empirical_power = Float64[]
empirical_fdr = Float64[]
for i in eachindex(FDR)
    # extract beta for current fdr
    βknockoff = knockoff_filter.βs[i]
    
    # compute power and false discovery proportion
    power = length(findall(!iszero, βknockoff) ∩ correct_position) / k
    fdp = length(setdiff(findall(!iszero, βknockoff), correct_position)) / max(count(!iszero, βknockoff), 1)
    push!(empirical_power, power)
    push!(empirical_fdr, fdp)
end

# visualize FDR and power
power_plot = plot(FDR, empirical_power, xlabel="Target FDR", ylabel="Empirical power", legend=false, w=2)
fdr_plot = plot(FDR, empirical_fdr, xlabel="Target FDR", ylabel="Empirical FDR", legend=false, w=2)
Plots.abline!(fdr_plot, 1, 0, line=:dash)
plot(power_plot, fdr_plot)
```




![png](output_31_0.png)



**Conclusion:** 

+ LASSO + knockoffs controls the false discovery rate at below the target (dashed line). Thus, one trade power for FDR control. 
+ The power of standard LASSO is better, but it comes with high empirical FDR that one cannot control via cross validation. 
+ Multiple simultaneous knockoffs increases power at the expensive of larger regression problem. 

If we repeated the simulation multiple times, we expect the empirical FDR to hug the target FDR more closely.
