
# Model-X knockoffs

This tutorial generates model-X knockoffs, which handles the cases where covariates outnumber sample size ($p > n$). The methodology is described in the following paper

> Candes, Emmanuel, et al. "Panning for gold:‘model‐X’knockoffs for high dimensional controlled variable selection." Journal of the Royal Statistical Society: Series B (Statistical Methodology) 80.3 (2018): 551-577.


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

    ┌ Info: Precompiling Knockoffs [878bf26d-0c49-448a-9df5-b057c815d613]
    └ @ Base loading.jl:1423


## Gaussian model-X knockoffs with known mean and covariance

To illustrate, lets simulate data X with covariance Σ and mean μ


```julia
Random.seed!(2022)
n = 100 # sample size
p = 200 # number of covariates
ρ = 0.4
Σ = Matrix(SymmetricToeplitz(ρ.^(0:(p-1)))) # true covariance matrix
μ = zeros(p) # true mean parameters
L = cholesky(Σ).L
X = randn(n, p) * L # var(X) = L var(N(0, 1)) L' = var(Σ)
zscore!(X, mean(X, dims=1), std(X, dims=1)) # center/scale columns to mean 0 variance 1
```




    100×200 Matrix{Float64}:
     -0.855648   -1.3585    -0.588546   …  -0.790298   -0.453196    0.569768
      2.27425     0.722645  -0.0893726      1.08469    -0.161144    0.401535
     -0.296408   -0.014653  -0.457594       0.790067    0.179435    0.533175
      0.427117   -1.28076   -2.08441        0.259216    1.21389     1.69026
     -2.03399    -0.507132  -0.0602429      0.0547179   1.10609     0.620456
      0.476491    1.44359   -0.836803   …  -0.0156081   1.48788     1.5315
      0.46173    -0.914939  -1.16043        1.99601    -0.803953   -0.491643
     -0.0290117   0.13497   -0.893341       1.28744     1.10225     1.04454
     -1.40741    -1.84656   -0.856639      -0.363204   -1.02214    -0.646904
      1.42185     1.26348    1.29048       -1.67072    -0.293038   -1.0753
      0.479654    0.775461   1.08922    …   0.988298    0.97409     0.418336
     -0.0566877   0.846679   0.130706      -0.873524   -1.56209    -1.44786
     -2.13326    -0.931856  -1.4358        -0.330799   -0.566802   -0.854292
      ⋮                                 ⋱                          
     -0.935986   -1.44599    0.210507      -0.0717222  -0.804146    0.178554
      0.377811   -0.19029   -0.749463       1.03586     0.820306    1.38969
     -0.337188    0.472623   0.908622   …  -1.45156    -1.35609     0.604581
     -0.106671   -0.168566  -1.61758       -1.30896    -0.846353   -2.02408
     -0.290333    0.823755   0.282728      -0.167859    0.15075     0.0275125
     -0.14137     0.37755   -0.409211      -1.53997    -0.310223   -0.197986
      0.9046      0.983381   1.09314        0.298495   -0.692161   -0.0506053
     -0.341419    0.395648   0.204759   …   0.283953   -0.205934    1.37711
      0.224003   -0.278054   0.0908944     -1.07826    -1.42549    -0.00686257
      1.20987    -0.705764  -0.354697      -0.780509    0.730195   -0.648271
      0.262826    0.333833   1.58557        0.7471     -1.44636    -1.30526
     -0.465877   -0.873825   0.221796      -0.565835    0.0731643  -1.21793



To generate knockoffs, the 4 argument function [modelX_gaussian_knockoffs](https://biona001.github.io/Knockoffs.jl/dev/man/api/#Knockoffs.modelX_gaussian_knockoffs) will generate exact model-X knockoffs. The 2nd argument can be ither `:equi`, `:sdp`, or `:asdp`. The SDP construction will yield more powerful knockoffs, but is more computationally expensive. 


```julia
@time Xko_equi = modelX_gaussian_knockoffs(X, :equi, μ, Σ)
@time Xko_sdp = modelX_gaussian_knockoffs(X, :sdp, μ, Σ);
```

     25.882800 seconds (76.93 M allocations: 4.564 GiB, 6.25% gc time, 99.98% compilation time)
     21.144031 seconds (46.67 M allocations: 2.981 GiB, 4.94% gc time, 87.14% compilation time)


The return type is a `Knockoff` struct, which contains the following fields

```julia
struct Knockoff{T}
    X::Matrix{T}    # n × p original design matrix
    X̃::Matrix{T}    # n × p knockoff of X
    s::Vector{T}    # p × 1 vector. Diagonal(s) and 2Σ - Diagonal(s) are both psd
    Σ::Matrix{T}    # p × p gram matrix X'X
    Σinv::Matrix{T} # p × p inv(X'X)
end
```

Thus, to access these fields, one can do e.g.


```julia
s = Xko_sdp.s
```




    200-element Vector{Float64}:
     0.9999999975421989
     0.9714285562750639
     0.8114286000388236
     0.8754285608972914
     0.8498285802412486
     0.8600685709048966
     0.8559725752335535
     0.8576109732729927
     0.856955614169666
     0.8572177577255956
     0.8571129003675105
     0.8571548432807583
     0.8571380661077549
     ⋮
     0.8571548433098494
     0.8571129003234808
     0.8572177577571737
     0.8569556141700471
     0.8576109732517373
     0.8559725752488924
     0.860068570920087
     0.8498285802136867
     0.8754285609148045
     0.8114286000369729
     0.9714285562700409
     0.9999999975421956



## Second order knockoffs

The 2 argument [modelX_gaussian_knockoffs](https://biona001.github.io/Knockoffs.jl/dev/man/api/#Knockoffs.modelX_gaussian_knockoffs) will estimate the mean and covariance of `X` and use them to generate model-X knockoffs


```julia
# make equi-correlated and SDP knockoffs
@time equi = modelX_gaussian_knockoffs(X, :equi)
@time sdp = modelX_gaussian_knockoffs(X, :sdp);
```

      1.521634 seconds (4.82 M allocations: 302.390 MiB, 5.24% gc time, 99.61% compilation time)
      1.579576 seconds (1.33 M allocations: 443.163 MiB, 4.65% gc time, 18.31% compilation time)


## LASSO example

Let us apply the generated knockoffs to the model selection problem

> Given response $\mathbf{y}_{n \times 1}$, design matrix $\mathbf{X}_{n \times p}$, we want to select a subset $S \subset \{1,...,p\}$ of variables that are truly causal for $\mathbf{y}$. 

### Simulate data

We will simulate 

$$\mathbf{y} \sim N(\mathbf{X}\mathbf{\beta}, \mathbf{\epsilon}), \quad \mathbf{\epsilon} \sim N(0, 1)$$

where $k=50$ positions of $\mathbf{\beta}$ is non-zero with effect size $\beta_j \sim N(0, 1)$. The goal is to recover those 50 positions using LASSO.


```julia
# set seed for reproducibility
Random.seed!(1234)

# simulate true beta
n, p = size(X)
k = 50
βtrue = zeros(p)
βtrue[1:k] .= randn(50)
shuffle!(βtrue)

# find true causal variables
correct_position = findall(!iszero, βtrue)

# simulate y
y = X * βtrue + randn(n);
```

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
power, FDR
```




    (0.7, 0.6022727272727273)



It seems LASSO have power 96% (it missed only 2/50 predictors), but the false discovery rate is 54%. This means that although LASSO finds almost every predictor, more than half of all discoveries are false positives. 

### Knockoff+LASSO

Now lets try applying the knockoff methodology. Recall that consists of a few steps 

1. Run LASSO on $[\mathbf{X} \mathbf{\tilde{X}}]$
2. Compare feature importance score $W_j = \text{score}(x_j) - \text{score}(\tilde{x}_j)$ for each $j = 1,...,p$. Here we use $W_j = |\beta_j| - |\tilde{\beta}_{j}|$
3. Choose target FDR $q \in [0, 1]$ and compute 
$$\tau = min_{t}\left\{t > 0: \frac{{\{\#j: W_j ≤ -t}\}}{max(1, {\{\#j: W_j ≥ t}\})} \le q\right\}$$


!!! note
    
    In step 1, $[\mathbf{X} \mathbf{\tilde{X}}]$ is written for notational convenience. In practice one must interleave knockoffs with the original variables, where either the knockoff come first or the original genotype come first with equal probability. This is due to the inherent bias of LASSO solvers: when the original and knockoff variable are equally valid, the one listed first will be selected. 


```julia
# step 1 (use SDP knockoffs)
Xfull, original, knockoff = merge_knockoffs_with_original(X, Xko_sdp.X̃ )
knockoff_cv = glmnetcv(Xfull, y)
λbest = knockoff_cv.lambda[argmin(knockoff_cv.meanloss)]
βestim = glmnet(Xfull, y, lambda=[λbest]).betas[:, 1]

# target FDR is 0.05, 0.1, ..., 0.5
FDR = collect(0.05:0.05:0.5)
empirical_power = Float64[]
empirical_fdr = Float64[]
for fdr in FDR
    βknockoff = extract_beta(βestim, fdr, original, knockoff) # steps 2-3 happen here

    # compute power and false discovery proportion
    power = length(findall(!iszero, βknockoff) ∩ correct_position) / k
    fdp = length(setdiff(findall(!iszero, βknockoff), correct_position)) / max(count(!iszero, βknockoff), 1)
    push!(empirical_power, power)
    push!(empirical_fdr, fdp)
end

# visualize FDR and power
power_plot = plot(FDR, empirical_power, xlabel="Target FDR", ylabel="Empirical power", legend=false)
fdr_plot = plot(FDR, empirical_fdr, xlabel="Target FDR", ylabel="Empirical FDR", legend=false)
Plots.abline!(fdr_plot, 1, 0, line=:dash)
plot(power_plot, fdr_plot)
```




![png](output_16_0.png)



**Conclusion:** Compared to LASSO, knockoff's empirical FDR is controlled below the target FDR (dashed line). Controlled FDR is compensated by a small price in power. If this experiment is repeated multiple times, we expected the empirical FDR to hug the target (dashed) line more closely. 
