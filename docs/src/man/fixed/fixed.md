
# Fixed-X knockoffs

This tutorial generates fixed-X knockoffs and checks some of its basic properties. The methodology is described in the following paper

> Barber, Rina Foygel, and Emmanuel J. Candès. "Controlling the false discovery rate via knockoffs." The Annals of Statistics 43.5 (2015): 2055-2085.


!!! note
    For fixed-X knockoffs, we assume $n > 2p$ where $n$ is sample size and $p$ is number of covariates, although in principle this method can be adapted to work for $n > p$ case.


```julia
# load packages needed for this tutorial
using Revise
using Knockoffs
using Plots
using Random
using GLMNet
using LinearAlgebra
using Distributions
gr(fmt=:png);
```

    ┌ Info: Precompiling Knockoffs [878bf26d-0c49-448a-9df5-b057c815d613]
    └ @ Base loading.jl:1423


## Generate knockoffs

We will

1. Simulate Gaussian design matrix
2. Generate knockoffs

Both equi-correlated and SDP knockoffs are supported. 


```julia
Random.seed!(2022)   # set random seed for reproducibility
X = randn(1000, 200) # simulate Gaussian matrix

# make equi-correlated and SDP knockoffs
@time Aequi = fixed_knockoffs(X, :equi)
@time Asdp = fixed_knockoffs(X, :sdp);
```

      0.062395 seconds (80 allocations: 20.696 MiB)
      3.067441 seconds (509.89 k allocations: 417.777 MiB, 3.49% gc time)


The return type is a `Knockoff` struct, which contains the following fields

```julia
struct GaussianKnockoff{T} <: Knockoff
    X::Matrix{T} # n × p design matrix
    X̃::Matrix{T} # n × p knockoff of X
    s::Vector{T} # p × 1 vector. Diagonal(s) and 2Σ - Diagonal(s) are both psd
    Σ::Symmetric{T, Matrix{T}} # p × p covariance matrix
    method::Symbol # :sdp or :equi
end
```

Thus, to access these fields, one can do


```julia
X = Asdp.X
X̃ = Asdp.X̃
s = Asdp.s
Σ = Asdp.Σ;
```

We can check some knockoff properties. For instance, is it true that $X'\tilde{X} \approx \Sigma - diag(s)$?


```julia
# compare X'X and Σ-diag(s) visually
[vec(X'*X̃) vec(Σ - Diagonal(s))]
```




    40000×2 Matrix{Float64}:
      0.221993     0.221993
     -0.0184072   -0.0184072
     -0.0274339   -0.0274339
     -0.0359705   -0.0359705
      0.0202419    0.0202419
      0.0514162    0.0514162
      0.0154546    0.0154546
     -0.0250518   -0.0250518
     -0.0468651   -0.0468651
     -0.0358149   -0.0358149
     -0.00507354  -0.00507354
     -0.0238295   -0.0238295
      0.0140797    0.0140797
      ⋮           
      0.0192772    0.0192772
     -0.0157356   -0.0157356
     -0.0116071   -0.0116071
      0.0338745    0.0338745
      0.029913     0.029913
     -0.03115     -0.03115
      0.0437582    0.0437582
      0.00350153   0.00350153
      0.00382205   0.00382205
     -0.0072671   -0.0072671
      0.00966888   0.00966888
      0.444122     0.444122



## LASSO example

Let us apply the generated knockoffs to the model selection problem. In layman's term, it can be stated as

> Given response $\mathbf{y}_{n \times 1}$, design matrix $\mathbf{X}_{n \times p}$, we want to select a subset $S \subset \{1,...,p\}$ of variables that are truly causal for $\mathbf{y}$. 

### Simulate data

We will simulate 

$$\mathbf{y}_{n \times 1} \sim N(\mathbf{X}_{n \times p}\mathbf{\beta}_{p \times 1} \ , \ \mathbf{\epsilon}_{n \times 1}), \quad \epsilon_i \sim N(0, 0.5)$$

where $k=50$ positions of $\mathbf{\beta}$ is non-zero with effect size $\beta_j \sim N(0, 1)$. The goal is to recover those 50 positions using LASSO.


```julia
# set seed for reproducibility
Random.seed!(2022)

# simulate true beta
n, p = size(X)
k = 50
βtrue = zeros(p)
βtrue[1:k] .= 3randn(50)
shuffle!(βtrue)

# find true causal variables
correct_position = findall(!iszero, βtrue)

# simulate y using normalized X
y = X * βtrue + rand(Normal(0, 0.5), n);
```

### Standard LASSO

Lets try running standard LASSO, which will produce $\hat{\mathbf{\beta}}_{p \times 1}$ where we typically declare SNP $j$ to be selected if $\hat{\beta}_j \ne 0$. We use LASSO solver in [GLMNet.jl](https://github.com/JuliaStats/GLMNet.jl) package, which is just a Julia wrapper for the GLMnet Fortran code. 

How well does LASSO perform in terms of power and FDR?


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




    (0.86, 0.5222222222222223)



It seems LASSO have power 86%, but the false discovery rate is 52%. This means that although LASSO finds almost every predictor, more than half of all discoveries are false positives. 

### Knockoff+LASSO

Now lets try applying the knockoff methodology. Recall that consists of a few steps 

1. Run LASSO on $[\mathbf{X} \mathbf{\tilde{X}}]$
2. Compare feature importance score $W_j = \text{score}(x_j) - \text{score}(\tilde{x}_j)$ for each $j = 1,...,p$. Here we use $W_j = |\beta_j| - |\tilde{\beta}_{j}|$
3. Choose target FDR $q \in [0, 1]$ and compute 
$$\tau = min_{t}\left\{t > 0: \frac{{\{\#j: W_j ≤ -t}\}}{max(1, {\{\#j: W_j ≥ t}\})} \le q\right\}$$

!!! note
    
    In step 1, $[\mathbf{X} \mathbf{\tilde{X}}]$ is written for notational convenience. In practice one must interleave knockoffs with the original variables, where either the knockoff come first or the original genotype come first with equal probability. This is due to the inherent bias of LASSO solvers: when the original and knockoff variable are equally valid, the one listed first will be selected. 


```julia
@time knockoff_filter = fit_lasso(y, Asdp.X, Asdp.X̃);
```

      1.710676 seconds (4.78 k allocations: 138.989 MiB)


The return type is now a `KnockoffFilter`, which contains the following information

```julia
struct KnockoffFilter{T}
    XX̃ :: Matrix{T} # n × 2p matrix of original X and its knockoff interleaved randomly
    original :: Vector{Int} # p × 1 vector of indices of XX̃ that corresponds to X
    knockoff :: Vector{Int} # p × 1 vector of indices of XX̃ that corresponds to X̃
    W :: Vector{T} # p × 1 vector of feature-importance statistics for fdr level fdr
    βs :: Vector{Vector{T}} # βs[i] is the p × 1 vector of effect sizes corresponding to fdr level fdr_target[i]
    a0 :: Vector{T}   # intercepts for each model in βs
    τs :: Vector{T}   # knockoff threshold for selecting Ws correponding to each FDR
    fdr_target :: Vector{T} # target FDR level for each τs and βs
    debiased :: Bool # whether βs and a0 have been debiased
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




![png](output_15_0.png)



Observe that

+ LASSO + knockoffs controls the false discovery rate at below the target (dashed line)
+ The power of LASSO + knockoffs is lower than standard LASSO

If we repeated the simulation multiple times, we expect the empirical FDR to hug the target FDR more closely.
