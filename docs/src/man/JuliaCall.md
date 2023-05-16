---
title: "Call Julia code from R"
output: 
  html_document:
    keep_md: true
---

# Calling from R

This notebook demonstrates how to call Julia code in `R`. We will use [Knockoffs.jl](https://github.com/biona001/Knockoffs.jl) to generate [MVR/ME knockoffs](https://projecteuclid.org/journals/annals-of-statistics/volume-50/issue-1/Powerful-knockoffs-via-minimizing-reconstructability/10.1214/21-AOS2104.full) (Minimum Variance-based Reconstructability and Maximum Entropy) in `R`. 

Interfacing between R and Julia is accomplished by the `JuliaCall` package

+ JuliaCall package repo: https://github.com/Non-Contradiction/JuliaCall
+ JuliaCall documentation: https://cran.r-project.org/web/packages/JuliaCall/JuliaCall.pdf

The following code was tested on Sherlock with `julia/1.6.2` and `R/4.1.2`.

## Installation

First install `JuliaCall` like any other R package 

```r
install.packages("JuliaCall", repos = "http://cran.us.r-project.org")
```

```
## 
## The downloaded binary packages are in
## 	/var/folders/vr/7w477ygd2513yzklk5pzsk1r0000gn/T//Rtmp3EMwdk/downloaded_packages
```
To use `JuliaCall` you must have a working installation of Julia. You can download the latest Julia version [here](https://julialang.org/downloads/), or a typical cluster can load Julia with
```
module load julia/1.6.2
```
Within Julia, we need to install `RCall.jl` and `Knockoffs.jl` packages. Start Julia, and execute the following
```
using Pkg
pkg"add RCall"
pkg"add https://github.com/biona001/Knockoffs.jl"
```

**Note:** On Stanford cluster, I encountered `GLIBCXX_3.4.20 not found` error when I call `julia_setup()` later. To solve this, I added 
```
export R_LD_LIBRARY_PATH=/share/software/user/open/julia/1.6.2/lib/julia
```
in my `~/.bash_profile` file. Once we `source ~/.bash_profile` and restart R, the problem should be resolved. 

To check that installation worked, try loading `JuliaCall` in R:


```r
# load JuliaCall package
library(JuliaCall)

# tell JuliaCall where is the Julia executable
julia <- julia_setup(JULIA_HOME = "/Applications/Julia-1.8.app/Contents/Resources/julia/bin")
```

Next, within R, also try loading Knockoffs.jl package


```r
# load Knockoffs.jl package
julia_library("Knockoffs")
```

## Examples

Below are a few examples of using `Knockoffs.jl` in R. Obviously, these code can be modified to suit your need. Note that the first call to any Julia function will be slower than subsequent call, because Julia uses JIT (just-in-time) compilation. For larger problems, compilation time becomes negligible. 

### Wrapper 1: generating knockoffs

Finally, `generate_group_ko_with_julia` is a wrapper that will solve for group knockoffs using the specified method. 

1. `X` is the original design matrix
2. `Sigma` is covariance matrix
3. `mu` is column mean
4. `method` can be `maxent`, `sdp`, `mvr`, or `equi`
5. `m` is the number of knockoffs to generate per sample (see [multiple knockoffs paper](http://proceedings.mlr.press/v89/gimenez19b.html)). 
6. `groups`: Vector of group membership

To generate second order knockoffs, simply do not specify `Sigma` and `mu`. 


```r
generate_group_ko_with_julia <- function(X, Sigma, mu, method, m, groups) {
  # pass variables from R to Julia
  julia_assign("X", X)
  julia_assign("Sigma", Sigma)
  julia_assign("mu", mu)
  julia_assign("method", method)
  julia_assign("m", m)
  julia_assign("groups", groups)
  
  # with Julia, solve group knockoffs problem
  julia_command("result = modelX_gaussian_group_knockoffs(X, Symbol(method), Int.(groups), mu, Sigma, m=Int(m), verbose=true)")

  # pull variables from Julia back to R
  Xko <- julia_eval("result.X̃ ") # the knockoffs
  S <- julia_eval("result.S") # the S matrix
  obj <- julia_eval("result.obj") # final objective value

  # return
  result <- list("Xko"=Xko, "S"=S, "obj"=obj)
  return (result)
}
```

### Wrapper 2: running lasso

If one wishes to run Lasso, we have a convenient function [fit_lasso](https://github.com/biona001/Knockoffs.jl/blob/group_MVR_ME/src/fit_lasso.jl#L28). By default, we run with 5 target FDR levels `0.01, 0.05, 0.1, 0.25, 0.5`, so there are 5 sets of selected variables.


```r
run_group_ko_lasso_with_julia <- function(y, X, Sigma, mu, method, m, groups) {
  # pass variables from R to Julia
  julia_assign("y", y)
  julia_assign("X", X)
  julia_assign("Sigma", Sigma)
  julia_assign("mu", mu)
  julia_assign("method", method)
  julia_assign("m", m)
  julia_assign("groups", groups)

  # with Julia, generate group knockoffs then solve lasso
  julia_command("result = fit_lasso(vec(y), X, mu, Sigma, groups=Int.(groups), method=Symbol(method), m=Int(m))")

  # pull variables from Julia back to R
  selected <- julia_eval("result.selected") # selected variables
  Xko <- julia_eval("result.ko.X̃ ") # the knockoffs
  S <- julia_eval("result.ko.S") # the S matrix
  obj <- julia_eval("result.ko.obj") # final objective value

  # return
  result <- list("selected"=selected, "Xko"=Xko, "S"=S, "obj"=obj)
  return (result)
}
```

### Wrapper 3: only solve for S matrix

Here's a function that only solves the Knockoff optimization problem and returns the S matrix, given a covariance matrix. 

1. `Sigma` is covariance matrix
2. `groups` is a group membership vector
4. `method` can be `maxent`, `sdp`, `mvr`, or `equi`
5. `m` is the number of knockoffs to generate per sample. 


```r
solve_S_with_julia <- function(Sigma, groups, method, m) {
  # pass Sigma from R to Julia
  julia_assign("Sigma", Sigma)
  julia_assign("groups", groups)
  julia_assign("method", method)
  julia_assign("m", m)

  # solve for s vector in Julia (note: one must wrap `Symmetric()` keyword around Sigma)
  julia_command("S, _, obj = solve_s_group(Symmetric(Sigma), groups, Symbol(method), tol=0.01, m=Int(m))")

  # put S matrix from Julia back to R
  S <- julia_eval("S")

  # return s vector
  return(S)
}
```

## Example 1: Generating knockoff

Let's simulate a covariance matrix, then solve the group knockoff problem with maximum entropy coordinate descent


```r
# simulate 1000 by 1000 toeplitz covariance matrix in R
n <- 1000 # number of samples
p <- 1000 # number of covariates
m <- 1    # number of knockoffs to generate per feature
Sigma <- toeplitz(0.7^(0:(p-1)))
mu <- rep(0, p)
method <- "maxent"
X <- MASS::mvrnorm(n=n, mu=mu, Sigma=Sigma)

# variables 1-5 are in group 1, then variables 6-10 in group 2...etc
groups = rep(1:200, each = 5)

# call Julia solver
result <- generate_group_ko_with_julia(X, Sigma, mu, method, m, groups)
Xko <- result$Xko
S <- result$S
obj <- result$obj

# print first few entries of S
S[1:5, 1:5]
```

```
##           [,1]      [,2]      [,3]      [,4]      [,5]
## [1,] 0.8960803 0.5703981 0.3503640 0.2140139 0.0954538
## [2,] 0.5703981 0.8344664 0.5220040 0.3149892 0.1377268
## [3,] 0.3503640 0.5220040 0.7898848 0.4708099 0.1992458
## [4,] 0.2140139 0.3149892 0.4708099 0.7051883 0.2873474
## [5,] 0.0954538 0.1377268 0.1992458 0.2873474 0.4067004
```

## Example 2: Running Lasso

Let's simulate a covariance matrix, design matrix $X$, and response $y$, then solve the group knockoff problem with maximum entropy coordinate descent. To solve the Lasso problem, Knockoffs.jl internally calls Fortran code from the GLMNet library. 


```r
# simulate 1000 by 1000 toeplitz covariance matrix in R
n <- 1000 # sample size
p <- 1000 # number of covariates
k <- 10   # number of causal variables
m <- 1    # number of knockoffs to generate per feature
mu <- rep(0, p)
Sigma <- toeplitz(0.4^(0:(p-1)))
method <- "maxent"

# variables 1-5 are in group 1, then variables 6-10 in group 2...etc
groups = rep(1:200, each = 5)

# simulate X, y, beta
X <- MASS::mvrnorm(n=n, mu=mu, Sigma=Sigma)
beta <- sample(c(rnorm(k), rep(0, p-k)))
y <- X %*% beta + rnorm(n)

# call Julia solver
result <- run_group_ko_lasso_with_julia(y, X, Sigma, mu, method, m, groups)
selected <- result$selected
Xko <- result$Xko
S <- result$S
obj <- result$obj

# check selected (here selected has 5 vectors, corresponding to target FDR 0.01, 0.05, 0.1, 0.25, 0.5)
selected[3]
```

```
##  [1]  14  19  21  38  61  66 123 168 182 183 185
```

For sanity check, compare selection to the causal groups

```r
correct_groups <- groups[which(beta != 0)]
correct_groups
```

```
##  [1]  14  19  38  61  66 123 168 182 183 185
```
