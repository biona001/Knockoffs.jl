# Call Julia code from R

This notebook demonstrates how to call Julia code in `R`. We will use [Knockoffs.jl](https://github.com/biona001/Knockoffs.jl) to generate [MVR/ME knockoffs](https://projecteuclid.org/journals/annals-of-statistics/volume-50/issue-1/Powerful-knockoffs-via-minimizing-reconstructability/10.1214/21-AOS2104.full) (Minimum Variance-based Reconstructability and Maximum Entropy) in `R`. 

Interfacing between R and Julia is accomplished by the `JuliaCall` package

+ JuliaCall package repo: https://github.com/Non-Contradiction/JuliaCall
+ JuliaCall documentation: https://cran.r-project.org/web/packages/JuliaCall/JuliaCall.pdf

The following code was tested on Sherlock with `julia/1.6.2` and `R/4.1.2`.

## Motivation and performance

The target application is ghost knockoffs. Chromosomes can be [partitioned into roughly independent blocks](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4731402/) of approximately 10000 SNPs. We will solve each block independently with `Knockoffs.jl`. 

According to rough benchmarks on 10000 by 10000 Toeplitz covariance matrices (i.e. timings should be taken with a grain of salt), the underlying convex optimization problem can be solved in 

+ ~15 min for ME (Maximum Entropy) knockoffs
+ ~44 min for MVR (Minimum Variance-based Reconstructability) knockoffs. 

The roughly 3x speed difference is caused by MVR knockoffs having to solve 3 systems of linear equations (which we dispatch to efficient LAPACK library) in each iterataion, while ME knockoffs need to solve only 1, suggesting that `Knockoffs.jl` has nearly optimal performance.

Note `Knockoffs.jl` is 9-22x faster than [knockpy](https://github.com/amspector100/knockpy) in my benchmarks, which is the python package that accompanies the original MVR/ME knockoff paper. 

## Installation

First install `JuliaCall` like any other R package 

```r
install.packages("JuliaCall", repos = "http://cran.us.r-project.org")
```
To use `JuliaCall` you must have a working installation of Julia. You can download the latest Julia version [here](https://julialang.org/downloads/), or on Sherlock, you can simply load Julia with 
```
module load julia/1.6.2
```
Within Julia, we need to install `RCall.jl`, `fastPHASE.jl` and `Knockoffs.jl` packages. Start Julia, and execute the following
```
using Pkg
pkg"add RCall"
pkg"add https://github.com/biona001/fastPHASE.jl"
pkg"add https://github.com/biona001/Knockoffs.jl"
```

**Note:** On Sherlock cluster I encountered `GLIBCXX_3.4.20 not found` error when I call `julia_setup()` later. To solve this, I added 
```
export R_LD_LIBRARY_PATH=/share/software/user/open/julia/1.6.2/lib/julia
```
in my `~/.bash_profile` file. Once we `source ~/.bash_profile` and restart R, the problem should be resolved. 

## Example: generate MVR/ME knockoffs with Knockoffs.jl 

Let's solve the following problem for the `s` vector using the Julia package `Knockoffs.jl`

```math
\begin{aligned}
min_{s} & \sum_j |1 - s_j|\\
\text{subject to } & 0 \le s_j \le 1\\
& 2\Sigma - diag(s) \succeq 0
\end{aligned}
```

In Knockoffs.jl, we need to call [solve_s](https://biona001.github.io/Knockoffs.jl/dev/man/api/#Knockoffs.solve_s) with `method=:maxent` or `method=:mvr`. These routines implement cyclic coordinate descent described in [this paper](https://projecteuclid.org/journals/annals-of-statistics/volume-50/issue-1/Powerful-knockoffs-via-minimizing-reconstructability/10.1214/21-AOS2104.full). Empirically, they can solve this problem with 10000 variables within 1 hour.

First load `JuliaCall` in R:


```r
# load JuliaCall package
library(JuliaCall)

# tell JuliaCall where is the Julia executable (the path can be found by typing `which julia` on terminal)
julia <- julia_setup(JULIA_HOME = "/share/software/user/open/julia/1.6.2/bin")
```

Next, within R, load Knockoffs.jl package in Julia


```r
# load Knockoffs.jl package
julia_library("Knockoffs")
```

```r
# test that `solve_s` function exists
julia_exists("solve_s")
```

```
## [1] TRUE
```
Finally, here is a simple wrapper function that calls Julia code. `solve_ME_s_with_julia` will solve the ME (maximum entropy) knockoffs using a convergence tolerance of `0.000001`. 


```r
solve_ME_s_with_julia <- function(Sigma) {
  # pass Sigma from R to Julia
  julia_assign("Sigma", Sigma)
  
  # solve for s vector in Julia (note: one must wrap `Symmetric()` keyword around Sigma)
  julia_command("s = solve_s(Symmetric(Sigma), :maxent, tol=0.000001);")

  # put s vector from Julia back to R
  s <- julia_eval("s")
  
  # return s vector
  return(s)
}
```

Let's simulate a covariance matrix and solve it


```r
# simulate 1000 by 1000 toeplitz covariance matrix in R
p <- 1000
Sigma <- toeplitz(0.4^(0:(p-1)))

# call Julia solver
s <- solve_ME_s_with_julia(Sigma)

# print a few values of s
s[1:10]
```

```
##  [1] 0.7606070 0.5997949 0.6114029 0.6105389 0.6106031 0.6105983 0.6105987
##  [8] 0.6105986 0.6105987 0.6105987
```

The first call to `solve_ME_s_with_julia` took ~30 seconds, while the second call took ~7 seconds. Because Julia uses JIT (just-in-time) compilation, the first call to `solve_s` will be slower in a fresh Julia session. For larger problems (e.g. 10000 dimensional $\Sigma$), compilation time becomes negligible. 