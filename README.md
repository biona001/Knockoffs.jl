# Variable Selection with Knockoffs

| **Documentation** | **Build Status** | **Code Coverage**  |
|-------------------|------------------|--------------------|
| [![](https://img.shields.io/badge/docs-latest-blue.svg)](https://biona001.github.io/Knockoffs.jl/dev/) [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://biona001.github.io/Knockoffs.jl/stable/) | [![build Actions Status](https://github.com/biona001/Knockoffs.jl/workflows/CI/badge.svg)](https://github.com/biona001/Knockoffs.jl/actions) [![CI (Julia nightly)](https://github.com/biona001/Knockoffs.jl/workflows/JuliaNightly/badge.svg)](https://github.com/biona001/Knockoffs.jl/actions/workflows/JuliaNightly.yml) | [![codecov](https://codecov.io/gh/biona001/Knockoffs.jl/branch/master/graph/badge.svg?token=YyPqiFpIM1)](https://codecov.io/gh/biona001/Knockoffs.jl) |

This is a Julia implementation of the [knockoff filter](https://web.stanford.edu/group/candes/knockoffs/), taking many inspirations from the MATLAB/R implementation of [knockoff-filter](https://github.com/msesia/knockoff-filter), python implementation of [knockpy](https://github.com/amspector100/knockpy), and also the C++/R code of [knockoffgwas](https://github.com/msesia/knockoffgwas). The knockoff filter is a general framework for controlling the false discovery rate when performing variable selection. As the name suggests, the knockoff filter operates by manufacturing knockoff variables that are cheap — their construction does not require collecting any new data — and are designed to mimic the correlation structure found within the original variables. The knockoffs serve as negative controls and they allow one to identify the truly important predictors, while controlling the false discovery rate (FDR) — the expected fraction of false discoveries among all discoveries.

For more information, please see the [main webpage](https://web.stanford.edu/group/candes/knockoffs/)

## Installation

Download and install [Julia](https://julialang.org/downloads/). Within Julia, copy and paste the following: 
```julia
using Pkg
Pkg.add("Knockoffs")
```
This package supports Julia `v1.8`+. 

## Calling Knockoffs.jl from R and Python

See documentation for usage from R. If many users request of it, I will write a similar tutorial for calling Knockoffs.jl from your favorite language. 

## Package Features

+ Fast coordinate descent algorithms for MVR, ME, and SDP model-X knockoffs
+ Grouped MVR/ME/SDP knockoffs for improved power when there are highly correlated features. We also provide a representative group knockoff approach, based on graphical models, which is much more computationally efficient and empirically has superior power. 
+ Preliminary support for many other kinds of knockoffs (ghost, HMM, IPAD...etc), see documentation. 
+ Built-in functions to compute feature importance scores via Lasso/marginal regressions

## Bug reports and feature requests

Please open an issue if you find a bug or have feature requests. Feature requests are welcomed!

If you want to make contributions to this package, you should follow this workflow:

1. Fork this repository
2. Make a new branch on your fork, named after whatever changes you'll be making
3. Apply your code changes to the branch on your fork
4. When you're done, submit a PR to `Knockoffs.jl` to merge your fork into master branch.
