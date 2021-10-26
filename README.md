# Variable Selection with Knockoffs

| **Documentation** | **Build Status** | **Code Coverage**  |
|-------------------|------------------|--------------------|
| [![](https://img.shields.io/badge/docs-latest-blue.svg)](https://biona001.github.io/Knockoffs.jl/dev/) [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://biona001.github.io/Knockoffs.jl/stable/) | [![build Actions Status](https://github.com/biona001/Knockoffs.jl/workflows/CI/badge.svg)](https://github.com/biona001/Knockoffs.jl/actions) [![CI (Julia nightly)](https://github.com/biona001/Knockoffs.jl/workflows/JuliaNightly/badge.svg)](https://github.com/biona001/Knockoffs.jl/actions/workflows/JuliaNightly.yml) | [![codecov](https://codecov.io/gh/biona001/Knockoffs.jl/branch/master/graph/badge.svg?token=YyPqiFpIM1)](https://codecov.io/gh/biona001/Knockoffs.jl) |

**This software is experimental in nature and should NOT be considered production ready**

This is a Julia implementation of the [knockoff filter](https://web.stanford.edu/group/candes/knockoffs/), taking many inspirations from the MATLAB/R implementation of [knockoff-filter](https://github.com/msesia/knockoff-filter) and also the C++/R code of [knockoffgwas](https://github.com/msesia/knockoffgwas). The knockoff filter is a general framework for controlling the false discovery rate when performing variable selection. As the name suggests, the knockoff filter operates by manufacturing knockoff variables that are cheap — their construction does not require collecting any new data — and are designed to mimic the correlation structure found within the original variables. The knockoffs serve as negative controls and they allow one to identify the truly important predictors, while controlling the false discovery rate (FDR) — the expected fraction of false discoveries among all discoveries.

For more information, please see the [main webpage](https://web.stanford.edu/group/candes/knockoffs/)

## Installation

Download and install [Julia](https://julialang.org/downloads/). Within Julia, copy and paste the following: 
```julia
using Pkg
pkg"add https://github.com/biona001/Knockoffs.jl"
```
This package supports Julia `v1.6`+. 

## Design principle

A `knockoff` is essentially a `n*2p` `AbstractMatrix` with custom defined operations. Internally, a `knockoff` stores the original design matrix and its knockoff separately in memory, in addition to a few other variables. You can plug a `knockoff` into any functions that supports `AbstractMatrix` as inputs (e.g. a LASSO solver) and internally, linear algebra will automatically be dispatched to relvant BLAS functions. 

```Julia
# simulate random matrix, then normalize columns
using Knockoffs
X = randn(1000, 200)
normalize_col!(X)

# fixed equi and SDP knockoffs
Aequi = fixed_knockoffs(X, :equi)
Asdp  = fixed_knockoffs(X, :sdp)

# model-X Gaussian knockoffs
X = randn(200, 400)
μtrue = zeros(400)
Aequi = modelX_gaussian_knockoffs(X, :equi, μtrue)
Asdp  = modelX_gaussian_knockoffs(X, :sdp, μtrue)
```

## Development Roadmap

+ Fixed equi-correlated knockoffs (done)
+ Fixed SDP knockoffs (done)
+ Multivariate normal knockoffs based on conditional formulas (done)
+ Parallelized ASDP knockoffs
+ Markov chain knockoffs (work in progress)
+ HMM knockoffs (wrap code from [knockoffgwas/snpknock2](https://github.com/msesia/knockoffgwas)) (done)
+ MRC minimizing knockoffs ([ref](https://arxiv.org/abs/2011.14625))
+ Threshold functions (done)
+ Example with lasso path
+ Example with IHT path
+ Compare to [existing implementations](https://github.com/msesia/knockoff-filter)
