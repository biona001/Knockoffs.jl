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
pkg"add https://github.com/biona001/fastPHASE.jl" # this currently supports only mac and windows
pkg"add https://github.com/biona001/Knockoffs.jl"
```
This package supports Julia `v1.6`+. 

## Examples

```Julia
# simulate random matrix, then normalize columns
using Knockoffs
X = randn(1000, 200)
standardize!(X)

# fixed equi and SDP knockoffs
Aequi = fixed_knockoffs(X, :equi)
Asdp  = fixed_knockoffs(X, :sdp)

# model-X Gaussian knockoffs
X = randn(200, 400)
μtrue = zeros(400)
Aequi = modelX_gaussian_knockoffs(X, :equi, μtrue)
Asdp  = modelX_gaussian_knockoffs(X, :sdp, μtrue)
```

## Features

+ Fixed equi-correlated knockoffs
+ Fixed SDP knockoffs
+ Multivariate normal knockoffs based on conditional formulas
+ Discrete Markov chain knockoffs

## Development Roadmap

+ fastPHASE HMM knockoffs (in progress) (note: this is a native Julia implementation)
+ SHAPEIT HMM knockoffs (wrap code from [knockoffgwas/snpknock2](https://github.com/msesia/knockoffgwas)) (in progress)
+ MRC minimizing knockoffs ([ref](https://arxiv.org/abs/2011.14625))
+ Parallelized ASDP knockoffs
+ [KnockoffScreen](https://www.nature.com/articles/s41467-021-22889-4) knockoffs (in progress)
