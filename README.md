# Variable Selection with Knockoffs

| **Documentation** | **Build Status** | **Code Coverage**  |
|-------------------|------------------|--------------------|
| [![](https://img.shields.io/badge/docs-latest-blue.svg)](https://biona001.github.io/Knockoffs.jl/dev/) [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://biona001.github.io/Knockoffs.jl/stable/) | [![build Actions Status](https://github.com/biona001/Knockoffs.jl/workflows/CI/badge.svg)](https://github.com/biona001/Knockoffs.jl/actions) [![CI (Julia nightly)](https://github.com/biona001/Knockoffs.jl/workflows/JuliaNightly/badge.svg)](https://github.com/biona001/Knockoffs.jl/actions/workflows/JuliaNightly.yml) | [![codecov](https://codecov.io/gh/biona001/Knockoffs.jl/branch/master/graph/badge.svg?token=YyPqiFpIM1)](https://codecov.io/gh/biona001/Knockoffs.jl) |

**This software is experimental in nature and should NOT be considered production ready**

This is a Julia implementation of the [knockoff filter](https://projecteuclid.org/journals/annals-of-statistics/volume-43/issue-5/Controlling-the-false-discovery-rate-via-knockoffs/10.1214/15-AOS1337.full), taking many inspirations from the [MATLAB and R implementations](https://github.com/msesia/knockoff-filter). The knockoff filter is a general framework for controlling the false discovery rate when performing variable selection. As the name suggests, the knockoff filter operates by manufacturing knockoff variables that are cheap — their construction does not require collecting any new data — and are designed to mimic the correlation structure found within the original variables. The knockoffs serve as negative controls and they allow one to identify the truly important predictors, while controlling the false discovery rate (FDR) — the expected fraction of false discoveries among all discoveries.

For more information, please see the [main webpage](https://web.stanford.edu/group/candes/knockoffs/)

## Installation

Download and install [Julia](https://julialang.org/downloads/). Within Julia, copy and paste the following: 
```julia
using Pkg
pkg"add https://github.com/biona001/Knockoffs.jl"
```
This package supports Julia `v1.6`+. 

## Design principle

The `knockoff` matrix is an `AbstractMatrix` with custom-defined operations so that it behaves like a matrix. You can plug a `knockoff` into any functions that supports `AbstractMatrix` as inputs (e.g. a LASSO solver) and it will be fast. 

Internally, a `knockoff` stores the original design matrix and its knockoff separately in memory, in addition to other variables such as the `s` vector. 

```Julia
# simulate random matrix, then normalize columns
using Knockoffs
X = randn(1000, 200)
normalize_col!(X)

# fixed equi and SDP knockoffs
Aequi = fixed_knockoffs(X, method=:equi)
Asdp  = fixed_knockoffs(X, method=:sdp)
```

## Development Roadmap

+ Multivariate normal knockoffs based on conditional formulas
+ Parallelized ASDP knockoffs
+ HMM knockoffs? Might be hard.
+ Threshold functions
+ Example with lasso path
+ Example with IHT path
+ Compare to [existing implementations](https://github.com/msesia/knockoff-filter)
