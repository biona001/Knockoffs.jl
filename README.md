# Variable Selection with Knockoffs

| **Documentation** | **Build Status** | **Code Coverage**  |
|-------------------|------------------|--------------------|
| [![](https://img.shields.io/badge/docs-latest-blue.svg)](https://biona001.github.io/Knockoffs.jl/dev/) [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://biona001.github.io/Knockoffs.jl/stable/) | [![build Actions Status](https://github.com/biona001/Knockoffs.jl/workflows/CI/badge.svg)](https://github.com/biona001/Knockoffs.jl/actions) [![CI (Julia nightly)](https://github.com/biona001/Knockoffs.jl/workflows/JuliaNightly/badge.svg)](https://github.com/biona001/Knockoffs.jl/actions/workflows/JuliaNightly.yml) | [![codecov](https://codecov.io/gh/biona001/Knockoffs.jl/branch/master/graph/badge.svg?token=YyPqiFpIM1)](https://codecov.io/gh/biona001/Knockoffs.jl) |

**This software is experimental in nature and should NOT be considered production ready**

This is a Julia implementation of the [knockoff filter](https://web.stanford.edu/group/candes/knockoffs/), taking many inspirations from the MATLAB/R implementation of [knockoff-filter](https://github.com/msesia/knockoff-filter), python implementation of [knockpy](https://github.com/amspector100/knockpy), and also the C++/R code of [knockoffgwas](https://github.com/msesia/knockoffgwas). The knockoff filter is a general framework for controlling the false discovery rate when performing variable selection. As the name suggests, the knockoff filter operates by manufacturing knockoff variables that are cheap — their construction does not require collecting any new data — and are designed to mimic the correlation structure found within the original variables. The knockoffs serve as negative controls and they allow one to identify the truly important predictors, while controlling the false discovery rate (FDR) — the expected fraction of false discoveries among all discoveries.

For more information, please see the [main webpage](https://web.stanford.edu/group/candes/knockoffs/)

## Installation

Download and install [Julia](https://julialang.org/downloads/). Within Julia, copy and paste the following: 
```julia
using Pkg
pkg"add https://github.com/biona001/fastPHASE.jl"
pkg"add https://github.com/biona001/Knockoffs.jl"
```
This package supports Julia `v1.6`+. 

**fastPHASE.jl supports only mac and linux systems.** We are also experiencing some comptability issues with mac's M1 CPUs. Please file an issue if installation is a problem. 

## Development Roadmap

+ Use Mosek for SDP problems by default
+ Docker
+ [Ghost knockoffs](https://www.biorxiv.org/content/10.1101/2021.12.06.471440v1.full) (in progress)
+ SDP/MVR/...etc constructions for [group knockoffs](https://proceedings.mlr.press/v48/daia16.html)
+ fastPHASE HMM knockoffs (in progress) (note: this is a native Julia implementation)
+ [KnockoffScreen](https://www.nature.com/articles/s41467-021-22889-4) knockoffs (in progress)
+ SHAPEIT HMM knockoffs (wrap code from [knockoffgwas/snpknock2](https://github.com/msesia/knockoffgwas)) (in progress)
