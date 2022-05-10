# Knockoffs.jl

This is a Julia implementation of the [knockoff filter](https://web.stanford.edu/group/candes/knockoffs/). The knockoff filter is a general framework for controlling the false discovery rate when performing variable selection. As the name suggests, the knockoff filter operates by manufacturing knockoff variables that are cheap — their construction does not require collecting any new data — and are designed to mimic the correlation structure found within the original variables. The knockoffs serve as negative controls and they allow one to identify the truly important predictors, while controlling the false discovery rate (FDR) — the expected fraction of false discoveries among all discoveries.

## Package Features

## Installation

Within Julia,
```julia
using Pkg
pkg"add https://github.com/biona001/Knockoffs.jl"
```
This package supports Julia `v1.6`+.

## Manual Outline

```@contents
Pages = [
    "man/fixed/fixed.md",
    "man/modelX/modelX.md",
    "man/fastphase_hmm/fastphase_hmm.md",
    "man/shapeit_hmm.md",
    "man/knockoffscreen/knockoffscreen.md",
    "man/api.md"
]
Depth = 2
```
