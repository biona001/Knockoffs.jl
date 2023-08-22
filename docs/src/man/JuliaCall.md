---
title: "Call Julia code from R"
output: 
  html_document:
    keep_md: true
---

# Calling from R and/or Python

[Knockoffs.jl](https://github.com/biona001/Knockoffs.jl) can be called directly from `R` or Python via the packages

+ `knockoffsr`: [https://github.com/biona001/knockoffsr](https://github.com/biona001/knockoffsr)
+ `knockoffspy`: [https://github.com/biona001/knockoffspy](https://github.com/biona001/knockoffspy)

The name space is setup so that standard syntax of Julia translates directly over to the R/Python environment. There are 3 things to keep in mind:

+ All `Knockoffs.jl` commands are prefaced by `ko$` (in R) or `ko.` (in python)
+ For `R` users, all commands with a `!` are replaced with `_bang`, for example `solve_s!` becomes `solve_s_bang`.
+ All `Knockoffs.jl` functions that require a `Symmetric` matrix as inputs now accepts a regular matrix. However, for `hc_partition_groups` and `id_partition_groups`, one must provide an extra argument `isCovariance` to indicate whether the input data should be treated as a design matrix or a covariance matrix.

The first 2 points follows the practice of [diffeqr](https://github.com/SciML/diffeqr/tree/master).
