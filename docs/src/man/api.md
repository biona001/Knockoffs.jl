
# API

Here is a list of available function calls. A detailed description can be found below. 

## Index

```@index
Pages = ["api.md"]
```

## Generating knockoffs

```@docs
  fixed_knockoffs
  modelX_gaussian_knockoffs
  modelX_gaussian_group_knockoffs
  modelX_gaussian_rep_group_knockoffs
  approx_modelX_gaussian_knockoffs
  hmm_knockoff
  full_knockoffscreen
  ghost_knockoffs
  ipad
  solve_s
  solve_s_group
  solve_s_graphical_group
```

## Regular knockoffs Solvers

```@docs
  solve_equi
  solve_max_entropy
  solve_MVR
  solve_sdp_ccd
  solve_SDP
```

## Group knockoffs Solvers

```@docs
  solve_group_equi
  solve_group_max_entropy_hybrid
  solve_group_mvr_hybrid
  solve_group_sdp_hybrid
```

## Other functions

```@docs
  threshold
  MK_statistics
  hc_partition_groups
  id_partition_groups
  choose_group_reps
  fit_lasso
  fit_marginal
  simulate_AR1
  simulate_ER
  simulate_block_covariance
  normalize_col!
  rapid
```

```@autodocs
Modules = [Knockoffs]
Order   = [:function, :type]
```
