
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

## Regular functions

```@docs
  threshold
  MK_statistics
  hc_partition_groups
  id_partition_groups
  choose_group_reps
  fit_lasso
  fit_marginal
  simulate_AR1
  shift_until_PSD!
  normalize_col!
  sample_DMC
  predict
```

## Wrapper functions for SHAPEIT HMM knockoffs

There functions will eventually be replaced by Julia wrappers that no longer require user inputs. 

```@docs
  rapid
```
