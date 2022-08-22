
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
  approx_modelX_gaussian_knockoffs
  modelX_gaussian_group_knockoffs
  hmm_knockoff
  full_knockoffscreen
  ghost_knockoffs
  solve_s
  solve_s_group
```

## Regular functions

```@docs
  coefficient_diff
  threshold
  extract_beta
  compare_correlation
  compare_pairwise_correlation
  merge_knockoffs_with_original
  simulate_AR1
  shift_until_PSD!
  normalize_col!
  decorrelate_knockoffs
  sample_DMC
  fit_lasso
  predict
```

## Wrapper functions for SHAPEIT HMM knockoffs

There functions will eventually be replaced by Julia wrappers that no longer require user inputs. 

```@docs
  rapid
```
