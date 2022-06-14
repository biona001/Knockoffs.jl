
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
  hmm_knockoff
  full_knockoffscreen
  solve_s
  solve_MVR
  solve_max_entropy
  solve_sdp_fast
  solve_SDP
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
```

## Wrapper functions for SHAPEIT HMM knockoffs

There functions will eventually be replaced by Julia wrappers that no longer require user inputs. 

```@docs
  rapid
```
