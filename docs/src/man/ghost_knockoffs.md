# Ghost Knockoffs

This tutorial is for generating Ghost (summary statistics) Knockoffs for analyzing genome-wide association studies (GWAS). The methodology is described in 

> He, Z., Liu, L., Belloy, M. E., Le Guen, Y., Sossin, A., Liu, X., ... & Ionita-Laza, I. (2021). Summary statistics knockoff inference empowers identification of putative causal variants in genome-wide association studies. bioRxiv.

It is assumed we do not have access to individual level genotype data $\mathbf{G}$. Rather, for each SNP, we have the z-scores $Z_j$ with respect to a phenotype $\mathbf{Y}$ from a GWAS. Then we sample the knockoff z-scores as 
```math
\begin{aligned}
\mathbf{\tilde{Z}} | \mathbf{G}, \mathbf{Y} \sim N(\mathbf{P}\mathbf{Z}, \mathbf{V}),
\end{aligned}
```
```math
\begin{aligned}
\mathbf{P} = \mathbf{I} - \mathbf{D}\mathbf{\Sigma}^{-1}, \quad \mathbf{V} = 2\mathbf{D} - \mathbf{D}\mathbf{\Sigma}^{-1}\mathbf{D},
\end{aligned}
```
where $\mathbf{I}$ is a $p \times p$ identity matrix, $\mathbf{\Sigma}$ is the correlation matrix among genotypes (characterizing linkage disequilibrium), and $\mathbf{D} = diag(s_1, ..., s_p)$ is a diagonal matrix given by solving the following convex optimization problem
```math
\begin{aligned}
\text{minimize } & \sum_j | 1 - s_j |\\
\text{subject to } & 2\mathbf{\Sigma} - \mathbf{D} \succeq 0\\
                   & s_j \ge 0.
\end{aligned}
```
In summary, we need to
1. Estimate the correlation matrix $\mathbf{\Sigma}$, and 
2. Solve for the vector $\mathbf{s}$.

These operations define $\mathbf{P}$ and $\mathbf{V}$ which can be used to sample the knockoffs $\mathbf{\tilde{Z}}$. We describe step 1 in detail below. Step 2 can be accomplished via standard SDP or MVR solvers.


```julia
# load package needed for this tutorial
using Knockoffs
using VCFTools
using StatsBase
using LinearAlgebra
using SnpArrays
using Random
```

## Estimate correlation matrix $\mathbf{\Sigma}$

Recall the only data we have is the score statistic $\mathbf{Z}$ for each SNP (i.e. we do not have individual level data $\mathbf{G}$). To obtain an estimate of $\mathbf{\Sigma}$, one way is to leverage a reference haplotype panel, which we denote by $\mathbf{H}$. For instance, the [1000 genomes phase 3 panel](https://bochet.gcc.biostat.washington.edu/beagle/1000_Genomes_phase3_v5a/b37.vcf) is freely availble to the public and contains 2504 samples of diverse ancestry and ~50 million SNPs. Since $\mathbf{H}$ is very dense, each SNP in $\mathbf{Z}$ either exist in $\mathbf{H}$, or is close to one that is typed in $\mathbf{H}$. Over 90% of SNPs in the UK Biobank can be mapped to this panel. Thus, we can roughly estimate $\mathbf{\Sigma}$. 

Of course, population structure may skew the estimation. If the GWAS samples are rather homogeneous, one may want include only ancestrally similar samples in the reference panel. Of course, using denser and larger reference panels such as the [HRC](https://www.nature.com/articles/ng.3679) or [TOPMed](https://www.nature.com/articles/s41586-021-03205-y) will also improve approximation, but those panels has restricted public access. 

## Obtain reference panel

First download the [1000 genomes reference panel](https://bochet.gcc.biostat.washington.edu/beagle/1000_Genomes_phase3_v5a/b37.vcf) in standard VCF format (warning: requires $\sim 9.4$ GB):


```julia
download_1000genomes(outdir = "/scratch/users/bbchu")
```

Import chr22 data


```julia
vcffile = "/scratch/users/bbchu/1000genomes/chr22.1kg.phase3.v5a.vcf.gz"
H, H_sampleID, H_chr, H_pos, H_ids, H_ref, H_alt = convert_gt(Float32, 
    vcffile, save_snp_info=true, msg="importing");
```

    [32mimporting 100%|██████████████████████████████████████████| Time: 0:02:36[39m


Here `H` is the reference panel, each row is a sample and each column is a SNP. The type of H is `Union{Missing, Float32}`, which potentially allows for missing data. To conserve memory, one could specify `UInt8` instead of `Float32`. Of course, a good reference panel such as the 1000 genomes featured here will have no missing data. 


```julia
H
```




    2504×424147 Matrix{Union{Missing, Float32}}:
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  …  0.0  0.0  0.0  0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  …  0.0  0.0  0.0  0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  …  0.0  0.0  0.0  0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  1.0  0.0  0.0
     ⋮                        ⋮              ⋱  ⋮                        ⋮    
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  …  0.0  0.0  0.0  0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     1.0  0.0  0.0  0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  1.0  1.0  0.0  1.0  0.0  0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  …  1.0  0.0  0.0  0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0



Here `H_pos` contains the position for all SNPs present on chromosome 22 on the reference panel. 


```julia
H_pos
```




    424147-element Vector{Int64}:
     16050115
     16050213
     16050607
     16050739
     16050783
     16050840
     16050847
     16050922
     16050984
     16051075
     16051249
     16051453
     16051722
            ⋮
     51239651
     51239652
     51239678
     51239794
     51240084
     51240820
     51241101
     51241102
     51241285
     51241386
     51244163
     51244237



On the UK Biobank, 9359 / 9537 SNPs can be mapped to the 1000 genomes on chromosome 22. Note when this number is low, check if the human genome build for the reference panel (here build 37) matches with the GWAS data.


```julia
xdata = SnpData("/scratch/users/bbchu/ukb_fastPHASE/subset/ukb.10k.chr22")
xdata.snp_info[!, "position"] ∩ H_pos
```




    9359-element Vector{Int64}:
     16495833
     16870425
     16888577
     16952830
     17054795
     17056415
     17057597
     17068748
     17070109
     17072347
     17079911
     17080190
     17091201
            ⋮
     51162059
     51162850
     51163138
     51163910
     51165664
     51171497
     51173542
     51174939
     51175626
     51183255
     51185848
     51193629



## Simulated Example

Now suppose we were given the z-scores for 10000 SNPs and their position. Note that a GWAS summary stats file may not contain z-scores, but rather effect sizes, odds-ratios, or p-values. To convert different measures to the standard z-score, I found [this reference](https://huwenboshi.github.io/data%20management/2017/11/23/tips-for-formatting-gwas-summary-stats.html) to be very useful.  


```julia
Random.seed!(2022)
p = 10000 # number of SNPs
Z = randn(p) # simulated z-scores
Z_pos = sort!(rand(16050000:51244000, p)) # simulated position for each SNP
```




    10000-element Vector{Int64}:
     16051161
     16051964
     16058006
     16061965
     16062797
     16066970
     16070158
     16071335
     16076125
     16078197
     16078220
     16080831
     16082209
            ⋮
     51197030
     51203484
     51209420
     51211742
     51211790
     51215620
     51222784
     51227760
     51227831
     51231582
     51235924
     51238476



Generating ghost knockoffs is accomplished via the [ghost\_knockoffs](https://biona001.github.io/Knockoffs.jl/dev/man/api/#Knockoffs.ghost_knockoffs) function. Here we employ the MVR construction (which tends to have the highest power) with a moderate window size:


```julia
Z̃ = ghost_knockoffs(Z, Z_pos, H_pos, H, :mvr, windowsize=500)
```

    [32mApproximating covariance by blocks... 100%|██████████████| Time: 0:00:32[39m





    10000-element Vector{Float64}:
      0.5410315696729918
     -0.4246081059629637
     -0.04956798334091112
     -1.5764005812497706
      0.5281884350544112
     -0.02190687576925561
      0.5249259412485039
      1.0111906565388558
     -1.4466613758239868
      0.2581548620476279
      0.09210845466626205
     -0.4317124835746688
      1.0672703288267222
      ⋮
      0.45921376701255845
      1.202714589200995
      0.6726136982607442
     -0.22241142386162988
     -0.05067713325684163
     -0.6315313886344935
     -0.20778832692909632
      0.12586124372837243
      0.37169961989672984
     -0.1551860169036035
      1.7102480919504381
     -0.03192549553328507


