# usage: 
# python hdf_to_vcf.py tree_CEU_GWAS_nofilt.hdf > CEU.vcf

import numpy as np
import pandas as pd
import msprime
import h5py
import gzip
import os
import sys

filename = sys.argv[1]
mytree = msprime.load(filename)

# print results (2 is for diploid, "legacy" = no matching positions)
with sys.stdout as vcffile:
    mytree.write_vcf(vcffile, 2)
