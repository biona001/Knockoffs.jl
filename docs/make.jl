using Knockoffs
using Documenter

makedocs(
    sitename = "Knockoffs.jl",
    format = Documenter.HTML(),
    modules = [Knockoffs],
    pages = [
        "Home" => "index.md",
        "Fixed-X Knockoffs" => "man/fixed.md",
        # "Model-X HMM Knockoffs" => "man/modelx.md",
        # "MRC minimizing Knockoffs" => "man/mrc.md",
        "fastPHASE HMM Knockoffs" => "man/fastphase_hmm.md",
        "SHAPEIT HMM Knockoffs" => "man/shapeit_hmm.md",
        # "KnockoffScreen Knockoffs" => "man/fastphase_hmm.md",
        # "LASSO Example" => "man/lasso_example.md",
        "API" => "man/api.md",
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo   = "github.com/biona001/Knockoffs.jl.git",
    target = "build"
)
