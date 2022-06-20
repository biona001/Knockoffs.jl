using Knockoffs
using Documenter

makedocs(
    sitename = "Knockoffs.jl",
    format = Documenter.HTML(),
    modules = [Knockoffs],
    pages = [
        "Home" => "index.md",
        "Fixed-X Knockoffs" => "man/fixed/fixed.md",
        "Model-X Knockoffs" => "man/modelX/modelX.md",
        "fastPHASE HMM Knockoffs" => "man/fastphase_hmm/fastphase_hmm.md",
        "SHAPEIT HMM Knockoffs" => "man/shapeit_hmm.md",
        "KnockoffScreen Knockoffs" => "man/knockoffscreen/knockoffscreen.md",
        "Ghost Knockoffs" => "man/ghost_knockoffs.md",
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
