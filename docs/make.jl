using Knockoffs
using Documenter

makedocs(
    sitename = "Knockoffs.jl",
    format = Documenter.HTML(),
    modules = [Knockoffs],
    pages = [
        "Home" => "index.md",
        "HMM Knockoffs" => "man/hmm.md",
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
