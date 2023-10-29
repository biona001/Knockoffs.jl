using Knockoffs
using Documenter

makedocs(
    sitename = "Knockoffs.jl",
    format = Documenter.HTML(size_threshold = nothing),
    modules = [Knockoffs],
    authors = "Benjamin Chu",
    clean = true,
    pages = [
        "Home" => "index.md",
        "Fixed-X Knockoffs" => "man/fixed/fixed.md",
        "Model-X Knockoffs" => "man/modelX/modelX.md",
        "Group Knockoffs" => "man/group.md",
        "KnockoffScreen Knockoffs" => "man/knockoffscreen/knockoffscreen.md",
        "Ghost Knockoffs" => "man/ghost_knockoffs.md",
        "HMM Knockoffs" => "man/hmm/hmm.md",
        "IPAD Knockoffs" => "man/ipad.md",
        "Calling from R/Python" => "man/JuliaCall.md",
        "API" => "man/api.md",
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo   = "github.com/biona001/Knockoffs.jl.git",
    target = "build",
    deps   = nothing,
    make   = nothing,
)
