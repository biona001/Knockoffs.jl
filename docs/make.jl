using Knockoffs
using Documenter

DocMeta.setdocmeta!(Knockoffs, :DocTestSetup, :(using Knockoffs); recursive=true)

makedocs(;
    modules=[Knockoffs],
    authors="Benjamin Chu <benchu99@hotmail.com> and contributors",
    repo="https://github.com/biona001/Knockoffs.jl/blob/{commit}{path}#{line}",
    sitename="Knockoffs.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://biona001.github.io/Knockoffs.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/biona001/Knockoffs.jl",
)
