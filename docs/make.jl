using MMDS
using Documenter

DocMeta.setdocmeta!(MMDS, :DocTestSetup, :(using MMDS); recursive=true)

makedocs(;
    modules=[MMDS],
    authors="Colin Moldenhauer <colin.moldenhauer@tum.de>",
    repo="https://github.com/ColinMoldenhauer/MMDS.jl/",
    sitename="MMDS.jl",
    format=Documenter.HTML(;
    prettyurls=get(ENV, "CI", "false") == "true",
    assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Functions" => "functions.md"
    ],
)
