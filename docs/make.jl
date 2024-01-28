using MMDS
using Documenter

DocMeta.setdocmeta!(MMDS, :DocTestSetup, :(using MMDS); recursive=true)

makedocs(;
    modules=[MMDS],
    authors="Colin Moldenhauer <colin.moldenhauer@tum.de>",
    sitename="MMDS.jl",
    format=Documenter.HTML(;
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
