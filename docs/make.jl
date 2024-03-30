using MMDS
using Documenter

DocMeta.setdocmeta!(MMDS, :DocTestSetup, :(using MMDS); recursive=true)

makedocs(;
    modules=[MMDS],
    authors="Colin Moldenhauer <colin.moldenhauer@tum.de>",
    repo="https://github.com/ColinMoldenhauer/MMDS.jl/blob/{commit}{path}#{line}",
    sitename="MMDS.jl",
    format=Documenter.HTML(;
    prettyurls=get(ENV, "CI", "false") == "true",
    assets=String[],
    ),
    pages=[
        "Overview" => "index.md",
        "Preprocessing" => "preprocessing.md",
        "ST-SuEIR" => "st_sueir.md",
        "AutoODE" => "autoode.md",
        "NeuralODE" => "neuralode.md",
        "Visualization" => "visualization.md",
        "Loss" => "loss.md",
        "Utils" => "utils.md",
    ],
)

deploydocs(
    repo = "github.com/ColinMoldenhauer/MMDS.jl.git",
)