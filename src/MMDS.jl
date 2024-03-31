module MMDS
    # import other packages that your package needs
    using Flux, Optimisers
    using CSV, DataFrames
    using DifferentialEquations
    using JLD

    # unregistered packages
    using NODEData              # https://github.com/maximilian-gelbrecht/NODEData.jl/tree/main
    using ChaoticNDETools       # https://github.com/maximilian-gelbrecht/ChaoticNDETools.jl/tree/main


    # include source code files where the actual functions of your project are
    include("preprocessing.jl")
    include("NeuralODE.jl")
    include("AutoODE.jl")
    include("ST_SuEIR.jl")
    include("visualization.jl")
    include("loss.jl")
    include("utilities.jl")

    # export some of the functions that the users can use directly
    export AbstractAutoODEModel, AbstractNeuralODEModel

    export read_covid_data, get_covid_IRD, prepare_data, read_adjacency
    export SimpleNeuralODE
    export ST_SuEIR, AutoODE_ST_SuEIR, wrap_model, ChaoticNDE
    export f_ST_SuEIR
    export plot_covid_data, plot_IRD, plot_sequence
    export loss_covid
    export split_sample, augment_sample, save_params, load_params, compute_MAE

    function __init__() # OPTIONAL: this special function is always executed when the module is loaded
        nothing
    end
end
