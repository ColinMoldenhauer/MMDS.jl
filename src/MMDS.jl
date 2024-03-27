module MMDS
    # import other packages that your package needs
    using Flux, Optimisers
    using CSV, DataFrames
    using DifferentialEquations

    # unregistered packages
    using NODEData              # https://github.com/maximilian-gelbrecht/NODEData.jl/tree/main
    using ChaoticNDETools       # https://github.com/maximilian-gelbrecht/ChaoticNDETools.jl/tree/main


    # define common dependencies here
    abstract type AbstractAutoODEModel end

    # include source code files where the actual functions of your project are
    include("preprocessing.jl")
    include("AutoODE.jl")
    include("ST_SuEIR.jl")
    include("NeuralODE.jl")
    include("visualization.jl")
    include("utilities.jl")

    # export some of the functions that the users can use directly
    export read_covid_data, get_covid_IRD, prepare_data
    export AbstractAutoODEModel, ChaoticNDE, loss_covid, AutoODE_ST_SuEIR
    export f_ST_SuEIR, ST_SuEIR, create_ST_SuEIR_initial_conditions

    export plot_covid_data, plot_IRD
    export split_sample, augment_sample

    function __init__() # OPTIONAL: this special function is always executed when the module is loaded
        nothing
    end
end
