module MMDS
    # import other packages that your package needs
    using Flux, Optimisers
    using CSV, DataFrames
    using DifferentialEquations
    using NODEData, ChaoticNDETools

    # include source code files where the actual functions of your project are
    include("preprocessing.jl")
    include("AutoODE.jl")
    include("ST_SuEIR.jl")
    include("visualization.jl")
    # export some of the functions that the users can use directly
    export read_covid_data, get_covid_IRD, prepare_data
    export AbstractAutoODEModel, ChaoticNDE, loss_covid
    export f_ST_SuEIR, ST_SuEIR, create_ST_SuEIR_initial_conditions
    export plot_covid_data, plot_IRD

    function __init__() # OPTIONAL: this special function is always executed when the module is loaded
        nothing
    end
end
