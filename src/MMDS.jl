module MMDS
    # import other packages that your package needs
    using CSV, DataFrames

    # include source code files where the actual functions of your project are 
    include("preprocessing.jl")
    include("visualization.jl")
    
    # export some of the functions that the users can use directly
    export read_covid_data, get_covid_CRD

    function __init__() # OPTIONAL: this special function is always executed when the module is loaded 
        nothing 
    end
end
