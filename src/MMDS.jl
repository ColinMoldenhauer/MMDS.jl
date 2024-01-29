module MMDS
    using StatsBase # import other packages that your package needs 

    include("functions.jl") # include source code files where the actual functions of your project are 
    
    export add_two # export some of the functions that the users can use directly

    function __init__() # OPTIONAL: this special function is always executed when the module is loaded 
        nothing 
    end
end
