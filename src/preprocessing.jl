using CSV, DataFrames
using JLD: load as jld_load


"""
    read_covid_data(data_dir::String; normalize::Union{String, Nothing}=nothing)

Read all CSV files contained in `data_dir` and concatenate them into a DataFrame.
Optionally normalize the data by state population.
"""
function read_covid_data(data_dir::String; normalize::Union{String, Nothing}=nothing)
    files = sort([f for f in readdir(data_dir, join=false) if endswith(f, "csv")], by = s -> prod(match(r"(\d{2})-(\d{2})-(\d{4})\.csv", s).captures[[3,1,2]]))
    dates = [match(r"(\d{2})-(\d{2})-(\d{4})", f).match for f in files]

    # determine US states
    df = CSV.read(joinpath(data_dir, files[1]), DataFrame)
    df_drop = filter(row -> row["Province_State"] ∉ ["Diamond Princess", "Grand Princess", "Recovered"], df)
    states = df_drop[:, "Province_State"]

    population = nothing
    if ~isnothing(normalize)
        df_pop = CSV.read(normalize, DataFrame)
        rename!(df_pop, Dict("State" => "Province_State", "2018 Population" => "Population"))
        df_join = outerjoin(df_pop, df_drop, on="Province_State")
        population = df_join[:, "Population"]
    end

    dfs = []
    for file in files
        df = CSV.read(joinpath(data_dir, file), DataFrame)
        try
            # filter out rows
            df_drop = filter(row -> row["Province_State"] ∉ ["Diamond Princess", "Grand Princess", "Recovered"], df)


            # select only relevant columns (use "Province_State" to join population dataframe)
            df_select = df_drop[:, ["Confirmed", "Recovered", "Deaths", "Province_State"]]

            # optionally normalize by state population
            if ~isnothing(normalize)
                df_join = select!(outerjoin(df_pop, df_select, on="Province_State"), Not("Province_State"))
                df_select = select!(mapcols(col -> col ./ df_join[:, "Population"] / 1e6, df_join), Not("Population"))
            end

            # fill missing values with 0
            df_repl = mapcols(col -> replace(col, missing => 0), df_select)

            # collect
            push!(dfs, df_repl)
        catch e
            println("Error in file '$file' ($e)")
        end
    end
    df_concat = hcat(dfs..., makeunique=true)

    return dates, states, population, df_concat
end


"""
    get_covid_IRD(data_dir::String; normalize::Union{String, Nothing}=nothing)

Extract infected cases, recovered cases and deaths in chronological order.
Returns data as matrices of size (N_states, N_epochs).
"""
function get_covid_IRD(data_dir::String; normalize::Union{String, Nothing}=nothing)
    dates, states, population, data = read_covid_data(data_dir, normalize=normalize)

    infected = data[!, r"Confirmed"]
    recovered = data[!, r"Recovered"]
    deaths = data[!, r"Deaths"]

    return dates, states, population, Matrix(infected), Matrix(recovered), Matrix(deaths)
end



"""
    prepare_data(I, R, D; N_states=nothing)

Prepare data for wrapping in `NODEDataloader` for later use in ST-SuEIR model.
Optionally choose a subset of the first N states.
"""
function prepare_data(I, R, D; N_states=nothing, N_ode=3)
    N_t = size(I, 2)
    t = 0:(N_t-1)

    N_states = isnothing(N_states) ? size(I, 1) : N_states

    # initialize all dimensions
    data_matrix = zeros(N_states, N_ode, N_t)     # TODO: type

    # populate with covid data
    for (i_data, data) in enumerate([I, R, D])
        data_matrix[:, end-3+i_data, :] = data[1:N_states, :]
    end
    return data_matrix, t
end


# auxiliary data for ST-SuEIR

"""
Stay-at-home data (dict `stayhome`) for US states and order of states (array `order_wang_et_al`) as specified by Wang et al. in
https://github.com/Rose-STL-Lab/AutoODE-DSL/blob/master/ode_nn/mobility/Mobility.py
"""
stayhome = Dict(
    "AK" => 29.3,
    "AL" => 23.8,
    "AR" => 24.2,
    "AZ" => 34.2,
    "CA" => 35.6,
    "CO" => 30.9,
    "CT" => 32.6,
    "DC" => 40.1,
    "DE" => 31.9,
    "FL" => 31.6,
    "GA" => 27.8,
    "HI" => 30.3,
    "IA" => 25.7,
    "ID" => 29.1,
    "IL" => 30.6,
    "IN" => 27.4,
    "KS" => 26.4,
    "KY" => 26.3,
    "LA" => 25.2,
    "MA" => 34.7,
    "MD" => 34.6,
    "ME" => 30.5,
    "MI" => 28.3,
    "MN" => 30.3,
    "MO" => 26.5,
    "MS" => 23.0,
    "MT" => 28.8,
    "NC" => 28.5,
    "ND" => 26.4,
    "NE" => 26.1,
    "NH" => 31.4,
    "NJ" => 33.3,
    "NM" => 31.7,
    "NV" => 33.8,
    "NY" => 35.4,
    "OH" => 27.7,
    "OK" => 23.8,
    "OR" => 33.1,
    "PA" => 30.8,
    "RI" => 32.7,
    "SC" => 26.6,
    "SD" => 26.1,
    "TN" => 26.5,
    "TX" => 31.0,
    "UT" => 30.2,
    "VA" => 31.7,
    "VT" => 32.2,
    "WA" => 34.2,
    "WI" => 28.8,
    "WV" => 26.7,
    "WY" => 28.6
)


"""
    order_wang_et_al

Order of states as specified by Wang et al.
"""
order_wang_et_al = [
    "NY", "NJ", "MA", "MI", "PA",
    "CA", "IL", "FL", "LA", "TX",
    "CT", "GA", "WA", "MD", "ID",
    "CO", "OH", "VA", "TN", "NC",
    "MO", "AL", "AZ", "WI", "SC",
    "NV", "MS", "RI", "UT", "OK", "KY",
    "DC", "DE", "IA", "MN", "OR",
    "IN", "AR", "KS", "NM", "NH",
    # "ID", "AR", "KS", "NM", "NH",     # replaced potentially faulty double assignment of "Idaho" by missing "Indiana"
    "PR", "SD", "NE", "VT", "ME",
    "WV", "HI", "MT", "ND", "AK",
    "WY", "GU", "VI", "MP", "AS"
]


"""
Map a two letter abbreviation to the corresponding US state's/territory's full name.
"""
us_abbr_to_state = Dict(
    "AK" => "Alaska",
    "AS" => "American Samoa",
    "AL" => "Alabama",
    "AR" => "Arkansas",
    "AZ" => "Arizona",
    "CA" => "California",
    "CO" => "Colorado",
    "CT" => "Connecticut",
    "DC" => "District of Columbia",
    "DE" => "Delaware",
    "FL" => "Florida",
    "GA" => "Georgia",
    "GU" => "Guam",
    "HI" => "Hawaii",
    "IA" => "Iowa",
    "ID" => "Idaho",
    "IL" => "Illinois",
    "IN" => "Indiana",
    "KS" => "Kansas",
    "KY" => "Kentucky",
    "LA" => "Louisiana",
    "MA" => "Massachusetts",
    "MD" => "Maryland",
    "ME" => "Maine",
    "MI" => "Michigan",
    "MP" => "Northern Mariana Islands",
    "MN" => "Minnesota",
    "MO" => "Missouri",
    "MS" => "Mississippi",
    "MT" => "Montana",
    "NC" => "North Carolina",
    "ND" => "North Dakota",
    "NE" => "Nebraska",
    "NH" => "New Hampshire",
    "NJ" => "New Jersey",
    "NM" => "New Mexico",
    "NV" => "Nevada",
    "NY" => "New York",
    "OH" => "Ohio",
    "OK" => "Oklahoma",
    "OR" => "Oregon",
    "PA" => "Pennsylvania",
    "PR" => "Puerto Rico",
    "RI" => "Rhode Island",
    "SC" => "South Carolina",
    "SD" => "South Dakota",
    "TN" => "Tennessee",
    "TX" => "Texas",
    "UT" => "Utah",
    "VA" => "Virginia",
    "VI" => "Virgin Islands",
    "VT" => "Vermont",
    "WA" => "Washington",
    "WI" => "Wisconsin",
    "WV" => "West Virginia",
    "WY" => "Wyoming"
)

"""
Map a US state/territory to the corresponding two letter abbreviation.
"""
us_state_to_abbr = Dict(state => abbr for (abbr, state) in us_abbr_to_state)

"""
Alphabetical order of states as used in the covid data.
"""
order_covid_data = [
    "Alabama",
    "Alaska",
    "American Samoa",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Connecticut",
    "Delaware",
    "District of Columbia",
    "Florida",
    "Georgia",
    "Guam",
    "Hawaii",
    "Idaho",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maine",
    "Maryland",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Hampshire",
    "New Jersey",
    "New Mexico",
    "New York",
    "North Carolina",
    "North Dakota",
    "Northern Mariana Islands",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Puerto Rico",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virgin Islands",
    "Virginia",
    "Washington",
    "West Virginia",
    "Wisconsin",
    "Wyoming"
]

"""Alphabetical order of abbreviations."""
order_covid_data_abbrs = [us_state_to_abbr[state] for state in order_covid_data]


"""
    read_adjacency(adjacency_file::String="../misc/adjacency.jld"; reorder::Union{Vector{String}, Nothing}=order_covid_data_abbrs)

Read the adjacency matrix representing the adjacency (neighborhood) of 56 US states.
Optionally reorder the adjacency matrix to a custom order. The adjacency matrix is originally given
in the order specified in `order_wang_et_al` (https://github.com/Rose-STL-Lab/AutoODE-DSL/blob/master/ode_nn/mobility/us_graph.pt).

By default gets reordered to the alphabetic ordering of the states defined in `order_covid_data`.
"""
function read_adjacency(adjacency_file::String="../misc/adjacency.jld"; reorder::Union{Vector{String}, Nothing}=order_covid_data_abbrs)
    adj = jld_load(adjacency_file)["adjacency"]

    if !isnothing(reorder)
        reorder_indices = indexin(reorder, order_wang_et_al)
        adj = adj[reorder_indices, reorder_indices]
    end

    return adj
end


"""
    confirm_adjacency(adj, idx, order)

Print the neighbors of a chosen state at index `idx` as given by adjacency matrix `adj`.
Array `order` must contain the corresponding state abbreviations in the same order as encoded in `adj`.
"""
function confirm_adjacency(adj, idx, order)
    println("idx $idx:\t$(us_abbr_to_state[order[idx]])")
    nghb_mask = Bool.(adj[idx, :])
    println("neighbors: $([us_abbr_to_state[st] for st in order[nghb_mask]])")
end
