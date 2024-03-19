using CSV, DataFrames


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

    if ~isnothing(normalize)
        df_pop = CSV.read(normalize, DataFrame)
        rename!(df_pop, Dict("State" => "Province_State", "2018 Population" => "Population"))
    end

    dfs = []
    for file in files
        # println(file)
        df = CSV.read(joinpath(data_dir, file), DataFrame)
        try
            # filter out rows
            df_drop = filter(row -> row["Province_State"] ∉ ["Diamond Princess", "Grand Princess", "Recovered"], df)


            # select only relevant columns
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
    return dates, states, df_concat
end


"""
    get_covid_IRD(data_dir::String; normalize::Union{String, Nothing}=nothing)

Extract infected cases, recovered cases and deaths in chronological order.
Returns data as matrices of size (N_states, N_epochs).
"""
function get_covid_IRD(data_dir::String; normalize::Union{String, Nothing}=nothing)
    dates, states, data = read_covid_data(data_dir, normalize=normalize)

    infected = data[!, r"Confirmed"]
    recovered = data[!, r"Recovered"]
    deaths = data[!, r"Deaths"]

    return dates, states, Matrix(infected), Matrix(recovered), Matrix(deaths)
end


"""
    prepare
"""
# TODO: parametrize and doc
function prepare_data(I, R, D, N_states; fill_unknowns=true)
    N_t = size(I, 2)
    t = 0:(N_t-1)

    if fill_unknowns
        N_ode = 6
    else
        N_ode = 3
    end

    snapshots = rand(N_states, N_ode, N_t)     # TODO: type
    # TODO: include in loading method?
    # u0 (n_states, n_ode)
    # I (n_states, t)
    # goal (n_states, n_ode/n_known, t)

    for (i_data, data) in enumerate([I, R, D])
        snapshots[:, i_data, :] = data[1:N_states, :]
    end
    return snapshots, t
end
