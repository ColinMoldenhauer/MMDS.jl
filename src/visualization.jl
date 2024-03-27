using Plots

"""
    plot_covid_data(data, states; select_states=nothing, legend::Bool=false, kwargs...)

Plot singular covid data for a subset of states.

# Arguments
- `data`: either I, R, or D returned by `get_covid_CRD()`.
- `states`: all states present in the data (rows).
- `select_states`: the states to be visualized.
- `legend`: whether to show the legend.
"""
function plot_covid_data(data, states; select_states=nothing, legend::Bool=false, kwargs...)
    # TODO: xticks = dates
    # TODO: ytick format: 
        # https://docs.juliaplots.org/latest/generated/attributes_axis/#:%7E:text=formatter,for%20tick%20labeling.
        # https://stackoverflow.com/questions/65977114/how-can-i-make-scientific-y-ticks-in-julia-plots
    legend = legend && :outertopright
    indices = isnothing(select_states) ? (1:length(states)) : [findall(x->x==s, states)[1] for s in select_states]
    plot(data'[:, indices], labels=reshape(states[indices], 1, :), legend=legend, kwargs...)
    plot!(xlabel="days", ylabel="# cases")
end


"""
    plot_IRD(I, R, D, states; select_states=nothing, legend::Bool=false, kwargs...)

Plot confirmed cases, recovered cases and deaths for a subset of states.

# Arguments
- `I, R, D`: I, R, D returned by `get_covid_CRD()`.
- `states`: all states present in the data (rows).
- `select_states`: the states to be visualized.
- `legend`: whether to show the legend.
"""
function plot_IRD(I, R, D, states; select_states=nothing, legend::Bool=false, kwargs...)
    legend = legend && :outertopright

    indices = nothing
    try
        indices = isnothing(select_states) ? (1:length(states)) : [findall(x->x==s, states)[1] for s in select_states]
    catch
        throw(KeyError("One of $select_states not found in `states`."))
    end
    colors = palette(:default)[1:length(indices)]'
    plot(I'[:, indices], labels=[s * "_infect" for s in reshape(states[indices], 1, :)], legend=legend, linestyle=:solid, color=colors, kwargs...)
    plot!(R'[:, indices], labels=[s * "_recov" for s in reshape(states[indices], 1, :)], legend=legend, linestyle=:dash, color=colors,kwargs...)
    plot!(D'[:, indices], labels=[s * "_dead" for s in reshape(states[indices], 1, :)], legend=legend, linestyle=:dashdotdot, color=colors,kwargs...)
    plot!(xlabel="days", ylabel="# cases")
end
