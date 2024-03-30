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


"""
    plot_sequence(u_true, û...; state_index=1, t=nothing, labels=nothing, colors=nothing, L_train=nothing, height=300, kwargs...)

Plot a ground truth sequence and an arbitrary number of predicted trajectories.
"""
function plot_sequence(u_true, û...; state_index=1, t=nothing, labels=nothing, colors=nothing, L_train=nothing, height=300, kwargs...)

    t = isnothing(t) ? (0:size(u_true, ndims(u_true))-1) : t
    labels = isnothing(labels) ? ["model $i" for i in 1:length(û)] : labels
    colors = isnothing(colors) ? palette(:default)[1:length(û)]' : colors
    vlines = isnothing(L_train) ? [] : (L_train isa Array ? L_train : [L_train])

    l = @layout [a b c]

    plt1 = plot(t, u_true[state_index, 1, :], color="black", label="True")
    vline!(vlines, label=nothing, ls=:dash, c="purple")
	for (û_, label, color) in zip(û, labels, colors)
		plot!(t, û_[state_index, 1, :], color=color, ls=:dash, label=label)
	end
    plot!(title="#Infected", xticks=t[1:2:end])

    plt2 = plot(t, u_true[state_index, 2, :], color="black", label="True")
    vline!(vlines, label=nothing, ls=:dash, c="purple")
	for (û_, label, color) in zip(û, labels, colors)
        plot!(t, û_[state_index, 2, :], color=color, ls=:dash, label=label)
	end
    plot!(title="#Removed", xticks=t[1:2:end])

    plt3 = plot(t, u_true[state_index, 3, :], color="black", label="True")
    vline!(vlines, label=nothing, ls=:dash, c="purple")
	for (û_, label, color) in zip(û, labels, colors)
        plot!(t, û_[state_index, 3, :], color=color, ls=:dash, label=label)
	end
    plot!(title="#Death", xticks=t[1:2:end])

    plot(plt1, plt2, plt3, layout=l, size = (3*height, height), kwargs...)
end
