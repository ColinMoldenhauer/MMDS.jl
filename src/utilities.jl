"""
    split_sample(X, L_train)

Split a sample of NODEDataloader into time, input/train sequence and output/prediction sequence.
"""
function split_sample(X::Tuple{Vector{I}, Array{F, 3}}, L_train::Int) where {I, F}
    t, x = X
    x_train = x[:, :, 1:L_train]
    t_train = t[1:L_train]
    return t_train, x_train, t, x
end


"""
    augment_sample(m::AbstractAutoODEModel, x::AbstractArray)

Append learnable initial conditions u₀ to known initial conditions as included
in the covid data x. Extend the last dimension to support the automatic extraction
of the initial conditions in ChaoticNDE().
"""
function augment_sample(m::AbstractAutoODEModel, x::AbstractArray)
    u₀ = m.u₀
    x₀ = x[:, :, 1]
    y₀ = [u₀ x₀]
    return reshape(y₀, size(y₀)..., 1)
end


"""
    save_params(m::ChaoticNDE{P,R,A,K,D}., file::String) where {P,R,A,K,D}

Save the parameters of model `m` to file. Used e.g. when saving parameters from trained models.
"""
function save_params(m::ChaoticNDE{P,R,A,K,D}, file::String) where {P,R,A,K,D}
    save(file, "p", m.p)
end


"""
    load_params(m::ChaoticNDE{P,R,A,K,D}, file::String) where {P,R,A,K,D}

Load the parameters of model `m` saved to file.
"""
function load_params(m::ChaoticNDE{P,R,A,K,D}, file::String) where {P,R,A,K,D}
    p = load(file)["p"]
    set_params!(m, p)
end