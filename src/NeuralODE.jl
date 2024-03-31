using Flux
import Base: display, show


"""
- recognition network: RNN with 25 hidden units. We use a 4-dimensional latent space
- dynamics function f:  one-hidden-layer network with 20 hidden units
- decoder: neural network with one hidden layer with 20 hidden units
"""



"""AbstractNeuralODEModel"""
abstract type AbstractNeuralODEModel end


"""
    LatentODE

The LatentODE model as implemented by `Wang et al. - Bridging Physics-based and Data-driven modeling for Learning Dynamical Systems`.
"""
struct LatentODE
    obs_dim::Int
    latent_dim::Int
    hidden_dim_rec::Int
    hidden_dim_fun::Int
    hidden_dim_dec::Int

    chain::Flux.Chain

    reverse_x::Bool

    function LatentODE(
        obs_dim::Int=3,
        latent_dim::Int=4,
        hidden_dim_rec::Int=25,
        hidden_dim_fun::Int=20,
        hidden_dim_dec::Int=20;
        reverse_x::Bool=true,
        n_hidden_fun::Int=3,
        σ_fun::Function=Flux.elu,
        σ_dec::Function=Flux.relu
        )

        chain = Flux.Chain(
            LatentODERecognition(
                obs_dim,
                latent_dim,
                hidden_dim_rec
            ),
            LatentODEDynamicFunction(
                latent_dim,
                hidden_dim_fun,
                n_hidden=n_hidden_fun,
                σ=σ_fun
            ),
            LatentODEDecoder(
                obs_dim,
                latent_dim,
                hidden_dim_dec,
                σ=σ_dec
            )
        )
        return new(obs_dim, latent_dim, hidden_dim_rec, hidden_dim_fun, hidden_dim_dec, chain, reverse_x)
    end
end

# define forward pass
function (l::LatentODE)(x)
    if l.reverse_x
        l.chain(x[:, end:-1:1])
    else
        l.chain(x)
    end
end

# make layer
Flux.@layer LatentODE

Base.show(l::LatentODE) = begin
    print("LatentODE with chain: $l.chain")
end

Base.display(l::LatentODE) = begin
    println("LatentODE with chain:")
    display(l.chain)
end


struct LatentODERecognition
    obs_dim::Int
    latent_dim::Int
    hidden_dim::Int

    rnn         # Flux.GRU
    linear      # Flux.Dense


    # constructor
    function LatentODERecognition(
        obs_dim::Int=3,
        latent_dim::Int=4,
        hidden_dim::Int=25
    )
        # TODO: aligns with python implementation, aligns with Chen et al.?
        rnn = GRU(obs_dim => hidden_dim)
        linear = Dense(hidden_dim => latent_dim)
        new(obs_dim, latent_dim, hidden_dim, rnn, linear)
    end
end

# define forward pass
# ST_SuEIR: (n_states, n_ode (6), t)
# LatentODE: (1 state?, n_ode (3?), t)
# NeuralODE: ??
function (r::LatentODERecognition)(x)
    y = [r.rnn(x_) for x_ in eachcol(x)]
    return r.linear(y[end])
end

# make layer
Flux.@layer LatentODERecognition


struct LatentODEDynamicFunction
    latent_dim::Int
    hidden_dim::Int
    n_hidden::Int

    chain::Flux.Chain

    # constructor
    function LatentODEDynamicFunction(
        latent_dim::Int=4,
        hidden_dim::Int=20;
        n_hidden=3,
        σ=Flux.elu
    )
        chain = Flux.Chain(
            Dense(latent_dim => hidden_dim), σ,
            repeat([Dense(hidden_dim => hidden_dim), σ], n_hidden)...,
            Dense(hidden_dim => latent_dim)
        )

        return new(latent_dim, hidden_dim, n_hidden, chain)
    end
end

# define forward pass
(d::LatentODEDynamicFunction)(x) = d.chain(x)

# make layer
Flux.@layer LatentODEDynamicFunction


struct LatentODEDecoder
    obs_dim::Int
    latent_dim::Int
    hidden_dim::Int

    chain::Flux.Chain

    function LatentODEDecoder(
        obs_dim::Int=3,
        latent_dim::Int=4,
        hidden_dim=20;
        σ=Flux.relu
    )
        chain = Flux.Chain(
            Dense(latent_dim, hidden_dim),
            σ,
            Dense(hidden_dim => obs_dim)
        )

        return new(obs_dim, latent_dim, hidden_dim, chain)
    end
end

# define forward pass
(d::LatentODEDecoder)(x) = d.chain(x)

# make layer
Flux.@layer LatentODEDecoder



"""
    NeuralODE

The linear encode variant of the NeuralODE implementation of `Wang et al. - Bridging Physics-based and Data-driven modeling for Learning Dynamical Systems`.
"""
struct NeuralODE
    input_length::Int
    input_dim::Int
    hidden_dim::Int

    chain::Flux.Chain

    function NeuralODE(
        input_length::Int=7,
        input_dim::Int=3,
        hidden_dim::Int=25;
        n_hidden::Int=5,
        σ=Flux.leakyrelu
        )

        chain = Flux.Chain(
            Dense(input_length*input_dim, hidden_dim), σ,
            repeat([Dense(hidden_dim => hidden_dim), σ], n_hidden)...,
            Dense(hidden_dim => input_dim)
        )
        new(input_length, input_dim, hidden_dim, chain)
    end
end

function (n::NeuralODE)(x)
    n.chain(vec(x))
end

Flux.@layer NeuralODE



"""
    NeuralODE_LSTM

The LSTM encoder variant of the NeuralODE implementation of `Wang et al. - Bridging Physics-based and Data-driven modeling for Learning Dynamical Systems`.
"""
struct NeuralODE_LSTM
    input_length::Int
    input_dim::Int
    hidden_dim::Int

    lstm
    linear

    function NeuralODE_LSTM(
        input_length::Int=7,
        input_dim::Int=3,
        hidden_dim::Int=25;
        )

        lstm = LSTM(input_dim => hidden_dim)    # num_layers? bidirectional? -> *2
        linear = Dense(hidden_dim, input_dim)

        new(input_length, input_dim, hidden_dim, lstm, linear)
    end
end

function (n::NeuralODE_LSTM)(x)
    y = [n.lstm(x_) for x_ in eachcol(x)]
    return n.linear(y[end])
end

Flux.@layer NeuralODE_LSTM


# TODO: output 4D or 3D (which ODE if 3D??)
function f_SEIR(u, p, t)
    S, E, I, R = eachcol(u)
    println("S $(size(S))")
    println("E $(size(E))")
    β, γ, σ = p


    dS = @. - β*S*I
    dE = @. β*S *I - σ*E
    dI = @. σ*E - γ*I
    dR = @. γ * I

    du = [dS dE dI dR]
    println("size(u)  $(size(u))")
    println("size(du)  $(size(du))")
    return du
end

"""
A simple neural ODE as coined by [Chen et al.: Neural Differential Equations](TODO: link).
Models the RHS of an ODE by means of a neural network. Can be used by general ODE solvers.
"""
struct SimpleNeuralODE <: AbstractNeuralODEModel

    dims::AbstractArray{Int}

    chain::Flux.Chain

    # default constructor
    function SimpleNeuralODE(dims::AbstractArray{Int}; hidden_dims::AbstractArray{Int}=[128, 256, 512, 256, 128], σ::Function=Flux.relu)
        chain = Chain(
            Dense(prod(dims) => hidden_dims[1]), σ,
            [Flux.Chain(Dense(in => out), σ) for (in, out) in zip(hidden_dims[1:end-1], hidden_dims[2:end])]...,
            Dense(hidden_dims[end] => prod(dims))
        ) |> f64

        return new(dims, chain)
    end

    # constructor for restructuring
    function SimpleNeuralODE(dims::AbstractArray{Int}, chain::Flux.Chain)
        return new(dims, chain)
    end
end

@Flux.layer SimpleNeuralODE trainable=(chain,)

flatten_all(x) = reshape(x, :)

# TODO: reshape supported by Zygote?

function (s::SimpleNeuralODE)(x; flatten=true)
    if flatten
        x = flatten_all(x)
    end
    pred = s.chain(x)
    return reshape(pred, s.dims...)
end

# forward
function (s::SimpleNeuralODE)(x, t; flatten=true)
    if flatten
        x = flatten_all(x)
    end
    pred = s.chain(x)
    return reshape(pred, s.dims...)
end
