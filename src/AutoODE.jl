using Flux, Optimisers

using Statistics
using DifferentialEquations
import ChaoticNDETools: ChaoticNDE


"""
    wrap_model(model::M; alg=RK4(), kwargs...) where {M <: Union{AbstractAutoODEModel, AbstractNeuralODEModel}}

Wrap `model` in a `ChaoticNDE` and additionally return the re-transformation `re`.
"""
function wrap_model(model::M; alg=RK4(), kwargs...) where {M <: Union{AbstractAutoODEModel, AbstractNeuralODEModel}}
    # initialize underlying Flux model and destructure
    p, re = destructure(model)

    # restructure and define ODEProblem
    ode(u, p, t) = re(p)(u, t)
    tspan = [0, 1]  # dummy tspan, not used TODO: type
    prob = ODEProblem(ode, model.u₀, tspan, p)

    # wrap in ChaoticNDE
    return re, ChaoticNDE(prob; alg=alg, kwargs...)
end


# TODO: parametrize
# TODO: make p0 optional? if m=Dense(), no params needed -> include initialization in ST_SuEIR.constructor
"""
ChaoticNDE(model::M; alg=RK4(), kwargs...) where {M <: Union{AbstractAutoODEModel, AbstractNeuralODEModel}}

Utility overload to quickly construct a `ChaoticNDE` from a Flux model.

# Arguments
- `m`: Custom Flux model type that allows for differentiation of its parameters.
- `alg`: DE solver used to solve the trajectory.
- `kwargs`: Additional parameters for initalization of `ChaoticNDE`.
"""
function ChaoticNDE(model::M; alg=RK4(), kwargs...) where {M <: Union{AbstractAutoODEModel, AbstractNeuralODEModel}}
    # initialize underlying Flux model and destructure
    p, re = destructure(model)

    # restructure and define ODEProblem
    ode(u, p, t) = re(p)(u, t)
    tspan = [0, 1]  # dummy tspan, not used TODO: type
    prob = ODEProblem(ode, model.u₀, tspan, p)

    # wrap in ChaoticNDE
    return ChaoticNDE(prob; alg=alg, kwargs...)
end


"""
    loss_covid(I, R, D, p, t; w=t->.√(t), α₁=1, α₂=.5)

Compute the loss as defined in `Wang et al. - Bridging Physics-based and Data-driven modeling for Learning Dynamical Systems`.

# Arguments
- `û`: Predicted trajectory of shape (N_states, N_ode, t), with the last 3 slices along dimension 2 represent the predicted I, R, D.
- `u`: True trajectory of shape (N_states, N_ode, t).
- `t`: Time vector of trajectory.

- `w`: Weighting function for loss of respective time steps.
- `α₁`: Weight of loss w.r.t R
- `α₂`: Weight of loss w.r.t D
"""
function loss_covid(û, u, t; w=t->.√((1:length(t))'), α₁=1, α₂=.5, q=.5, σ=mean)

    l_I = @. quantile_regression_loss(u[:, 1, :], û[:, 1, :], q)
    l_R = @. α₁ * quantile_regression_loss(u[:, 2, :], û[:, 2, :], q)
    l_D = @. α₂ * quantile_regression_loss(u[:, 3, :], û[:, 3, :], q)

    weights = w(t)
    l = @. weights * (l_I + l_R + l_D)

    return σ(l)
end


"""
    quantile_regression_loss(y, ŷ, q=.5)

Calculate the scalar quantile qegression loss as specified in `Wen et al. - A Multi-Horizon Quantile Recurrent Forecaster`.
"""
function quantile_regression_loss(y, ŷ, q)
    return @. q * max(0, (y - ŷ)) + (1-q) * max(0, (ŷ - y))
end


"""
    create_ST_SuEIR_initial_conditions(T::Type, N_states::Int=56)

Create initial data for ST_SuEIR unknowns.
"""
function create_ST_SuEIR_initial_conditions(T::Type, N_states::Int=56)
    # parameters
    β = rand(T, N_states, 1)
    γ = rand(T, N_states, 1)
    μ = rand(T, N_states, 1)
    σ = rand(T, N_states, 1)
    a = rand(T, N_states, 1)
    b = rand(T, N_states, 1)
    A = rand(T, N_states, N_states)

    θ = [β, γ, μ, σ, a, b, A]


    # initial conditions
    S₀ = fill(T(0.5), N_states, 1)
    E₀ = fill(T(0.5), N_states, 1)
    U₀ = @. ((1 - μ) * σ) * E₀

    u₀ = [S₀ E₀ U₀]

    return u₀, θ
end


struct AutoODE_ST_SuEIR{T<:Real} <: AbstractAutoODEModel
    y₀::AbstractArray{T}                    # known initial conditions
    u₀::AbstractArray{T}                    # unknown, learnable initial conditions

    θ::AbstractArray{Matrix{Float64}}       # learnable parameters
    q::AbstractArray                        # known parameters


    # constructor
    function AutoODE_ST_SuEIR(y₀::AbstractArray{T}; q::AbstractArray=[]) where T<:Real
        N_states = size(y₀, 1)
        u₀, θ = create_ST_SuEIR_initial_conditions(T, N_states)
        new{T}(y₀, u₀, θ, q)
    end

    function AutoODE_ST_SuEIR(y, u, thet, p)
        new{typeof(y[1])}(y, u, thet, p)
    end
end

@Flux.layer AutoODE_ST_SuEIR trainable=(u₀, θ,)

function (a::AutoODE_ST_SuEIR)(u, t)
    p = [a.θ, a.q]
    f_ST_SuEIR(u, p, t)
end
