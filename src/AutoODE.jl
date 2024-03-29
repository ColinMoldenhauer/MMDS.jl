using DifferentialEquations: ODEProblem
import ChaoticNDETools: ChaoticNDE

using Flux



"""
    AbstractAutoODEModel

Abstract type for any AutoODE model.
Concrete subtypes should implement fields y₀, u₀, θ, q (see `ST-SuEIR`).
"""
abstract type AbstractAutoODEModel end


# abstract constructor for no constant params
function (::Type{A})(y₀::AbstractArray{T}, u₀::AbstractArray{T}, θ::AbstractArray{Matrix{T}}, f::Function; q::AbstractArray=[]) where {A<:AbstractAutoODEModel, T<:Real}
    A{T}(y₀, u₀, θ, q, f)
end

# make parameters trainable
@Flux.layer AbstractAutoODEModel trainable=(u₀, θ,)

# forward pass
function (a::AbstractAutoODEModel)(u, t)
    a.f(u, a.θ, t, a.q)
end


struct ST_SuEIR{T<:Real} <: AbstractAutoODEModel
    y₀::Matrix{T}                           # known initial conditions
    u₀::Matrix{T}                           # unknown, learnable initial conditions

    θ::Vector{Matrix{T}}                    # learnable parameters
    q::Vector{Matrix{T}}                    # constant parameters

    f::Function                             # ODE right hand side: f(u, θ, t, q)
end

# constructor
function ST_SuEIR(y₀::AbstractArray{T}; adjacency::Union{Matrix, Nothing}=nothing, low_rank::Union{Int, Nothing}=nothing) where T<:Real
    N_states = size(y₀, 1)
    adjacency = isnothing(adjacency) ? ones(T, (N_states, N_states)) : adjacency
    @assert N_states == size(adjacency, 1) == size(adjacency, 2) "Adjacency matrix of size $(size(adjacency)) must be of size ($N_states, $N_states)."

    u₀, θ = create_ST_SuEIR_initial_conditions(T, N_states, low_rank)
    f = isnothing(low_rank) ? f_ST_SuEIR : f_ST_SuEIR_low_rank
    ST_SuEIR{T}(y₀, u₀, θ, [adjacency], f)
end


"""
    create_ST_SuEIR_initial_conditions(T::Type, N_states::Int=56, low_rank=Union{Int, Nothing})

Create random initial data for ST_SuEIR unknowns.
"""
function create_ST_SuEIR_initial_conditions(T::Type, N_states::Int=56, low_rank=Union{Int, Nothing})
    # parameters
    β = rand(T, N_states, 1)
    γ = rand(T, N_states, 1)
    μ = rand(T, N_states, 1)
    σ = rand(T, N_states, 1)
    a = rand(T, N_states, 1)
    b = rand(T, N_states, 1)

    # optionally use a low rank approximation of C
    if isnothing(low_rank)
        C = rand(T, N_states, N_states)
        θ = [β, γ, μ, σ, a, b, C]
    else
        B = rand(T, N_states, low_rank)
        B2 = rand(T, low_rank, N_states)
        θ = [β, γ, μ, σ, a, b, B, B2]
    end

    # initial conditions
    S₀ = fill(T(0.5), N_states, 1)
    E₀ = fill(T(0.5), N_states, 1)
    U₀ = @. ((1 - μ) * σ) * E₀

    u₀ = [S₀ E₀ U₀]

    return u₀, θ
end



"""
    wrap_model(model::M; alg=RK4(), kwargs...) where {M <: Union{AbstractAutoODEModel, AbstractNeuralODEModel}}

Wrap `model` in a `ChaoticNDE` and additionally return the re-transformation `re`. For details, see constructor of `ChaoticNDE` below.
"""
function wrap_model(model::M; alg=RK4(), kwargs...) where {M <: Union{AbstractAutoODEModel, AbstractNeuralODEModel}}
    # initialize underlying Flux model and destructure
    p, re = destructure(model)

    # restructure and define ODEProblem
    ode(u, p, t) = re(p)(u, t)
    tspan = eltype(st.y₀).((0, 1))  # dummy tspan, will be overwritten in forward call of ChaoticNDE
    prob = ODEProblem(ode, model.u₀, tspan, p)

    # wrap in ChaoticNDE
    return re, ChaoticNDE(prob; alg=alg, kwargs...)
end


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
    tspan = eltype(st.y₀).((0, 1))  # dummy tspan, will be overwritten in forward call of ChaoticNDE
    prob = ODEProblem(ode, model.u₀, tspan, p)

    # wrap in ChaoticNDE
    return ChaoticNDE(prob; alg=alg, kwargs...)
end
