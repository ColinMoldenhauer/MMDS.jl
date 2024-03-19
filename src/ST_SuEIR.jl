using Flux, Optimisers
using ChaoticNDETools
using MMDS: AbstractAutoODEModel

import Base: iterate


function f_ST_SuEIR(u, p, t)
    β, γ, μ, σ, a, b, A = p
    S, E, U, I, R, D = [reshape(u_, (:, 1)) for u_ in eachcol(u)]

    transm = A * (I + E)
    dS = @. - β * transm * S
    dE = @. β * transm * S - σ*E
    dU = @. (1-γ)*σ*E

    dI = @. μ*σ*E - γ*I
    dR = @. γ*I
    r = @. a*t + b
    dD = @. r * dR

    du = [dS dE dU dI dR dD]
    return du
end


# TODO: further optionals via overloading train! ?
# TODO: train, valid = NODEDataloader(sol, 10; dt=dt, valid_set=0.8) -> train[1] or train[1][2]
# TODO: random initialization of parameters (python: torch.rand(num_regions)/10)

"""
need
    - layer (struct -> @layer)              # RHS/DE
    # - function:    (m::myLayer)(...)
    # how is re_nn behavior defined?
    - destructure
    - define rhs with restructure
    - ODEProblem
    - model = ChaoticNDE(node_prob)         # wrap
    - train!!!!
"""


struct ST_SuEIR{T<:Real} <: AbstractAutoODEModel
    # learnable parameters
    β::AbstractArray{T}
    γ::AbstractArray{T}
    μ::AbstractArray{T}
    σ::AbstractArray{T}
    a::AbstractArray{T}
    b::AbstractArray{T}

    A::AbstractMatrix{T}
end

# make model parameters trainable
Flux.@layer ST_SuEIR trainable=(β, γ, μ, σ, a, b, A,)


# fwd
function (m::ST_SuEIR)(u, p, t)
    return f_ST_SuEIR(u, p, t)
end

function (m::ST_SuEIR)(u, t)
    return f_ST_SuEIR(u, [m...], t)
end

function iterate(m::ST_SuEIR, state=1)
    return iterate([m.β, m.γ, m.μ, m.σ, m.a, m.b, m.A], state)
end



function create_ST_SuEIR_initial_conditions(N_states)
    # initial conditions
    S₀ = rand(N_states, 1)
    E₀ = rand(N_states, 1)
    U₀ = rand(N_states, 1)
    I₀ = rand(N_states, 1)
    R₀ = rand(N_states, 1)
    D₀ = rand(N_states, 1)

    u₀ = [S₀ E₀ U₀ I₀ R₀ D₀]   # shape (n_states, n_ODE)

    # parameters
    β = rand(N_states, 1)
    γ = rand(N_states, 1)
    μ = rand(N_states, 1)
    σ = rand(N_states, 1)
    a = rand(N_states, 1)
    b = rand(N_states, 1)
    A = rand(N_states, N_states)

    p = [β, γ, μ, σ, a, b, A]

    return u₀, p
end
