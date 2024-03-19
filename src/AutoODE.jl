using Flux, Optimisers

using DifferentialEquations
import ChaoticNDETools: ChaoticNDE


abstract type AbstractAutoODEModel end

# TODO: parametrize
# TODO: make p0 optional? if m=Dense(), no params needed -> include initialization in ST_SuEIR.constructor
"""
ChaoticNDE(m::Type{M}, p0, u0, tspan; alg=RK4(), kwargs...) where {M <: AbstractAutoODEModel}

Overload the constructor of `ChaoticNDE` to support the AutoODE functionality.

# Arguments
- `m`: Custom Flux model type that allows for differentiation of its parameters.
- `p0`: Initial parameters of m.
- `u0`: Initial conditions of the trajectory to be fitted.
- `tspan`: Timespan for which m's underlying DE is to be solved.
- `alg`: DE solver used to solve the trajectory.
- `kwargs`: Additional parameters for initalization of `ChaoticNDE`.
"""
function ChaoticNDE(m::Type{M}, p0, u0, tspan; alg=RK4(), kwargs...) where {M <: AbstractAutoODEModel}
    # initialize underlying Flux model and destructure
    model = m(p0...)
    p, re = destructure(model)

    # restructure and define ODEProblem
    ode(u, p, t) = re(p)(u, t)
    prob = ODEProblem(ode, u0, tspan, p)

    # wrap in ChaoticNDE
    return ChaoticNDE(prob; alg=alg, kwargs...)
end