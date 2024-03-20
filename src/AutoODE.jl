using Flux, Optimisers

using Statistics
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

"""
    loss_covid(I, R, D, p, t; w=t->.√(t), α₁=1, α₂=.5)

Compute the loss as defined in [Wang et al.: Bridging Physics-based and Data-driven modeling for Learning Dynamical Systems](https://arxiv.org/abs/2011.10616)

# Arguments
- `û`
- `u`
"""
function loss_covid(û, u, t; w=t->.√(t'), α₁=1, α₂=.5, q=.5, σ=mean)
    _..., I, R, D = eachslice(u, dims=2)
    _..., Î, Ȓ, Ď = eachslice(û, dims=2)

    l_I = @. quantile_regression_loss(I, Î, q)
    l_R = @. α₁ * quantile_regression_loss(R, Ȓ, q)
    l_D = @. α₂ * quantile_regression_loss(D, Ď, q)

    weights = w(t)
    l = @. weights * (l_I + l_R + l_D)

    return σ(l)
end


"""
    quantile_regression_loss(y, ŷ, q=.5)

Calculate the scalar quantile qegression loss as specified in [Wen et al.: A Multi-Horizon Quantile Recurrent Forecaster](https://arxiv.org/abs/1711.11053).
"""
function quantile_regression_loss(y, ŷ, q)
    return @. q * max(0, (y - ŷ)) + (1-q) * max(0, (ŷ - y))
end