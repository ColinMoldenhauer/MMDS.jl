
"""
    f_ST_SuEIR(u, p, t)

RHS of the ST-SuEIR model.
"""
function f_ST_SuEIR(u, p, t)
    θ, q = p
    β, γ, μ, σ, a, b, A = θ
    # M, = q

    S, E, U, I, R, D = eachcol(u)

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
