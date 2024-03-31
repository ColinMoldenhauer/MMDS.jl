
"""
    f_ST_SuEIR(u, θ, t, q)

RHS of the ST-SuEIR model.
"""
function f_ST_SuEIR(u, θ, t, q)
    β, γ, μ, σ, a, b, C = θ
    M, = q

    A = C .* M

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


"""
    f_ST_SuEIR_low_rank(u, θ, t, q)

RHS of the ST-SuEIR model. Use a low rank approximation for the inter-state transmission matrix C.
"""
function f_ST_SuEIR_low_rank(u, θ, t, q)
    β, γ, μ, σ, a, b, B, B2 = θ
    M, = q

    C = B * B2
    A = C .* M

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