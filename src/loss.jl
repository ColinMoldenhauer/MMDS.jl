
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
