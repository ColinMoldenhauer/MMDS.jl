using Pkg
Pkg.activate("..")

using Revise

using Dates, Random
using NODEData
using ChaoticNDETools: ChaoticNDE
using Optimisers, Flux
using Printf, Dates, Statistics
using Plots

using MMDS


seed = 600
Random.seed!(seed)



####### set settings
train_idx = 300
N_epochs = 2_000
N_epochs = 200

run_dir = "../runs/autoode_sample_$(train_idx)_epochs_$(N_epochs)_seed_$seed"

checkpoint_file = "checkpoints.jld"

data_dir = "../../covid_data/csse_covid_19_data/csse_covid_19_daily_reports_us"
pop_file = "../../AutoODE-DSL/ode_nn/population_states.csv"

N_states = 56
L_train = 10        # input sequence length
L_pred = 7          # prediction horizon

# loss(ŷ, y, t) = loss_covid(ŷ, y, t)
loss(ŷ, y, t) = Flux.mse(ŷ, y)





###### start script
mkpath(run_dir)

# data
dates, states, population, I, R, D = get_covid_IRD(data_dir; normalize=pop_file)
covid_data, t_covid = prepare_data(I, R, D; N_states=N_states)
println("data: $(covid_data |> summary) \nt:    $(t_covid |> summary)")

# dataloader
dataloader = NODEDataloader(covid_data, t_covid, L_train+L_pred)

# choose sample on which to train
sample_train = dataloader[train_idx]
t_train, x_train, t, x = split_sample(sample_train, L_train)
t_train = eltype(x_train).(t_train)

# visualize the chosen sequence
plot_seq = plot_sequence(x, state_index=1, L_train=L_train)
savefig(plot_seq, joinpath(run_dir, "fig_seq.png"))

# model
st_sueir = ST_SuEIR(x[:, :, 1])     # initialize model with y₀
re_st_sueir, model = wrap_model(st_sueir, maxiters=Int(1e9), tstops=1e7, dt=0.01)
# re_st_sueir, model = wrap_model(st_sueir, maxiters=Int(1e9), dt=0.01)
# re_st_sueir, model = wrap_model(st_sueir, maxiters=Int(1e7), abstol = 1e-5, reltol = 1e-5)    # solve(prob, saveat = 0.01, abstol = 1e-9, reltol = 1e-9)

# enrich input sequence with trainable initial conditions
x_augm = augment_sample(re_st_sueir, model, x_train)

# initialize the model
@info "Initializing model..."; t0_init = now()
model((t_train, x_augm)) |> summary
@info "Initialized model after $(now() - t0_init)."

# save initial conditions to check training progress later
u₀_pre = deepcopy(st_sueir.u₀)
y₀_pre = deepcopy(st_sueir.y₀)
θ_pre = deepcopy(st_sueir.θ)
q_pre = deepcopy(st_sueir.q)
p_pre = deepcopy(model.p)


# optimizer
global η = 0.01
optim = Optimisers.setup(Flux.Adam(η), model)

# initialize the gradient
@info "Initializing gradient..."; t0_grad = now()
loss_train_init, grads_init = Flux.withgradient(model) do m
    x_augm = augment_sample(re_st_sueir, m, x_train)
    pred = m((t_train, x_augm))
    pred_IRD = pred[:, 4:6, :]
    return loss(pred_IRD, x_train, t_train)
end
@info "Initialized gradient after $(now() - t0_grad)."


# training loop
pred_val_init = model((t, x_augm))

losses_train = zeros(N_epochs)
losses_val = zeros(N_epochs)
params_history = zeros(N_epochs, length(model.p))
t0_train = now()


function myloss(m::ChaoticNDE, t::Vector{T2}, x::Matrix{T}) where {T <: Real, T2 <: Int}
    x_augm = augment_sample(re_st_sueir, m, x)
    pred = m((t, x_augm))
    pred_IRD = pred[:, 4:6, :]
    return loss(pred_IRD, x, t)
end

@code_warntype myloss(model, t_train, x_train)




loss_train = 23.3

eval_T = 1
i_e = 23
for i_e = 1:N_epochs

    t0_epoch = now()

    # try     # sometimes, maxiter of the ODE solver is reached
        # Flux.train!(model, [(t_train, x_train)], optim) do m, t, x
        #     x_augm = augment_sample(re_st_sueir, m, x)
        #     pred = m((t, x_augm))
        #     pred_IRD = pred[:, 4:6, :]
        #     return loss(pred_IRD, x, t)
        # end
        loss_train, grads = Flux.withgradient(model) do m
            x_augm = augment_sample(re_st_sueir, m, x_train)
            pred = m((t_train, x_augm))
            pred_IRD = pred[:, 4:6, :]
            return loss(pred_IRD, x_train, t_train)
        end
        t1_epoch = now()
        optim2, model2 = Flux.update!(optim, model, grads[1])
        t2_epoch = now()


        params_history[i_e, :] = model.p

        # collect loss metric
        losses_train[i_e] = loss_train

        if (i_e % 20) == 0  # reduce the learning rate every 20
            global η /= 2
            Optimisers.adjust!(optim, η)
        end
        if (i_e % eval_T) == 0  # log training progress
            # @info """Epoch $i_e \t train loss = $(@sprintf("%5.3f", loss_train))   η = $η   t = $(t2_epoch-t0_epoch)   t_avg = $(typeof(now()-t0_train)(Int(round((now()-t0_train).value/i_e))))"""
            println("Epoch $i_e \t train loss = $(@sprintf("%5.3f", loss_train))   η = $η   t = $(t2_epoch-t0_epoch)   t_avg = $(typeof(now()-t0_train)(Int(round((now()-t0_train).value/i_e))))")
            # @info """Times  t_epoch = $(t2_epoch-t0_epoch)    t_fwd = $(t1_epoch-t0_epoch)   t_bckwd = $(t2_epoch-t1_epoch)"""
        end

        # "validation"
        # if (i_e % eval_T) == 0  # calculate loss for validation set?
        #     pred_val = model((t, x_augm))
        #     pred_val_IRD = pred_val[:, 4:6, :]
        #     loss_val = loss(pred_val_IRD, x, t)
        #     losses_val[i_e] = loss_val
        #     # @info """\t\t val   loss = $(@sprintf("%5.3f", loss_val))"""
        #     println("\t\t val   loss = $(@sprintf("%5.3f", loss_val))")
        # end
    # catch e
    #     println("Caught error while training (epoch $i_e)")
    #     println(e)
    #     continue
    # end

end

println("Finished training for sample $(train_idx): $N_epochs epochs after $(now()-t0_train) ($(typeof(now()-t0_train)(Int(round((now()-t0_train).value/N_epochs))))/epoch)")


# save model training
save_params(model, joinpath(run_dir, checkpoint_file))


# visualization
# plot training progress
@info "Generating plots..."
pick_params = [1, N_states+1, N_states*2+1, N_states*3+1]

plt1 = plot(1:N_epochs, losses_train, label="train loss")
plot!(title="Loss", titlefont=font(20), xticks=1:2:N_epochs)
plot!(1:N_epochs, losses_val, label="validation loss", xticks=1:Int(round(N_epochs/10)):N_epochs)
plt2 = plot(1:N_epochs, params_history[:, pick_params], label=pick_params')
# plt2 = plot(1:N_epochs, params_history[:, pick_params], label=(1:length(model.p))')
plot!(title="Parameters", titlefont=font(20), legend=:outertopright, legend_column=2)
plt_training = plot(plt1, plt2, layout=(1,2), dpi=500, size=(1000, 400))
savefig(plt_training, joinpath(run_dir, "fig_train.png"))


# test sample after training and plot
pred_val = model((t, x_augm))
state_index = 1
plot_variables = [1]
colors = palette(:default)[1:size(pred_val, 2)]'
plt_pred = plot(sample_train[1], sample_train[2][state_index, plot_variables, :]', lw=2,label="target", color=colors)
plot!(sample_train[1], pred_val_init[state_index, 3 .+ plot_variables, :]', ls=:dashdotdot, label="initial prediction", color=colors, marker=:circle)
plot!(sample_train[1], pred_val[state_index, 3 .+ plot_variables, :]', lw=2, label="prediction after training", color=colors, marker=:diamond)
plot!(title="Predictions before vs after training\n(sample $train_idx, $N_epochs epochs)", titlefont=font(10), dpi=500, size=(1000, 400))
savefig(plt_pred, joinpath(run_dir, "fig_pred.png"))


# plot initial conditions
plot_init = plot(u₀_pre, label="pre", title=["S" "E" "U"], layout=(1,3), xlabel="states")
# u₀_post = st_sueir.u₀
u₀_post = re_st_sueir(model.p).u₀
plot!(u₀_post, label="post", dpi=500, size=(1200, 300))
savefig(plot_init, joinpath(run_dir, "fig_init.png"))

# plot some parameters
params = ["β" "γ" "μ" "σ" "a" "b" "A"]
β, γ, μ, σ, a, b, A = θ_pre
pick_params = [1, 2, 3, 4]
plot_params = plot(θ_pre[pick_params], label="pre", title=reshape(params[pick_params], 1, :), layout = (1, length(pick_params)), xlabel="states")
# θ_post = st_sueir.θ
θ_post = re_st_sueir(model.p).θ
plot!(θ_post[pick_params], dpi=500, label="post", size=(1200, 300))
savefig(plot_params, joinpath(run_dir, "fig_params.png"))

# test model and plot results
pred_st_sueir = model((t, x_augm))[:, 4:end, :]
plt_results = plot_sequence(x, pred_st_sueir, state_index=state_index, colors=["red"], labels=["ST-SuEIR"], L_train=L_train)
savefig(plt_results, joinpath(run_dir, "fig_results.png"));