using Flux


struct LatentODE
    obs_dim::Int
    latent_dim::Int
    hidden_dim_rec::Int=25
    hidden_dim_fun::Int=20
    hidden_dim_dec::Int=20

    chain::Flux.Chain

    function LatentODE(
        obs_dim::Int=3,
        latent_dim::Int=4,
        hidden_dim_rec::Int=25,
        hidden_dim_fun::Int=20,
        hidden_dim_dec::Int=20;
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
        return new(obs_dim, latent_dim, hidden_dim_rec, hidden_dim_fun, hidden_dim_dec, chain)
    end
end


struct LatentODERecognition
    obs_dim::Int
    latent_dim::Int
    hidden_dim::Int

    chain::Flux.Chain

    # constructor
    function LatentODERecognition(
        obs_dim::Int=3,
        latent_dim::Int=4,
        hidden_dim::Int=25
    )
        # TODO: aligns with python implementation, aligns with Chen et al.?
        chain = Chain(
            GRU(obs_dim => hidden_dim),
            Dense(hidden_dim => latent_dim)
        )
        new(obs_dim, latent_dim, hidden_dim, chain)
    end
end

# define forward pass
(r::LatentODERecognition)(x) = r.chain(x)

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
            repeat([Dense(hidden_dim, hidden_dim => 2), σ], n_hidden)...,
            Dense(hidden_dim => latent_dim)
        )

        return new(latent_dim, hidden_dim, chain)
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
