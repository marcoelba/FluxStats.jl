# Training wrapper

using Flux
using Distributions

# Internal
using FluxStats: Penalties, CustomFluxLayers, FunctionalFluxModel


"""
    Model training wrapper

    model_train(
    X_train::Matrix{Float32}, y_train::Matrix{Float32};
    model::Flux.Chain, loss_function, optim::Flux.Optimise.AbstractOptimiser, n_iter::Integer,
    X_val::Matrix{Float32}=nothing, y_val::Matrix{Float32}=nothing
    )
"""
function model_train(
    X_train::Matrix{Float32},
    y_train::Matrix{Float32};
    model::Union{Flux.Chain, FunctionalFluxModel.FluxRegModel},
    loss_function::Union{Distributions.Distribution},
    optim::Flux.Optimise.AbstractOptimiser,
    n_iter::Integer,
    X_val::Union{Matrix{Float32}, Nothing}=nothing,
    y_val::Union{Matrix{Float32}, Nothing}=nothing,
    track_weights::Bool=false
    )

    optim = Flux.setup(optim, model)

    train_loss = Float32[]
    val_loss = Float32[]

    dict_weights = Dict()
    dict_dims = Dict()
    if track_weights
        for layer in model
            l_name = split(string(layer), "(")[1]
            dict_weights[l_name] = Dict()
            dict_dims[l_name] = Dict()
            for (pos, param) in enumerate(Flux.params(layer))
                param_size = size(param)
                dict_weights[l_name][string(pos)] = zeros32(param_size..., n_iter)
                dict_dims[l_name][string(pos)] = ntuple(_ -> (:), length(param_size))
            end
        end
    end

    for epoch in 1:n_iter
        n_batch = length(y_train)
        loss, grads = Flux.withgradient(model) do m
            # Evaluate model and loss inside gradient context:
            model_predictions = m(X_train)
            loss_function(y_train, y_hat) + Penalties.penalty(m)
        end
        Flux.update!(optim, model, grads[1])
        push!(train_loss, loss)  # logging, outside gradient context

        # validation
        if !isnothing(y_val)
            model_predictions_val = model(X_val)
            v_loss = loss_function(y_val, y_val_hat)
            push!(val_loss, v_loss)
        end

        if track_weights
            for layer in model
                l_name = split(string(layer), "(")[1]
                for (pos, param) in enumerate(Flux.params(layer))
                    dict_weights[l_name][string(pos)][dict_dims[l_name][string(pos)]..., epoch] = param
                end
            end
        end
    end

    out_dict = Dict(
        ("model" => model),
        ("train_loss" => train_loss),
        ("val_loss" => val_loss),
        ("dict_weights" => dict_weights)
    )

    return out_dict
end
