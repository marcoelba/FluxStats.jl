# Functions for weights tracking
module WeightTracking

using Flux


# --------------- weights containers ---------------

# Intitialisation for standard Flux.Dense layer
function weight_container_init(layer::Flux.Dense; n_iter::Int64)
    w_dict = Dict()
    for (pos, param) in enumerate(Flux.params(layer))
        param_size = size(param)
        w_dict[string(pos)] = zeros32(param_size..., n_iter)
    end

    return w_dict
end


# Extension to Flux.Chain
function weight_container_init(chain::Flux.Chain; n_iter::Int64)
    w_dict = Dict()
    for layer in chain
        layer_name = split(string(layer), "(")[1]
        layer_dict = weight_container_init(layer, n_iter=n_iter)
        w_dict[layer_name] = layer_dict
    end

    return w_dict
end


# --------------- weights dimension ---------------

# Intitialisation for standard Flux.Dense layer
function container_dim_init(layer::Flux.Dense)
    dim_dict = Dict()
    for (pos, param) in enumerate(Flux.params(layer))
        param_size = size(param)
        dim_dict[string(pos)] = ntuple(_ -> (:), length(param_size))
    end

    return dim_dict
end

function container_dim_init(chain::Flux.Chain)
    dim_dict = Dict()
    for layer in chain
        layer_name = split(string(layer), "(")[1]
        layer_dict = container_dim_init(layer)
        dim_dict[layer_name] = layer_dict
    end

    return dim_dict
end


# --------------- weights tracking update ---------------

function weight_tracking_push!(epoch::Int64, layer::Flux.Dense, dict_weights_layer::Dict, dict_dims_layer::Dict)
    for (pos, param) in enumerate(Flux.params(layer))
        dict_weights_layer[string(pos)][dict_dims_layer[string(pos)]..., epoch] = param
    end
end

function weight_tracking_push!(epoch::Int64, chain::Flux.Chain, dict_weights::Dict, dict_dims::Dict)
    for layer in chain
        layer_name = split(string(layer), "(")[1]
        weight_tracking_push!(epoch, layer, dict_weights[layer_name], dict_dims[layer_name])
    end
end


end # module
