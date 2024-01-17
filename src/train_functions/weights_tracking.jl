# Functions for weights tracking
module WeightTracking

using Flux


# Intitialisation for standard Flux.Dense layer
function weight_container_init(layer::Flux.Dense; n_iter::Int64)
    w_dict = Dict()
    dim_dict = Dict()
    for (pos, param) in enumerate(Flux.params(layer))
        param_size = size(param)
        w_dict[string(pos)] = zeros32(param_size..., n_iter)
        dim_dict[string(pos)] = ntuple(_ -> (:), length(param_size))
    end

    return w_dict, dim_dict
end


# Extension to Flux.Chain
function weight_container_init(chain::Flux.Chain; n_iter::Int64)
    w_dict = Dict()
    dim_dict = Dict()
    for layer in chain
        layer_name = split(string(layer), "(")[1]
        layer_dicts = weight_container_init(layer, n_iter=n_iter)
        w_dict[layer_name] = layer_dicts[1]
        dim_dict[layer_name] = layer_dicts[2]
    end

    return w_dict, dim_dict
end

end # module
