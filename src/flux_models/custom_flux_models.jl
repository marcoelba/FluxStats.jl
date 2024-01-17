# Custom Flux Models structure
module FunctionalFluxModel

using Flux
using Distributions

# Internal
using FluxStats: Penalties, WeightTracking


struct FluxRegModel{LM<:Flux.Chain, LV<:Flux.Chain}
    layer_mean::LM
    layer_var::LV
end

Flux.@functor FluxRegModel

function (m::FluxRegModel)(x_mean::AbstractVecOrMat, x_var::AbstractVecOrMat)
    # layer mean is a standard chain model
    mean_pred = m.layer_mean(x_mean)
    # layer var is needed just to create the parameter for the variance of the outcome (like var of a Normal)
    var_pred = m.layer_var(x_var)

    return (mean_pred, var_pred)
end

function (m::FluxRegModel)(x::Tuple{AbstractVecOrMat, AbstractVecOrMat})
    # layer mean is a standard chain model
    mean_pred = m.layer_mean(x[1])
    # layer var is needed just to create the parameter for the variance of the outcome (like var of a Normal)
    var_pred = m.layer_var(x[2])

    return (mean_pred, var_pred)
end

function (m::FluxRegModel)(x_mean::AbstractVecOrMat)
    # layer mean is a standard chain model
    mean_pred = m.layer_mean(x_mean)
    # layer var is needed just to create the parameter for the variance of the outcome (like var of a Normal)
    x_var = ones32(1, size(x_mean, 2))
    var_pred = m.layer_var(x_var)

    return (mean_pred, var_pred)
end


function Base.show(io::IO, l::FluxRegModel)
    println(io, "FluxRegModel(")
    println(io, "\t", l.layer_mean, ",")
    println(io, "\t", l.layer_var)
    print(io, ")")
end

# Add penalty functionality (multiple dispatch)
function Penalties.penalty(reg_model::FluxRegModel)
    Penalties.penalty(reg_model.layer_mean) + Penalties.penalty(reg_model.layer_var)
end

# Extension of WeightTracking.weight_container_init to FluxRegModel
function WeightTracking.weight_container_init(chain::FluxRegModel; n_iter::Int64)
    w_dict = Dict()
    dim_dict = Dict()
    for layer in chain
        layer_name = split(string(layer), "(")[1]
        layer_dicts = WeightTracking.weight_container_init(layer, n_iter=n_iter)
        w_dict[layer_name] = layer_dicts[1]
        dim_dict[layer_name] = layer_dicts[2]
    end

    return w_dict, dim_dict
end


end
