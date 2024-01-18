# Custom Flux Models structure
module FunctionalFluxModel

using Flux
using Distributions

# Internal
using FluxStats: Penalties, WeightTracking


struct FluxRegModel{LM<:Flux.Chain, LV<:Flux.Chain}
    chain_mean::LM
    chain_var::LV
end

Flux.@functor FluxRegModel

function (m::FluxRegModel)(x_mean::AbstractVecOrMat, x_var::AbstractVecOrMat)
    # layer mean is a standard chain model
    mean_pred = m.chain_mean(x_mean)
    # layer var is needed just to create the parameter for the variance of the outcome (like var of a Normal)
    var_pred = m.chain_var(x_var)

    return (mean_pred, var_pred)
end

function (m::FluxRegModel)(x::Tuple{AbstractVecOrMat, AbstractVecOrMat})
    # layer mean is a standard chain model
    mean_pred = m.chain_mean(x[1])
    # layer var is needed just to create the parameter for the variance of the outcome (like var of a Normal)
    var_pred = m.chain_var(x[2])

    return (mean_pred, var_pred)
end

function (m::FluxRegModel)(x_mean::AbstractVecOrMat)
    # layer mean is a standard chain model
    mean_pred = m.chain_mean(x_mean)
    # layer var is needed just to create the parameter for the variance of the outcome (like var of a Normal)
    x_var = ones32(1, size(x_mean, 2))
    var_pred = m.chain_var(x_var)

    return (mean_pred, var_pred)
end


function Base.show(io::IO, l::FluxRegModel)
    println(io, "FluxRegModel(")
    println(io, "\t", "chain mean: ", l.chain_mean, ",")
    println(io, "\t", "chain var: ", l.chain_var)
    print(io, ")")
end

# Add penalty functionality (multiple dispatch)
function Penalties.penalty(reg_model::FluxRegModel)
    Penalties.penalty(reg_model.chain_mean) + Penalties.penalty(reg_model.chain_var)
end

# Extension of WeightTracking.weight_container_init to FluxRegModel
function WeightTracking.weight_container_init(model::FluxRegModel; n_iter::Int64)
    w_dict = Dict()
    w_dict["chain_mean"] = WeightTracking.weight_container_init(model.chain_mean, n_iter=n_iter)
    w_dict["chain_var"] = WeightTracking.weight_container_init(model.chain_var, n_iter=n_iter)

    return w_dict
end

function WeightTracking.weight_tracking_push!(epoch::Int64, model::FluxRegModel, dict_weights::Dict, dict_dims::Dict)
    WeightTracking.weight_tracking_push!(epoch, model.chain_mean, dict_weights["chain_mean"], dict_dims["chain_mean"])
    WeightTracking.weight_tracking_push!(epoch, model.chain_var, dict_weights["chain_var"], dict_dims["chain_var"])
end

function WeightTracking.container_dim_init(model::FluxRegModel)
    dim_dict = Dict()
    dim_dict["chain_mean"] = WeightTracking.container_dim_init(model.chain_mean)
    dim_dict["chain_var"] = WeightTracking.container_dim_init(model.chain_var)

    return dim_dict
end


end
