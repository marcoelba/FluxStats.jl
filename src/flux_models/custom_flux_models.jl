# Custom Flux Models structure
module FunctionalFluxModel

using Flux
using Distributions

# Internal
using FluxStats: Penalties


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

function Base.show(io::IO, l::FluxRegModel)
    println(io, "FluxRegModel(")
    println(io, "\t", l.layer_mean, ",")
    println(io, "\t", l.layer_var)
    print(io, ")")
end

# Add penalty functionality (multiple dispatch)
function Penalties.penalty(reg_model::FluxRegModel)
    penalty(reg_model.layer_mean) + penalty(reg_model.layer_var)
end

end
