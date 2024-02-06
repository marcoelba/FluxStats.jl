# utilities
module Utilities

using Random
using StatsBase
using Distributions

"""
    Normalise data between a given min and max
"""
function normaliser(x::AbstractArray; a::Real=0., b::Real=1.)
    min_x = minimum(x)
    max_x = maximum(x)
    norm_x = (b - a) .* (x .- min_x) / (max_x - min_x) .+ a
    return norm_x
end


"""
    Get train, validation
"""
function train_val_split(X::Matrix{Float32}, y::Matrix{Float32}; train_prop::Real, observations_in_col::Bool=true)
    n = size(X, 2)
    n_train = floor(Int, n * train_prop)

    train_index = sample(range(1, n), n_train, replace=false)
    val_index = setdiff(range(1, n), train_index)

    return (y_train = y[:, train_index], y_val = y[:, val_index],
        X_train = X[:, train_index], X_val = X[:, val_index]
    )
end


"""
    shrinkage coefficients for a scale-mixture penalty function
"""
function shrinking_coeffs(;lambda, tau, n, sigma2_y=1., var_x=1.)
    pd = (n .* tau.^2 .* lambda.^2 .* var_x^2) ./ sigma2_y
    k_s = 1. .- 1. ./ (1 .+ pd)
    return k_s
end


"""
     weights initalisers
"""
function init_weights(in, out)
    distro = Distributions.Normal(0f0, 0.01f0)
    return rand(distro, (in, out))
end

function init_scales(in, out)
    distro = Distributions.Normal(-2f0, 0.05f0)
    return rand(distro, (in, out))
end


end
