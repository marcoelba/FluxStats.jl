module Losses

using Zygote
using Flux
using Distributions


"""
    negloglik(x::Array{Float32}, model_predictions::Array{Float32}, negloglik_function)

    Compute the sum negative loglikelihood (or loss) over all provided input.

    # Arguments
        negloglik_function: must have the same number of input parameters as the number of output from the model PLUS the true label

    # Examples
        ```jldoctest
        julia>
            x = Float32.([1, 2, -1])
            mu = Float32.([1, 0, 0])
            sigma = Float32.([2, 1, 2])

            model_predictions = (mu, sigma)
            FluxStats.Losses.negloglik(x, model_predictions, FluxStats.Losses.negloglik_normal)
        6.2681103f0
        ```
"""
function negloglik(x::Array{Float32}, model_predictions::Array{Float32}, negloglik_function)
    negloglik_function(x, model_predictions)
end

function negloglik(x::Array{Float32}, model_predictions::Tuple, negloglik_function)
    negloglik_function(x, model_predictions...)
end

"""
        gaussian_negloglik(x::Array{Float32}, mu::Array{Float32}, sigma::Array{Float32}=Flux.ones32(size(mu)...))

    Compute the Gaussian negative loglikelihood over all provided input.

    # Examples
        ```jldoctest
        julia>
            x = Float32.([1, 2, -1])
            mu = Float32.([1, 0, 0])
            sigma = Float32.([2, 1, 2])

            FluxStats.Losses.negloglik(x, mu, sigma)
        6.2681103f0
        ```
"""
function gaussian_negloglik(x::Array{Float32}, mu::Array{Float32}, sigma::Array{Float32}=Flux.ones32(size(mu)...))
    distribution = Distributions.Normal{Float32}.(mu, sigma)
    -mean(Distributions.logpdf.(distribution, x))
end


"""
    Huber loss function
"""
function huber(x, scale=1f0; delta::Float32=1.f0)
    abs_x = abs.(x / scale)
    temp = Zygote.ignore_derivatives(abs_x .<  delta)
    return ((abs_x .* abs_x) * 0.5f0) * temp + (delta * (abs_x - 0.5f0 * delta)) * (1.f0 - temp)
end


end
