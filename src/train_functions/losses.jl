module Losses

using Zygote
using Flux
using Distributions


"""
    negloglik(x::Array{Float32}, model_predictions::Array{Float32}, negloglik_function, aggregation_function)

    Compute the sum negative loglikelihood (or loss) over all provided input.

    # Arguments
        negloglik_function: must have the same number of input parameters as the number of output from the model PLUS the true label
        aggregation_function: how to aggregate the loss values over batch (eg sum, mean)

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
function negloglik(x::Array{Float32}, model_predictions::Array{Float32}, negloglik_function, aggregation_function)
    aggregation_function(negloglik_function(x, model_predictions))
end

function negloglik(x::Array{Float32}, model_predictions::Tuple, negloglik_function, aggregation_function)
    aggregation_function(negloglik_function(x, model_predictions...))
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
function gaussian_negloglik(
    x::Array{Float32},
    mu::Array{Float32},
    sigma::Array{Float32}=Flux.ones32(size(mu)...)
    )
    distribution = Distributions.Normal{Float32}.(mu, sigma)
    -Distributions.logpdf.(distribution, x)
end


"""
    Huber loss function
"""
function huber(x, scale=1f0; delta::Float32=1.f0)
    abs_x = abs.(x / scale)
    temp = Zygote.ignore_derivatives(abs_x .<  delta)
    return ((abs_x .* abs_x) * 0.5f0) * temp + (delta * (abs_x - 0.5f0 * delta)) * (1.f0 - temp)
end


"""
    logpdf_truncated_normal(x::Array{Float32}, mu::Float32=0f0, sd::Float32=1f0, t::Float32=0.5f0)
"""
function logpdf_truncated_normal(x::Array{Float32}, mu::Float32=0f0, sd::Float32=1f0, t::Float32=0.5f0)
    -0.5f0 .* log(2f0*pi) - log(sd) .- 0.5f0 .* ((x .- mu) ./ sd).^2f0 .- log.(t)
end


"""
    logpdf_truncated_mixture_normal(x::Array{Float32}, mu::Float32=0f0, sd::Float32=1f0, t::Float32=0.5f0)
"""
function logpdf_truncated_mixture_normal(
    x::Matrix{Float32};
    w::Vector{Float32}=Float32.([0.5, 0.25, 0.25]),
    mu::Vector{Float32}=Float32.([0, -1, 1]),
    sd::Vector{Float32}=Float32.([0.1, 0.5, 0.5]),
    t::Vector{Float32}=Float32.([1, 0.5, 0.5])
    )
    xstd = -0.5f0 .* ((x .- mu) ./ sd).^2f0
    wstd = w ./ (sqrt(2f0 .* pi) .* sd) ./ t
    offset = maximum(xstd .* wstd, dims=1)
    xe = exp.(xstd .- offset)
    s = sum(xe .* wstd, dims=1)
    log.(s) .+ offset
end


end
