module Losses

using Zygote
using Flux
using Distributions


"""
    Generic function for the model loss
    Takes as input the model predictions, which can consist of one or multiple elements,
    and a negative log likelihood as defined below.
"""
function negloglik(x::Array{Float32}, model_predictions::Array{Float32}, negloglik_function)
    negloglik_function(x, model_predictions)
end

function negloglik(x::Array{Float32}, model_predictions::Tuple, negloglik_function)
    negloglik_function(x, model_predictions...)
end

"""
    negloglik_normal(x::Array{Float32}, mu::Array{Float32}, sigma::Array{Float32}=[1f0])

    Guassian negative log-likelihood
"""
function negloglik_normal(x::Array{Float32}, mu::Array{Float32}, sigma::Array{Float32}=[1f0])
    distribution = Distributions.Normal{Float32}.(mu, sigma)
    -sum(Distributions.logpdf.(distribution, x))
end


"""
    Bernoulli negative log-likelihood
"""
function bernoulli_negloglik(ytrue, yhat)
    ytrue .* log(yhat) + (1 .- ytrue) .* log(1 - yhat)
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
    Log Laplace
"""
function log_laplace_kernel(x, scale=1f0)
    return abs(x) / scale + log(2 * scale)
end

"""
    Log Gaussian
"""
function log_gaussian_kernel(x, scale=1f0)
    x_std = x / scale
    return 0.5 * x_std * x_std + log(scale) + 0.5 * log(2 * Float32(pi))
end

"""
    Log Half-Gaussian
"""
function log_half_gaussian_penalty(x, sigma2)
    x_std = (x .* x) / sigma2
    return 0.5f0 * x_std + log(sigma) + 0.5f0 * log(Float32(pi)) - 0.5f0 * log(2f0)
end


end
