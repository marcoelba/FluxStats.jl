module Losses

using Zygote
using Flux
using Distributions


"""
    Guassian negative log-likelihood
"""
function neg_gaussian_loglik(y_true, y_pred; sigma2_y=1f0)
    y_res = y_true .- y_pred
    y_std = (y_res .* y_res) ./ sigma2_y

    0.5 * mean(y_std .+ log.(2f0 * Float32(pi)) .+ log(sigma2_y))
end


"""
    Bernoulli negative log-likelihood
"""
function bernoulli_loglik(ytrue, yhat)
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

end # module
