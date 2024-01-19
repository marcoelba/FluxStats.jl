using FluxStats
using Test


# generate some toy data
x = Float32.([1, 2, -1])
mu = Float32.([1, 0, 0])
sigma = Float32.([2, 1, 2])
x_t = collect(transpose(x))
mu_t = collect(transpose(mu))
sigma_t = collect(transpose(sigma))

@testset "gaussian loglik" begin
    @test round(FluxStats.Losses.gaussian_negloglik(x, mu, sigma), digits=3) == 6.268f0
    @test round(FluxStats.Losses.gaussian_negloglik(x, mu), digits=3) == 5.257f0

    @test round(FluxStats.Losses.gaussian_negloglik(x_t, mu_t, sigma_t), digits=3) == 6.268f0
    @test round(FluxStats.Losses.gaussian_negloglik(x_t, mu_t), digits=3) == 5.257f0
    @test round(FluxStats.Losses.gaussian_negloglik(x_t, mu_t, [1f0;;]), digits=3) == 5.257f0

end


@testset "negloglik" begin
    # if Tuple
    model_predictions = (mu, sigma)
    @test round(FluxStats.Losses.negloglik(x, model_predictions, FluxStats.Losses.gaussian_negloglik), digits=3) == 6.268f0
    model_predictions = (mu_t, sigma_t)
    @test round(FluxStats.Losses.negloglik(x_t, model_predictions, FluxStats.Losses.gaussian_negloglik), digits=3) == 6.268f0

    # If Array, only mu, sigma default to 1.
    model_predictions = mu
    @test round(FluxStats.Losses.negloglik(x, model_predictions, FluxStats.Losses.gaussian_negloglik), digits=3) == 5.257f0
    model_predictions = mu_t
    @test round(FluxStats.Losses.negloglik(x_t, model_predictions, FluxStats.Losses.gaussian_negloglik), digits=3) == 5.257f0

end
