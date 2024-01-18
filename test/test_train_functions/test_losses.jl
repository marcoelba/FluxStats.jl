using FluxStats
using Test


# generate some toy data
x = Float32.([1, 2, -1])
mu = Float32.([1, 0, 0])
sigma = Float32.([2, 1, 2])

@testset "gaussian loglik" begin
    @test round(FluxStats.Losses.negloglik_normal(x, mu, sigma), digits=3) == 6.268f0
    @test round(FluxStats.Losses.negloglik_normal(x, mu), digits=3) == 5.257f0
end


@testset "negloglik" begin
    # if Tuple
    model_predictions = (mu, sigma)
    @test round(FluxStats.Losses.negloglik(x, model_predictions, FluxStats.Losses.negloglik_normal), digits=3) == 6.268f0

    # If Array, only mu, sigma default to 1.
    model_predictions = mu
    @test round(FluxStats.Losses.negloglik(x, model_predictions, FluxStats.Losses.negloglik_normal), digits=3) == 5.257f0
end
