using Flux
using Test
using Distributions

using FluxStats
using FluxStats: WeightTracking


n_iter = 10

@testset "weight_tracking_Dense" begin
    p = 3
    dd = Dense(p, 1)
    output = WeightTracking.weight_container_init(dd, n_iter=n_iter)

    @test typeof(output) <: Tuple
    @test length(output) == 2
    @test length(output[1]) == 2
    @test size(output[1]["1"]) == (1, p, n_iter)
    @test typeof(output[1]["1"]) <: Array{Float32}
    @test size(output[1]["2"]) == (1, n_iter)
    @test typeof(output[1]["2"]) <: Array{Float32}
    
    @test typeof(output[2]["1"]) == Tuple{Colon, Colon}
    @test typeof(output[2]["2"]) == Tuple{Colon}
end


@testset "weight_tracking_Chain" begin
    p1 = 3
    p2 = 2
    l1 = Dense(p1, p2)
    name_l1 = "Dense"
    l2 = FluxStats.CustomFluxLayers.DensePrior(
        Flux.Dense((p2 => 1), Flux.softplus; bias=false),
        Distributions.TruncatedNormal(0f0, 3f0, 0f0, Inf32)
    )
    name_l2 = "DensePrior"
    chain = Flux.Chain(l1, l2)
    output = WeightTracking.weight_container_init(chain, n_iter=n_iter)

    @test typeof(output) <: Tuple
    @test length(output) == 2

    @test length(output[1]) == 2
    @test name_l1 in keys(output[1])
    @test name_l2 in keys(output[1])

    @test length(output[2]) == 2
    @test name_l1 in keys(output[2])
    @test name_l2 in keys(output[2])

end


@testset "weight_tracking_ScaleMixture" begin
    p = 3
    sm = FluxStats.CustomFluxLayers.ScaleMixtureDense(
        (p => 1);
        bias=true,
        lambda=1f0,
        prior_scale=Distributions.Truncated(Cauchy(0f0, 3f0), 0f0, Inf32)
    )
    output = WeightTracking.weight_container_init(sm, n_iter=n_iter)

    @test typeof(output) <: Tuple
    @test length(output) == 2

    @test length(output[1]) == 3
    @test size(output[1]["1"]) == (1, p, n_iter)
    @test typeof(output[1]["1"]) <: Array{Float32}
    @test size(output[1]["2"]) == (1, p, n_iter)
    @test typeof(output[1]["2"]) <: Array{Float32}
    @test size(output[1]["3"]) == (1, n_iter)
    @test typeof(output[1]["3"]) <: Array{Float32}
    
    @test typeof(output[2]["1"]) == Tuple{Colon, Colon}
    @test typeof(output[2]["2"]) == Tuple{Colon, Colon}
    @test typeof(output[2]["3"]) == Tuple{Colon}
end


@testset "weight_tracking_DensePrior" begin
    p = 1
    dp = FluxStats.CustomFluxLayers.DensePrior(
        Flux.Dense((p => 1), Flux.softplus; bias=false),
        Distributions.TruncatedNormal(0f0, 3f0, 0f0, Inf32)
    )
    output = WeightTracking.weight_container_init(dp, n_iter=n_iter)

    @test typeof(output) <: Tuple
    @test length(output) == 2

    @test length(output[1]) == 1
    @test size(output[1]["1"]) == (1, p, n_iter)
    @test typeof(output[1]["1"]) <: Array{Float32}
    
    @test typeof(output[2]["1"]) == Tuple{Colon, Colon}
end
