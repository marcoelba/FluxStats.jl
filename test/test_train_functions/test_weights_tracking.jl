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

    @test typeof(output) <: Dict
    @test length(output) == 2
    @test size(output["1"]) == (1, p, n_iter)
    @test typeof(output["1"]) <: Array{Float32}
    @test size(output["2"]) == (1, n_iter)
    @test typeof(output["2"]) <: Array{Float32}
end


@testset "dim_container_Dense" begin
    p = 3
    dd = Dense(p, 1)
    output = WeightTracking.container_dim_init(dd)

    @test typeof(output) <: Dict
    @test length(output) == 2
    
    @test typeof(output["1"]) == Tuple{Colon, Colon}
    @test typeof(output["2"]) == Tuple{Colon}
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

    @test typeof(output) <: Dict
    @test length(output) == 2

    @test name_l1 in keys(output)
    @test name_l2 in keys(output)
end

@testset "dim_container_Chain" begin
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

    @test typeof(output) <: Dict
    @test length(output) == 2

    @test name_l1 in keys(output)
    @test name_l2 in keys(output)
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

    @test typeof(output) <: Dict
    @test length(output) == 3

    @test size(output["1"]) == (1, p, n_iter)
    @test typeof(output["1"]) <: Array{Float32}
    @test size(output["2"]) == (1, p, n_iter)
    @test typeof(output["2"]) <: Array{Float32}
    @test size(output["3"]) == (1, n_iter)
    @test typeof(output["3"]) <: Array{Float32}

    output_dim = WeightTracking.container_dim_init(sm)

    @test typeof(output_dim) <: Dict
    @test length(output_dim) == 3

    @test typeof(output_dim["1"]) == Tuple{Colon, Colon}
    @test typeof(output_dim["2"]) == Tuple{Colon, Colon}
    @test typeof(output_dim["3"]) == Tuple{Colon}
end


@testset "weight_tracking_DensePrior" begin
    p = 1
    dp = FluxStats.CustomFluxLayers.DensePrior(
        Flux.Dense((p => 1), Flux.softplus; bias=false),
        Distributions.TruncatedNormal(0f0, 3f0, 0f0, Inf32)
    )
    output = WeightTracking.weight_container_init(dp, n_iter=n_iter)

    @test typeof(output) <: Dict
    @test length(output) == 1

    @test size(output["1"]) == (1, p, n_iter)
    @test typeof(output["1"]) <: Array{Float32}

    output_dim = WeightTracking.container_dim_init(dp)

    @test typeof(output_dim) <: Dict
    @test length(output_dim) == 1
    @test typeof(output_dim["1"]) == Tuple{Colon, Colon}
end


@testset "flux_reg_model" begin
    p = 3
    mean_model = Chain(
        FluxStats.CustomFluxLayers.ScaleMixtureDense(
            (p => 1);
            bias=true,
            lambda=1f0,
            prior_scale=Distributions.Truncated(Cauchy(0f0, 1f0), 0f0, Inf32)
        )
    )
    var_model = Chain(
        (FluxStats.CustomFluxLayers.DensePrior(
            Flux.Dense((1 => 1), Flux.softplus; bias=false),
            Distributions.TruncatedNormal(0f0, 1f0, 0f0, Inf32)
        ))
    )

    model = FluxStats.FunctionalFluxModel.FluxRegModel(
        mean_model,
        var_model
    )

    output = WeightTracking.weight_container_init(model, n_iter=n_iter)

    @test typeof(output) <: Dict
    @test "chain_mean" in keys(output)
    @test "chain_var" in keys(output)

    @test "ScaleMixtureDense" in keys(output["chain_mean"])
    @test "DensePrior" in keys(output["chain_var"])

    output_dim = WeightTracking.container_dim_init(model)

    @test typeof(output) <: Dict
    @test "chain_mean" in keys(output)
    @test "chain_var" in keys(output)

    @test "ScaleMixtureDense" in keys(output["chain_mean"])
    @test "DensePrior" in keys(output["chain_var"])

end
