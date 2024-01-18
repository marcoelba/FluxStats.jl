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

# -------------- weights update --------------
epoch_1 = 1
epoch_2 = 2

@testset "dict_weights_update_Dense" begin
    p = 3
    dd = Dense(p, 1, bias=[10f0])
    w_dict = WeightTracking.weight_container_init(dd, n_iter=n_iter)
    dim_dict = WeightTracking.container_dim_init(dd)

    # update
    WeightTracking.weight_tracking_push!(epoch_1, dd, w_dict, dim_dict)
    WeightTracking.weight_tracking_push!(epoch_2, dd, w_dict, dim_dict)

    @test typeof(w_dict) <: Dict
    @test length(w_dict) == 2
    @test sum(isapprox.(w_dict["1"][:, :, epoch_1], dd.weight)) == 3
    @test w_dict["2"][epoch_1] == 10f0
    @test sum(isapprox.(w_dict["1"][:, :, epoch_2], dd.weight)) == 3
    @test w_dict["2"][epoch_2] == 10f0

end

@testset "dict_weights_update_Chain" begin
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
    w_dict = WeightTracking.weight_container_init(chain, n_iter=n_iter)
    dim_dict = WeightTracking.container_dim_init(chain)

    # update
    WeightTracking.weight_tracking_push!(epoch_1, chain, w_dict, dim_dict)
    WeightTracking.weight_tracking_push!(epoch_2, chain, w_dict, dim_dict)

    @test typeof(w_dict) <: Dict
    @test length(w_dict) == 2
    @test sum(isapprox.(w_dict["Dense"]["1"][:, :, epoch_1], l1.weight)) == 6
    @test sum(isapprox.(w_dict["Dense"]["1"][:, :, epoch_2], l1.weight)) == 6
    @test sum(isapprox.(w_dict["DensePrior"]["1"][:, :, epoch_1], l2.dense_layer.weight)) == 2
    @test sum(isapprox.(w_dict["DensePrior"]["1"][:, :, epoch_2], l2.dense_layer.weight)) == 2
end

@testset "dict_weights_update_RegModel" begin
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

    w_dict = WeightTracking.weight_container_init(model, n_iter=n_iter)
    dim_dict = WeightTracking.container_dim_init(model)

    # update
    WeightTracking.weight_tracking_push!(epoch_1, model, w_dict, dim_dict)
    WeightTracking.weight_tracking_push!(epoch_2, model, w_dict, dim_dict)

    @test typeof(w_dict) <: Dict
    @test length(w_dict) == 2

    @test "chain_mean" in keys(w_dict)
    @test "chain_var" in keys(w_dict)

    @test "ScaleMixtureDense" in keys(w_dict["chain_mean"])
    @test "DensePrior" in keys(w_dict["chain_var"])

    @test sum(isapprox.(w_dict["chain_mean"]["ScaleMixtureDense"]["1"][:, :, epoch_1], mean_model[1].weight)) == 3
    @test sum(isapprox.(w_dict["chain_mean"]["ScaleMixtureDense"]["1"][:, :, epoch_2], mean_model[1].weight)) == 3

    @test sum(isapprox.(w_dict["chain_var"]["DensePrior"]["1"][:, :, epoch_1], var_model[1].dense_layer.weight)) == 1
    @test sum(isapprox.(w_dict["chain_var"]["DensePrior"]["1"][:, :, epoch_2], var_model[1].dense_layer.weight)) == 1
end
