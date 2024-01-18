using Flux
using Test
using Distributions

using FluxStats


# generate some toy data for n=3 and p=2
p = 2
n = 3
# dims: p x n
X_train = ones32(p, n)
X_val = ones32(p, n) .* 0.5f0
y_train = ones32(1, n)
y_val = ones32(1, n) .* 0.8f0

# Define model params
lambda = 0.1f0
scale_half_cauchy = 3f0
scale_half_norm = 5f0
n_iter = 10

# optimiser
optim = Flux.RMSProp(0.01)

# define a model
mean_model = Chain(
    FluxStats.CustomFluxLayers.ScaleMixtureDense(
        (p => 1);
        bias=true,
        lambda=lambda,
        prior_scale=Distributions.Truncated(Cauchy(0f0, scale_half_cauchy), 0f0, Inf32)
    )
)
var_model = Chain(
    (FluxStats.CustomFluxLayers.DensePrior(
        Flux.Dense((1 => 1), Flux.softplus; bias=false),
        Distributions.TruncatedNormal(0f0, scale_half_norm, 0f0, Inf32)
    ))
)

model = FluxStats.FunctionalFluxModel.FluxRegModel(
    mean_model,
    var_model
)


@testset "wrapper" begin
    results = FluxStats.model_train(
        (X_train, [1f0]),
        y_train;
        model=model,
        loss_function=FluxStats.Losses.negloglik_normal,
        optim=optim,
        n_iter=n_iter,
        X_val=(X_val, [1f0]),
        y_val=y_val,
        track_weights=false
    )

    @test typeof(results) <: Dict
    @test length(keys(results)) == 4
    @test "train_loss" in keys(results)
    @test "val_loss" in keys(results)
    @test "model" in keys(results)
    @test "dict_weights" in keys(results)

    @test length(results["train_loss"]) == n_iter
    @test length(results["val_loss"]) == n_iter
    
end
