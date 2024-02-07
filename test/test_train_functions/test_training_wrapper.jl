using Flux
using Test
using Distributions

using FluxStats


# generate some toy data for n=3 and p=2
p = 2
n = 3
# dims: p x n
X_train = Float32.(vcat([1;; 2;; -1], [1;; 1;; 1]))
X_val = ones32(p, n) .* 0.5f0
y_train = ones32(1, n)
y_val = ones32(1, n) .* 0.8f0

# Define model params
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
        prior_scale=Distributions.truncated(Cauchy(0f0, scale_half_cauchy), 0f0, Inf32)
    )
)
var_model = Chain(
    (FluxStats.CustomFluxLayers.DensePrior(
        Flux.Dense((1 => 1), Flux.softplus; bias=false),
        Distributions.truncated(Normal(0f0, scale_half_norm), 0f0, Inf32)
    ))
)

model = FluxStats.FunctionalFluxModel.FluxRegModel(
    mean_model,
    var_model
)


@testset "training_wrapper" begin
    results = FluxStats.model_train(
        X_train,
        y_train;
        model=model,
        loss_function=FluxStats.Losses.gaussian_negloglik,
        optim=optim,
        n_iter=n_iter,
        X_val=X_val,
        y_val=y_val,
        track_weights=false
    )

    @test typeof(results) <: Dict
    @test length(keys(results)) == 4
    @test "train_loss" in keys(results)
    @test "val_loss" in keys(results)
    @test "model" in keys(results)
    @test "dict_weights" in keys(results)
    @test isnothing(results["dict_weights"])

    @test length(results["train_loss"]) == n_iter
    @test length(results["val_loss"]) == n_iter


    # with weights tracking
    results = FluxStats.model_train(
        X_train,
        y_train;
        model=model,
        loss_function=FluxStats.Losses.gaussian_negloglik,
        optim=optim,
        n_iter=n_iter,
        X_val=X_val,
        y_val=y_val,
        track_weights=true
    )

    @test typeof(results) <: Dict
    @test length(keys(results)) == 4
    @test "train_loss" in keys(results)
    @test "val_loss" in keys(results)
    @test "model" in keys(results)
    @test "dict_weights" in keys(results)
    @test typeof(results["dict_weights"]) <: Dict
    
    @test length(results["train_loss"]) == n_iter
    @test length(results["val_loss"]) == n_iter

end

@testset "test only model prediction" begin
    # set specific parameters
    model.chain_mean[1].weight .= [1f0;; 2f0]
    model.chain_mean[1].scale .= [0.5f0;; 1.5f0]
    model.chain_mean[1].bias .= [1f0]

    model.chain_var[1].dense_layer.weight .= 1.5f0

    model_pred = model(X_train)
    tot1 = FluxStats.Losses.negloglik(y_train, model_pred, FluxStats.Losses.gaussian_negloglik, sum)

    model_pred = ([1f0;; 1f0;; 1f0], [1f0;; 1f0;; 1f0])
    tot2 = FluxStats.Losses.negloglik(y_train, model_pred, FluxStats.Losses.gaussian_negloglik, sum)

    @test tot2 <= tot1

    # test if sum of single prediction losses is equal to the total
    y_pred = [1f0;; 2f0;; 3f0]
    sum_ind = 0f0
    for (pos, y) in enumerate(y_pred)
        model_pred = ([y], [1f0])
        sum_ind += FluxStats.Losses.negloglik([1f0], model_pred, FluxStats.Losses.gaussian_negloglik, sum)
    end

    model_pred = (y_pred, [1f0;; 1f0;; 1f0])
    sum_tot = FluxStats.Losses.negloglik([1f0;; 1f0;; 1f0], model_pred, FluxStats.Losses.gaussian_negloglik, sum)

    @test round(sum_ind, digits=5) == round(sum_tot, digits=5)
    
end
