# Examples Flux Regression Model with unknown variance
using Plots
using Flux
using Random
using Distributions

# Pkg.precompile()
# Pkg.resolve()
using FluxStats
using FluxStats: Penalties, Losses, WeightTracking


# Data generation
n = 300
p = 10
prop_non_zero = 0.5
cor_x = 0.

Random.seed!(124)
data = FluxStats.data_generation.linear_regression_data(
    n=n,
    p=p,
    sigma2=1.,
    beta_intercept=1.,
    correlation_coefficients=[cor_x],
    block_covariance=true,
    beta_pool=[-1., 1.],
    prop_zero_coef=1. - prop_non_zero
)

X = copy(transpose(data.X))
y = reshape(data.y, (1, size(data.y)...))

X = Float32.(X)
y = Float32.(y)

# train val split
y_train, y_val, X_train, X_val = FluxStats.utilities.train_val_split(X, y, train_prop=0.5)
n_train = length(y_train)


# -----------------------------------------------------
# Custom model with variance layer
# -----------------------------------------------------
lambda = 1f0
tau = 3f0
scale_half_norm = 5f0
n_iter = 1000

mean_model = Chain(
    FluxStats.CustomFluxLayers.ScaleMixtureDense(
        (p => 1);
        bias=true,
        lambda=lambda,
        prior_scale=Distributions.truncated(Cauchy(0f0, tau), 0f0, Inf32)
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

optim = Flux.RMSProp(0.01)

# run training
res = FluxStats.model_train(
    X_train,
    y_train;
    model=model,
    loss_function=FluxStats.Losses.gaussian_negloglik,
    optim=optim,
    n_iter=n_iter,
    aggregation_function=sum,
    X_val=X_val,
    y_val=y_val,
    track_weights=true
)


plot(res["train_loss"]; label="train_loss")
plot!(res["val_loss"]; label="val_loss")
# plot last part only
plot(res["train_loss"][500:n_iter]; label="train_loss")
plot!(res["val_loss"][500:n_iter]; label="val_loss")

println("Validation loss: ", res["val_loss"][n_iter] / (n - n_train))
println("Train loss: ", res["train_loss"][n_iter] / (n_train))

# If weights are tracked:
weights = res["dict_weights"]["chain_mean"]["ScaleMixtureDense"]["1"][1,:,:];
scales = Flux.softplus.(res["dict_weights"]["chain_mean"]["ScaleMixtureDense"]["2"][1,:,:])
sigma = Flux.softplus.(res["dict_weights"]["chain_var"]["DensePrior"]["1"][1,:,:])

plot(transpose(weights))
plot(transpose(scales))
plot(transpose(sigma))

# Optimised parameters
opt_weights = transpose(Flux.params(res["model"])[1])[:, 1]
opt_scales = transpose(Flux.softplus.(Flux.params(res["model"])[2]))[:, 1]
opt_bias = Flux.params(res["model"])[3]
opt_sigma = Flux.softplus.(Flux.params(res["model"])[4][1])

# inclusion probs
k_probs = FluxStats.utilities.shrinking_coeffs(
    lambda=opt_scales,
    tau=tau,
    n=n_train,
    sigma2_y=opt_sigma^2,
    var_x=1.
)
histogram(k_probs, bins=10)
