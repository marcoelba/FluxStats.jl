module FluxStats

include(joinpath("train_functions", "penalty_functions.jl"))
using .Penalties
export penalty

include(joinpath("train_functions", "losses.jl"))
include(joinpath("train_functions", "weights_tracking.jl"))

include(joinpath("flux_models", "custom_flux_layers.jl"))
include(joinpath("flux_models", "custom_flux_models.jl"))

include(joinpath("utilities", "utilities.jl"))
include(joinpath("utilities", "classification_metrics.jl"))
include(joinpath("utilities", "data_generation.jl"))

include(joinpath("train_functions", "training_wrapper.jl"))


end
