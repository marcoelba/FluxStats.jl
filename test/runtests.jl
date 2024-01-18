using FluxStats
using Test


@testset "FluxStats.jl" begin
    include(joinpath("test_train_functions", "test_losses.jl"))
    include(joinpath("test_train_functions", "test_training_wrapper.jl"))
    include(joinpath("test_train_functions", "test_weights_tracking.jl"))
end
