# Custom Flux Layers
module CustomFluxLayers

using Flux
using Distributions
using Zygote

# Internal
using FluxStats
using FluxStats: Penalties, WeightTracking


"""
    MyScale(
        s1::Integer;
        bias=false, init=ones32, _act=identity, lambda=0.f0
    )

    Custom Flux Scale layer
    Apply sigmoid transformation to the scale weights and 
    add a penalty (L1) to the sum of the weights
"""
struct MyScale{A<:AbstractArray, B, F, L<:Union{Float64, Float32}}
    scale::A
    bias::B
    activation::F
    lambda::L

    function MyScale(scale::A, bias::B = false, activation::F = identity, lambda::L = 0.f0) where {A<:AbstractArray, B<:Union{Bool, AbstractArray}, F, L}
        b = Flux.create_bias(scale, bias, size(scale)...)
        new{A, typeof(b), F, L}(scale, b, activation, lambda)
    end
end

MyScale(
    s1::Integer;
    bias=false, init=ones32, _act=identity, lambda=0.f0
) = MyScale(init(s1), bias, _act, lambda)

Flux.@functor MyScale

function (l::MyScale)(x::AbstractArray)
    scale_transorm = Flux.sigmoid_fast.(l.scale)
    scale_transorm .* x .+ l.bias
end

function Base.show(io::IO, l::MyScale)
    println(io, "MyScale(")
    println(io, "\t", join(size(l.scale), ","))
    l.activation == identity || println(io, "\t", l.activation, ";")
    l.bias == false && println(io, "\t", "bias=false")
    print(io, ")")
end

# Add penalty for MyScale
function Penalties.penalty(l::MyScale)
    l.lambda * FluxStats.Losses.huber(sum(Flux.sigmoid_fast.(l.scale)) - 1.f0)
end


"""
    ElementwiseScale(
        p::Integer;
        init=ones32,
        activation=identity,
        prior_scale=Distributions.truncated(Distributions.Cauchy(0f0, 1f0), 0f0, Inf32)
    )

"""
struct ElementwiseScale{M<:AbstractArray, F, P<:Distributions.Distribution}
    scale::M
    activation::F
    prior_scale::P
end

ElementwiseScale(
    p::Integer;
    init=ones32,
    activation=identity,
    prior_scale=Distributions.truncated(Distributions.Cauchy(0f0, 1f0), 0f0, Inf32)
) = ElementwiseScale(init(p, 1), activation, prior_scale)

Flux.@functor ElementwiseScale

function (l::ElementwiseScale)(x::AbstractArray)
    scale_pos = Flux.softplus.(l.scale)
    x_scaled = scale_pos .* x
    return (x_scaled, scale_pos)
end

function Base.show(io::IO, l::ElementwiseScale)
    println(io, "ElementwiseScale(")
    println(io, "\t", size(l.scale, 2), ";")
    println(io, "\t", "activation = ", l.activation, ",")
    println(io, "\t", "prior_scale = ", l.prior_scale, ",")
    print(io, ")")
end

# Add penalty for MyScale
function Penalties.penalty(l::ElementwiseScale)
    -sum(Distributions.logpdf.(l.prior_scale, l.scale_pos))
end


"""
    ScaleMixtureDense(
        (in, out)::Pair{<:Integer, <:Integer};
        bias=true,
        activation=identity,
        init=Flux.glorot_normal,
        lambda=0f0,
        prior_scale=Distributions.TruncatedNormal(0f0, 10f0, 0f0, Inf32)
    )

    Custom Flux Scale-Mixture Dense layer
    Works like a standard Dense layer, but creates a set of additional scale coefficients (one for each weight),
        which are used in the weights loss function.
"""
struct ScaleMixtureDense{M <: AbstractMatrix, B, F, L<:Float32, P<:Distributions.Distribution}
    weight::M
    scale::M
    bias::B
    activation::F
    lambda::L
    prior_scale::P

    function ScaleMixtureDense(
        W::M,
        S::M,
        bias::B = true,
        activation::F = identity,
        lambda::L = 0f0,
        prior_scale::P = Distributions.truncated(Distributions.Normal(0f0, 10f0), 0f0, Inf32)
    ) where {M <: AbstractMatrix, B<:Union{Bool, AbstractArray}, F, L, P}
        b = Flux.create_bias(W, bias, size(W, 1))
        new{M, typeof(b), F, L, P}(W, S, b, activation, lambda, prior_scale)
    end
end

ScaleMixtureDense(
    (in, out)::Pair{<:Integer, <:Integer};
    bias=true,
    activation=identity,
    init=Flux.glorot_normal,
    lambda=0f0,
    prior_scale=Distributions.truncated(Distributions.Normal(0f0, 10f0), 0f0, Inf32)
) = ScaleMixtureDense(init(out, in), init(out, in), bias, activation, lambda, prior_scale)

Flux.@functor ScaleMixtureDense

function (l::ScaleMixtureDense)(x::AbstractVecOrMat)
    Flux._size_check(l, x, 1 => size(l.weight, 2))
    xT = Flux._match_eltype(l, x)
    return l.activation(l.weight * xT .+ l.bias)
end

function Base.show(io::IO, l::ScaleMixtureDense)
    println(io, "ScaleMixtureDense(")
    println(io, "\t", size(l.weight, 2), " => ", size(l.weight, 1), ";")
    println(io, "\t", "activation = ", l.activation, ",")
    println(io, "\t", "bias = ", l.bias, ",")
    println(io, "\t", "lambda = ", l.lambda, ",")
    println(io, "\t", "prior_scale = ", l.prior_scale, ",")
    print(io, ")")
end

# Add penalty for ScaleMixtureDense layer - Horseshoe hierachical structure here
function Penalties.penalty(l::ScaleMixtureDense)
    # transform the scale parameters first
    scale = Flux.softplus.(l.scale)
    prior_weight = Distributions.Normal.(0f0, scale)

    -l.lambda * (
        sum(Distributions.logpdf.(l.prior_scale, scale)) +
        sum(Distributions.logpdf.(prior_weight, l.weight))
    )
end

# Extend weights tracking function
function WeightTracking.weight_container_init(layer::ScaleMixtureDense; n_iter::Int64)
    w_dict = Dict()
    for (pos, param) in enumerate(Flux.params(layer))
        param_size = size(param)
        w_dict[string(pos)] = zeros32(param_size..., n_iter)
    end

    return w_dict
end

function WeightTracking.weight_tracking_push!(epoch::Int64, layer::ScaleMixtureDense, dict_weights_layer::Dict, dict_dims_layer::Dict)
    for (pos, param) in enumerate(Flux.params(layer))
        dict_weights_layer[string(pos)][dict_dims_layer[string(pos)]..., epoch] = param
    end
end

function WeightTracking.container_dim_init(layer::ScaleMixtureDense)
    dim_dict = Dict()
    for (pos, param) in enumerate(Flux.params(layer))
        param_size = size(param)
        dim_dict[string(pos)] = ntuple(_ -> (:), length(param_size))
    end

    return dim_dict
end


"""
    NonCentredScaleMixtureDense(
        (in, out)::Pair{<:Integer, <:Integer};
        bias=true,
        activation=identity,
        init=Flux.glorot_normal,
        lambda=0f0,
        prior_scale=Distributions.TruncatedNormal(0f0, 10f0, 0f0, Inf32)
    )

    Custom Flux Scale-Mixture Dense layer
    Works like a standard Dense layer, but creates a set of additional scale coefficients (one for each weight),
        which are used in the weights loss function.
"""
struct NonCentredScaleMixtureDense{
    M <: AbstractMatrix,
    B, F, L<:Float32,
    P_S <: Distributions.Distribution,
    P_W <: Distributions.Distribution}
    weight::M
    scale::M
    bias::B
    activation::F
    lambda::L
    prior_scale::P_S
    prior_weight::P_W

    function NonCentredScaleMixtureDense(
        W::M,
        S::M,
        bias::B = true,
        activation::F = identity,
        lambda::L = 1f0,
        prior_scale::P_S = Distributions.truncated(Distributions.Cauchy(0f0, 1f0), 0f0, Inf32),
        prior_weight::P_W = Distributions.Normal(0f0, 1f0)
    ) where {M <: AbstractMatrix, B<:Union{Bool, AbstractArray}, F, L,
        P_S <: Distributions.Distribution,
        P_W <: Distributions.Distribution
    }
        b = Flux.create_bias(W, bias, size(W, 1))
        new{M, typeof(b), F, L, P_S, P_W}(W, S, b, activation, lambda, prior_scale, prior_weight)
    end
end

NonCentredScaleMixtureDense(
    (in, out)::Pair{<:Integer, <:Integer};
    bias=true,
    activation=identity,
    init=Flux.glorot_normal,
    lambda=1f0,
    prior_scale=Distributions.truncated(Distributions.Cauchy(0f0, 5f0), 0f0, Inf32),
    prior_weight=Distributions.Normal(0f0, 1f0)
) = NonCentredScaleMixtureDense(
    init(out, in),
    init(out, in),
    bias,
    activation,
    lambda,
    prior_scale,
    prior_weight
)

Flux.@functor NonCentredScaleMixtureDense
Flux.trainable(l::NonCentredScaleMixtureDense) = (weight = l.weight, scale = l.scale, bias=l.bias)

function (l::NonCentredScaleMixtureDense)(x::AbstractVecOrMat)
    Flux._size_check(l, x, 1 => size(l.weight, 2))
    xT = Flux._match_eltype(l, x)
    # scale and tau must be positive
    scale_t = Flux.softplus.(l.scale)
    beta = l.weight .* scale_t
    return l.activation(beta * xT .+ l.bias)
end

function Base.show(io::IO, l::NonCentredScaleMixtureDense)
    println(io, "NonCentredScaleMixtureDense(")
    println(io, "\t", size(l.weight, 2), " => ", size(l.weight, 1), ";")
    println(io, "\t", "activation = ", l.activation, ",")
    println(io, "\t", "bias = ", l.bias, ",")
    println(io, "\t", "lambda = ", l.lambda, ",")
    println(io, "\t", "prior_scale = ", l.prior_scale, ",")
    println(io, "\t", "prior_weight = ", l.prior_weight, ",")
    print(io, ")")
end

# Add penalty for NonCentredScaleMixtureDense layer - Horseshoe hierachical structure here
function Penalties.penalty(l::NonCentredScaleMixtureDense)
    scale_t = Flux.softplus.(l.scale)

    -l.lambda * (
        sum(Distributions.logpdf.(l.prior_scale, scale_t)) +
        sum(Distributions.logpdf.(l.prior_weight, l.weight))
    )
end

# Extend weights tracking function
function WeightTracking.weight_container_init(layer::NonCentredScaleMixtureDense; n_iter::Int64)
    w_dict = Dict()
    for (pos, param) in enumerate(Flux.params(layer))
        param_size = size(param)
        w_dict[string(pos)] = zeros32(param_size..., n_iter)
    end

    return w_dict
end

function WeightTracking.weight_tracking_push!(epoch::Int64, layer::NonCentredScaleMixtureDense, dict_weights_layer::Dict, dict_dims_layer::Dict)
    for (pos, param) in enumerate(Flux.params(layer))
        dict_weights_layer[string(pos)][dict_dims_layer[string(pos)]..., epoch] = param
    end
end

function WeightTracking.container_dim_init(layer::NonCentredScaleMixtureDense)
    dim_dict = Dict()
    for (pos, param) in enumerate(Flux.params(layer))
        param_size = size(param)
        dim_dict[string(pos)] = ntuple(_ -> (:), length(param_size))
    end

    return dim_dict
end


"""
NonCentredTauScaleMixtureDense(
        (in, out)::Pair{<:Integer, <:Integer};
        bias=true,
        activation=identity,
        init=Flux.glorot_normal,
        lambda=0f0,
        prior_scale=Distributions.TruncatedNormal(0f0, 10f0, 0f0, Inf32)
    )

    Custom Flux Scale-Mixture Dense layer
    Works like a standard Dense layer, but creates a set of additional scale coefficients (one for each weight),
        which are used in the weights loss function.
"""
struct NonCentredTauScaleMixtureDense{
    M <: AbstractMatrix,
    B, F, L<:Float32,
    P_S <: Distributions.Distribution,
    P_T <: Distributions.Distribution,
    P_W <: Distributions.Distribution}
    weight::M
    scale::M
    tau::M
    bias::B
    activation::F
    lambda::L
    prior_scale::P_S
    prior_tau::P_T
    prior_weight::P_W

    function NonCentredTauScaleMixtureDense(
        W::M,
        S::M,
        T::M,
        bias::B = true,
        activation::F = identity,
        lambda::L = 1f0,
        prior_scale::P_S = Distributions.truncated(Distributions.Cauchy(0f0, 5f0), 0f0, Inf32),
        prior_tau::P_T = Distributions.truncated(Distributions.Cauchy(0f0, 5f0), 0f0, Inf32),
        prior_weight::P_W = Distributions.Normal(0f0, 1f0)
    ) where {M <: AbstractMatrix, B<:Union{Bool, AbstractArray}, F, L,
        P_S <: Distributions.Distribution,
        P_T <: Distributions.Distribution,
        P_W <: Distributions.Distribution
    }
        b = Flux.create_bias(W, bias, size(W, 1))
        new{M, typeof(b), F, L, P_S, P_T, P_W}(W, S, T, b, activation, lambda, prior_scale, prior_tau, prior_weight)
    end
end

NonCentredTauScaleMixtureDense(
    (in, out)::Pair{<:Integer, <:Integer};
    bias=true,
    activation=identity,
    init=Flux.glorot_normal,
    lambda=1f0,
    prior_scale=Distributions.truncated(Distributions.Cauchy(0f0, 5f0), 0f0, Inf32),
    prior_tau=Distributions.truncated(Distributions.Cauchy(0f0, 5f0), 0f0, Inf32),
    prior_weight=Distributions.Normal(0f0, 1f0)
) = NonCentredTauScaleMixtureDense(
    init(out, in),
    init(out, in),
    [1f0;;],
    bias,
    activation,
    lambda,
    prior_scale,
    prior_tau,
    prior_weight
)

Flux.@functor NonCentredTauScaleMixtureDense
Flux.trainable(l::NonCentredTauScaleMixtureDense) = (weight = l.weight, scale = l.scale, tau = l.tau, bias=l.bias)

function (l::NonCentredTauScaleMixtureDense)(x::AbstractVecOrMat)
    Flux._size_check(l, x, 1 => size(l.weight, 2))
    xT = Flux._match_eltype(l, x)
    # scale and tau must be positive
    scale_t = Flux.softplus.(l.scale)
    tau_t = Flux.softplus.(l.tau)
    beta = l.weight .* scale_t .* tau_t
    return l.activation(beta * xT .+ l.bias)
end

function Base.show(io::IO, l::NonCentredTauScaleMixtureDense)
    println(io, "NonCentredScaleMixtureDense(")
    println(io, "\t", size(l.weight, 2), " => ", size(l.weight, 1), ";")
    println(io, "\t", "activation = ", l.activation, ",")
    println(io, "\t", "bias = ", l.bias, ",")
    println(io, "\t", "lambda = ", l.lambda, ",")
    println(io, "\t", "prior_scale = ", l.prior_scale, ",")
    println(io, "\t", "prior_tau = ", l.prior_tau, ",")
    println(io, "\t", "prior_weight = ", l.prior_weight, ",")
    print(io, ")")
end

# Add penalty for NonCentredScaleMixtureDense layer - Horseshoe hierachical structure here
function Penalties.penalty(l::NonCentredTauScaleMixtureDense)
    scale_t = Flux.softplus.(l.scale)
    tau_t = Flux.softplus.(l.tau)

    -l.lambda * (
        sum(Distributions.logpdf.(l.prior_tau, tau_t)) +
        sum(Distributions.logpdf.(l.prior_scale, scale_t)) +
        sum(Distributions.logpdf.(l.prior_weight, l.weight))
    )
end

# Extend weights tracking function
function WeightTracking.weight_container_init(layer::NonCentredTauScaleMixtureDense; n_iter::Int64)
    w_dict = Dict()
    for (pos, param) in enumerate(Flux.params(layer))
        param_size = size(param)
        w_dict[string(pos)] = zeros32(param_size..., n_iter)
    end

    return w_dict
end

function WeightTracking.weight_tracking_push!(epoch::Int64, layer::NonCentredTauScaleMixtureDense, dict_weights_layer::Dict, dict_dims_layer::Dict)
    for (pos, param) in enumerate(Flux.params(layer))
        dict_weights_layer[string(pos)][dict_dims_layer[string(pos)]..., epoch] = param
    end
end

function WeightTracking.container_dim_init(layer::NonCentredTauScaleMixtureDense)
    dim_dict = Dict()
    for (pos, param) in enumerate(Flux.params(layer))
        param_size = size(param)
        dim_dict[string(pos)] = ntuple(_ -> (:), length(param_size))
    end

    return dim_dict
end


"""
    Dense layer with prior distribution on weights.
    The activation function defined in the Dense layer is also used as a bijector transformation on the params in the penalty

    DensePrior(
        dense_layer::Flux.Dense
        prior::Distributions.Distribution
    )

    Standard Dense layer with prior distribution on weights
"""
struct DensePrior{D<:Flux.Dense, P<:Distributions.Distribution}
    dense_layer::D
    prior::P
end

Flux.@functor DensePrior

function (l::DensePrior)(x::AbstractVecOrMat)
    return l.dense_layer(x)
end

function Base.show(io::IO, l::DensePrior)
    println(io, "DensePrior(")
    println(io, "\t", l.dense_layer)
    println(io, "\t", "Prior distribution: ", l.prior)
    print(io, ")")
end

function Penalties.penalty(l::DensePrior)
    dense_t = l.dense_layer.σ.(l.dense_layer.weight)
    -sum(Distributions.logpdf.(l.prior, dense_t))
end

# Extend weight tracking function
function WeightTracking.weight_container_init(layer::DensePrior; n_iter::Int64)
    w_dict = Dict()
    for (pos, param) in enumerate(Flux.params(layer))
        param_size = size(param)
        w_dict[string(pos)] = zeros32(param_size..., n_iter)
    end

    return w_dict
end

function WeightTracking.weight_tracking_push!(epoch::Int64, layer::DensePrior, dict_weights_layer::Dict, dict_dims_layer::Dict)
    for (pos, param) in enumerate(Flux.params(layer))
        dict_weights_layer[string(pos)][dict_dims_layer[string(pos)]..., epoch] = param
    end
end

function WeightTracking.container_dim_init(layer::DensePrior)
    dim_dict = Dict()
    for (pos, param) in enumerate(Flux.params(layer))
        param_size = size(param)
        dim_dict[string(pos)] = ntuple(_ -> (:), length(param_size))
    end

    return dim_dict
end


end
