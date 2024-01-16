# Custom Flux Layers
module CustomFluxLayers

using Flux
using Distributions

script_path = normpath(joinpath(@__FILE__, "..", ".."))
# include(joinpath(script_path, "train_functions", "losses.jl"))

# Internal
using FluxStats
using FluxStats: Penalties


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
    l.lambda * FluxStats.huber(sum(Flux.sigmoid_fast.(l.scale)) - 1.f0)
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
        prior_scale::P = Distributions.TruncatedNormal(0f0, 10f0, 0f0, Inf32)
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
    prior_scale=Distributions.TruncatedNormal(0f0, 10f0, 0f0, Inf32)
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

    -sum(Distributions.logpdf.(l.prior_scale, scale)) - l.lambda * sum(Distributions.logpdf.(prior_weight, l.weight))
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


end
