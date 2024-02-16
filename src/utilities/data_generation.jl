# Module for data generation

module data_generation

    using Distributions
    using Random
    using LinearAlgebra


    """
    Generate data for a linear regression of n observation and p covariates
    """
    function linear_regression_data(;
        n::Int64,
        p::Int64,
        sigma2::Float64=1.,
        beta_intercept::Float64=0.,
        covariance_matrix::Union{Matrix{Float64}, Nothing}=nothing,
        correlation_coefficients::Vector{Float64}=Float64[],
        block_covariance::Bool=false,
        beta_pool::Vector{Float64}=Float64[],
        beta_signal_strength::Float64=0.,
        prop_zero_coef::Float64=0.
        )
        if prop_zero_coef >= 1.
            throw(error("prop_zero_coef MUST be < 1"))
        end
        if sigma2 <= 0.
            throw(error("The Normal error varaince sigma2 MUST be > 0"))
        end

        if length(beta_pool) > 0
            beta_true = Random.rand(beta_pool, p)
        elseif beta_signal_strength > 0.
            beta_true = Random.rand(Distributions.Normal(0, beta_signal_strength * sqrt(log(p)/n)), p)
        else
            throw(error("One of beta_pool or beta_signal_strength must be specified"))
        end

        # Set coefficients to 0
        if prop_zero_coef > 0.
            n_zero_coef = floor(Int, p * prop_zero_coef)
            which_zero = range(1, n_zero_coef)
            beta_true[which_zero] .= 0.
        end

        if !isnothing(covariance_matrix)
            if dim(covariance_matrix) == p
                covariance_x = covariance_matrix
            else
                throw(error("The provided covariance matrix has wrong dimension"))
            end
        else
            if block_covariance
                covariance_x = create_block_diagonal_toeplitz_matrix(
                    p=p,
                    p_blocks=[n_zero_coef, p-n_zero_coef],
                    correlation_coefficients=correlation_coefficients
                )
            else
                covariance_x = create_toeplitz_covariance_decreasing_corr(
                    p=p,
                    corr_coeff=correlation_coefficients[1]
                )
            end
        end
        # Generate X from a multivariate Normal distribution
        x_distr = Distributions.MultivariateNormal(covariance_x)
        X = transpose(Random.rand(x_distr, n))

        # Get y = X * beta + err ~ N(0, 1)
        y = X * beta_true + Random.rand(Distributions.Normal(0., sqrt(sigma2)), n)
        if beta_intercept > 0
            y .+= beta_intercept
        end

        return (y=y, X=X, beta_true=beta_true, sigma2=sigma2, covariance_matrix=covariance_x)

    end


    """
    Generate data for a linear regression of n observation and p covariates
    """
    function logistic_regression_data(;
        n::Int64,
        p::Int64,
        sigma2::Float64=1.,
        beta_intercept::Float64=0.,
        covariance_matrix::Union{Matrix{Float64}, Nothing}=nothing,
        correlation_coefficients::Vector{Float64}=Float64[],
        block_covariance::Bool=false,
        beta_pool::Vector{Float64}=Float64[],
        beta_signal_strength::Float64=0.,
        prop_zero_coef::Float64=0.
        )
        if prop_zero_coef >= 1.
            throw(error("prop_zero_coef MUST be < 1"))
        end
        if sigma2 <= 0.
            throw(error("The Normal error varaince sigma2 MUST be > 0"))
        end

        if length(beta_pool) > 0
            beta_true = Random.rand(beta_pool, p)
        elseif beta_signal_strength > 0.
            beta_true = Random.rand(Distributions.Normal(0, beta_signal_strength * sqrt(log(p)/n)), p)
        else
            throw(error("One of beta_pool or beta_signal_strength must be specified"))
        end

        # Set coefficients to 0
        if prop_zero_coef > 0.
            n_zero_coef = floor(Int, p * prop_zero_coef)
            which_zero = range(1, n_zero_coef)
            beta_true[which_zero] .= 0.
        end

        if !isnothing(covariance_matrix)
            if dim(covariance_matrix) == p
                covariance_x = covariance_matrix
            else
                throw(error("The provided covariance matrix has wrong dimension"))
            end
        else
            if block_covariance
                covariance_x = create_block_diagonal_toeplitz_matrix(
                    p=p,
                    p_blocks=[n_zero_coef, p-n_zero_coef],
                    correlation_coefficients=correlation_coefficients
                )
            else
                covariance_x = create_toeplitz_covariance_decreasing_corr(
                    p=p,
                    corr_coeff=correlation_coefficients[1]
                )
            end
        end
        # Generate X from a multivariate Normal distribution
        x_distr = Distributions.MultivariateNormal(covariance_x)
        X = transpose(Random.rand(x_distr, n))

        # Get y = X * beta + err ~ N(0, 1)
        y = X * beta_true + Random.rand(Distributions.Normal(0., sqrt(sigma2)), n)
        if beta_intercept > 0
            y .+= beta_intercept
        end

        return (y=y, X=X, beta_true=beta_true, sigma2=sigma2, covariance_matrix=covariance_x)

    end


    """
        create_toeplitz_covariance_matrix(;p, correlation_coefficients::Union{Vector{Float64}, Vector{Any}}=[])

        Generate a covariance matrix with a Toepliz structure, given the provided correlation coefficient for the main off-diagonal entries.
        Default is the (diagonal) Identity matrix.
    """
    function create_toeplitz_covariance_matrix(;p, correlation_coefficients::Vector{Float64}=Float64[])
        covariance_x = diagm(ones(p))
        if length(correlation_coefficients) > 0
            diag_offset = 0
            for cor_coef in correlation_coefficients
                diag_offset += 1
                for kk in range(1, p - diag_offset)
                    covariance_x[kk, kk + diag_offset] = cor_coef
                    covariance_x[kk + diag_offset, kk] = cor_coef
                end
            end
        end

        return covariance_x

    end

    """
        Generate a covariance matrix following the same procedure of the MS paper.
    """
    function create_toeplitz_covariance_decreasing_corr(; p::Int64, corr_coeff::Float64)
        covariance_x = diagm(ones(p))

        diag_offset = 0
        for ll in range(2, p)
            diag_offset += 1
            cor_coef = corr_coeff * (p - ll) / (p - 1)
            for kk in range(1, p - diag_offset)
                covariance_x[kk, kk + diag_offset] = cor_coef
                covariance_x[kk + diag_offset, kk] = cor_coef
            end
        end

        return covariance_x

    end

    """
        create_block_diagonal_toeplitz_matrix(;p, correlation_coefficients::Union{Vector{Float64}, Vector{Any}}=[])

        Create a block diagonal matrix with 2 blocks
    """
    function create_block_diagonal_toeplitz_matrix(;
        p::Int64,
        p_blocks::Vector{Int64},
        correlation_coefficients::Vector{Float64}=Float64[],
        )
        covariance_full = diagm(ones(p))
        # make 2 blocks
        
        start_position = 1
        end_position = 0
        for p_block in p_blocks
            covariance_block = create_toeplitz_covariance_decreasing_corr(p=p_block, corr_coeff=correlation_coefficients[1])
            end_position += p_block
            covariance_full[start_position:end_position, start_position:end_position] = covariance_block
            start_position += p_block
        end
        
        return covariance_full
    end

end
