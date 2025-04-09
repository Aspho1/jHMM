module DistributionFitting

    using Distributions

    function update_distribution(dist::Distribution, state_idx::Int, dim::Int, X, γ)
        if dist isa DiscreteUnivariateDistribution
            return update_discrete_distribution(dist, state_idx, dim, X, γ)
        elseif dist isa ContinuousUnivariateDistribution
            return update_continuous_distribution(dist, state_idx, dim, X, γ)
        elseif dist isa MultivariateDistribution
            return update_multivariate_distribution(dist, state_idx, dim, X, γ)
        else
            # Default case - return original distribution
            @warn "No update method for distribution type $(typeof(dist))"
            return dist
        end
    end

    function update_discrete_distribution(dist::DiscreteUnivariateDistribution, state_idx::Int, dim::Int, X, γ)
        # Extract dimension-specific observations
        Xd = [X[t][dim] for t in 1:length(X)]
        
        if dist isa Bernoulli
            # Update for Bernoulli
            numerator = sum(γ[state_idx, t] * Xd[t] for t ∈ 1:length(X))
            denominator = sum(γ[state_idx, t] for t ∈ 1:length(X))
            
            if denominator > 0
                p = numerator / denominator
                return Bernoulli(p)
            end
        elseif dist isa Categorical
            # Update for Categorical
            n_categories = length(dist.p)
            new_probs = zeros(n_categories)
            
            for k in 1:n_categories
                numerator = sum(γ[state_idx, t] * (Xd[t] == k ? 1 : 0) for t ∈ 1:length(X))
                denominator = sum(γ[state_idx, t] for t ∈ 1:length(X))
                new_probs[k] = denominator > 0 ? numerator / denominator : dist.p[k]
            end
            
            # Normalize probabilities
            new_probs = new_probs ./ sum(new_probs)
            return Categorical(new_probs)
        elseif dist isa Poisson
            # Update for Poisson
            numerator = sum(γ[state_idx, t] * Xd[t] for t ∈ 1:length(X))
            denominator = sum(γ[state_idx, t] for t ∈ 1:length(X))
            
            if denominator > 0
                λ = numerator / denominator
                return Poisson(max(λ, 1e-10))  # Ensure λ is positive
            end
        elseif dist isa Binomial
            # Update for Binomial (assuming n is fixed)
            n = dist.n
            numerator = sum(γ[state_idx, t] * Xd[t] for t ∈ 1:length(X))
            denominator = n * sum(γ[state_idx, t] for t ∈ 1:length(X))
            
            if denominator > 0
                p = numerator / denominator
                return Binomial(n, clamp(p, 0.0, 1.0))  # Ensure p is between 0 and 1
            end
        end
        
        # Default - return original distribution
        return dist
    end

    function update_continuous_distribution(dist::ContinuousUnivariateDistribution, state_idx::Int, dim::Int, X, γ)
        # Extract dimension-specific observations
        Xd = [X[t][dim] for t in 1:length(X)]
        
        if dist isa Normal
            # Update for Normal distribution
            numerator_mean = sum(γ[state_idx, t] * Xd[t] for t ∈ 1:length(X))
            denominator = sum(γ[state_idx, t] for t ∈ 1:length(X))
            
            if denominator > 0
                μ = numerator_mean / denominator
                
                # Update variance
                numerator_var = sum(γ[state_idx, t] * (Xd[t] - μ)^2 for t ∈ 1:length(X))
                σ² = numerator_var / denominator
                
                return Normal(μ, sqrt(max(σ², 1e-10)))
            end
        elseif dist isa Exponential
            # Update for Exponential distribution
            numerator = sum(γ[state_idx, t] * Xd[t] for t ∈ 1:length(X))
            denominator = sum(γ[state_idx, t] for t ∈ 1:length(X))
            
            if denominator > 0 && numerator > 0
                θ = numerator / denominator  # Mean parameter
                return Exponential(θ)
            end
        elseif dist isa Gamma
            # Updating Gamma is more complex as it has shape and scale parameters
            # One approach is to use method of moments
            numerator1 = sum(γ[state_idx, t] * Xd[t] for t ∈ 1:length(X))
            numerator2 = sum(γ[state_idx, t] * (Xd[t]^2) for t ∈ 1:length(X))
            denominator = sum(γ[state_idx, t] for t ∈ 1:length(X))
            
            if denominator > 0 && numerator1 > 0
                mean = numerator1 / denominator
                second_moment = numerator2 / denominator
                variance = second_moment - mean^2
                
                if variance > 0
                    # For Gamma, α = mean²/variance, θ = variance/mean
                    α = mean^2 / variance  # shape parameter
                    θ = variance / mean    # scale parameter
                    return Gamma(α, θ)
                end
            end
        elseif dist isa Weibull
            # Weibull parameter estimation is complex
            # Usually requires numerical optimization
            # Simplified approach using method of moments
            numerator1 = sum(γ[state_idx, t] * log(max(Xd[t], 1e-10)) for t ∈ 1:length(X))
            numerator2 = sum(γ[state_idx, t] * (Xd[t]^2) for t ∈ 1:length(X))
            denominator = sum(γ[state_idx, t] for t ∈ 1:length(X))
            
            if denominator > 0
                mean_log = numerator1 / denominator
                second_moment = numerator2 / denominator
                
                # Simplified parameter estimation
                # For detailed implementation, consider using optimization methods
                k = 1.2  # Shape parameter (fixed or derived via optimization)
                λ = second_moment^(1/2)  # Scale parameter approximation
                return Weibull(k, λ)
            end
        elseif dist isa LogNormal
            # Update for LogNormal distribution
            # Transform to normal space, estimate parameters, transform back
            numerator_mean = sum(γ[state_idx, t] * log(max(Xd[t], 1e-10)) for t ∈ 1:length(X))
            denominator = sum(γ[state_idx, t] for t ∈ 1:length(X))
            
            if denominator > 0
                μ = numerator_mean / denominator
                
                # Update variance in log space
                numerator_var = sum(γ[state_idx, t] * (log(max(Xd[t], 1e-10)) - μ)^2 for t ∈ 1:length(X))
                σ² = numerator_var / denominator
                
                return LogNormal(μ, sqrt(max(σ², 1e-10)))
            end
        elseif dist isa VonMises
            # Update for Von Mises distribution
            # Parameter estimation for circular data
            sin_sum = sum(γ[state_idx, t] * sin(Xd[t]) for t ∈ 1:length(X))
            cos_sum = sum(γ[state_idx, t] * cos(Xd[t]) for t ∈ 1:length(X))
            denominator = sum(γ[state_idx, t] for t ∈ 1:length(X))
            
            if denominator > 0
                # Mean direction
                μ = atan(sin_sum, cos_sum)
                
                # Concentration parameter
                R = sqrt(sin_sum^2 + cos_sum^2) / denominator
                κ = max(R * (2 - R^2) / (1 - R^2), 1e-10)  # Approximation
                
                return VonMises(μ, κ)
            end
        elseif dist isa Uniform
            # Update for Uniform distribution
            # Find min and max values weighted by γ
            min_val = Inf
            max_val = -Inf
            
            for t in 1:length(X)
                if γ[state_idx, t] > 1e-10
                    min_val = min(min_val, Xd[t])
                    max_val = max(max_val, Xd[t])
                end
            end
            
            if min_val < max_val
                return Uniform(min_val, max_val)
            end
        end
        
        # Default - return original distribution
        return dist
    end

    function update_multivariate_distribution(dist::MultivariateDistribution, state_idx::Int, dim::Int, X, γ)
        # For multivariate distributions, we might need to consider all dimensions together
        # This would typically be handled differently from the dimension-by-dimension approach
        
        # Example for multivariate normal:
        if dist isa MvNormal
            # For MvNormal, we need all dimensions, not just one
            # This would be a special case in your implementation
            
            # Extract all observations
            d = length(dist.μ)
            X_full = [X[t] for t in 1:length(X)]
            
            # Weighted mean
            numerator = zeros(d)
            denominator = sum(γ[state_idx, t] for t ∈ 1:length(X))
            
            for t in 1:length(X)
                numerator .+= γ[state_idx, t] .* X_full[t]
            end
            
            if denominator > 0
                μ = numerator ./ denominator
                
                # Weighted covariance
                Σ = zeros(d, d)
                for t in 1:length(X)
                    diff = X_full[t] .- μ
                    Σ .+= γ[state_idx, t] .* (diff * diff')
                end
                Σ ./= denominator
                
                # Ensure covariance matrix is positive definite
                Σ += 1e-10 * I  # Add small regularization
                
                return MvNormal(μ, Σ)
            end
        end
        
        # Default - return original distribution
        return dist
    end

end