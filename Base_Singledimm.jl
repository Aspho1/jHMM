module HiddenMM
    using Distributions
    using Random

    # λ is the HMM with:
    struct HMM
        # * A=transition matrix
        A::Matrix
        # * B=vector of emission distributions
        B::Vector
        # * N=distinct states
        N::Int64
        # * π₀=initital probabilities
        π₀::Vector
    end
    
    function ForwardsAlgorithm(λ, X)
        # X is the series of observations
        # Given λ, what is the probability of emitting X?
        
        T = length(X)
        α = zeros(λ.N, T)

        for i ∈ 1:λ.N
            α[i, 1] = λ.π₀[i] * pdf(λ.B[i], X[1])
        end
        for t ∈ 2:T
            for j ∈ 1:λ.N
                α[j, t] = 0.0
                for i ∈ 1:λ.N
                    α[j, t] += α[i, t-1] * λ.A[i, j]
                end
                α[j, t] *= pdf(λ.B[j], X[t])
            end
        end

        p = sum(α[:, T])
        return log(p)
    end

    
    function BackwardAlgorithm(λ, X)
        T = length(X)
        β = zeros(λ.N, T)
        
        # Initialization (t=T)
        for i in 1:λ.N
            β[i, T] = 1.0
        end
        
        # Induction
        for t ∈ (T-1):-1:1
            for i ∈ 1:λ.N
                β[i, t] = 0.0
                for j ∈ 1:λ.N
                    β[i, t] += λ.A[i, j] * pdf(λ.B[j], X[t+1]) * β[j, t+1]
                end
            end
        end
        

        p = 0.0
        for i ∈ 1:λ.N
            p += λ.π₀[i] * pdf(λ.B[i], X[1]) * β[i, 1]
        end
        
        return log(p)
    end

    function ViterbiAlgorithm(λ, X)
        T = length(X)
        N = λ.N
        
        log_δ = fill(-Inf, N, T)
        ψ = zeros(Int, N, T)
        
        for i ∈ 1:N
            log_δ[i, 1] = log(λ.π₀[i]) + log(pdf(λ.B[i], X[1]))
            ψ[i, 1] = 0 # NOT USED
        end
        
        # For each index, check both states
        for t ∈ 2:T
            for j ∈ 1:N
                # Find the maximum previous state in log space
                vals = log_δ[:, t-1] .+ log.(λ.A[:, j])
                max_val, max_idx = findmax(vals)
                # Update log-delta and psi
                log_δ[j, t] = max_val + log(pdf(λ.B[j], X[t]))
                ψ[j, t] = max_idx
            end
        end
        
        # Termination
        log_P_star = maximum(log_δ[:, T])
        # q_star_T = argmax(log_δ[:, T])
        q_star_T = findmax(log_δ[:, T])[2]
        
        # Path backtracking
        q_star = zeros(Int, T)
        q_star[T] = q_star_T
        
        for t in (T-1):-1:1
            q_star[t] = ψ[q_star[t+1], t+1]
        end
        
        return q_star, log_P_star
    end

    function BaumWelch(λ, X, max_iterations=100, tolerance=1e-6)
        T = length(X)
        N = λ.N
        
        # Current model parameters
        A = copy(λ.A)
        B = deepcopy(λ.B)  # Deep copy for distributions
        π₀ = copy(λ.π₀)
        
        old_log_likelihood = -Inf
        
        for iter in 1:max_iterations
            # E-step: Compute forward and backward variables
            α = zeros(N, T)
            β = zeros(N, T)
            
            # Forward pass (similar to ForwardsAlgorithm but keeping raw α values)
            for i ∈ 1:N
                α[i, 1] = π₀[i] * pdf(B[i], X[1])
            end
            for t ∈ 2:T
                for j ∈ 1:N
                    α[j, t] = 0.0
                    for i ∈ 1:N
                        α[j, t] += α[i, t-1] * A[i, j]
                    end
                    α[j, t] *= pdf(B[j], X[t])
                end
            end
            
            # Backward pass (similar to BackwardAlgorithm but keeping raw β values)
            for i in 1:N
                β[i, T] = 1.0
            end
            for t ∈ (T-1):-1:1
                for i ∈ 1:N
                    β[i, t] = 0.0
                    for j ∈ 1:N
                        β[i, t] += A[i, j] * pdf(B[j], X[t+1]) * β[j, t+1]
                    end
                end
            end
            
            # Calculate log-likelihood
            log_likelihood = log(sum(α[:, T]))
            
            # Check convergence
            if abs(log_likelihood - old_log_likelihood) < tolerance && iter > 1
                break
            end
            old_log_likelihood = log_likelihood
            
            # M-step: Update model parameters
            
            # Compute ξ and γ
            ξ = zeros(N, N, T-1)  # Probability of being in state i at t and state j at t+1
            γ = zeros(N, T)       # Probability of being in state i at time t
            
            # Calculate γ for all time steps
            for t ∈ 1:T
                denominator = sum(α[i, t] * β[i, t] for i ∈ 1:N)
                for i ∈ 1:N
                    γ[i, t] = (α[i, t] * β[i, t]) / denominator
                end
            end
            
            # Calculate ξ for transitions between time steps
            for t ∈ 1:(T-1)
                denominator = 0.0
                for i ∈ 1:N
                    for j ∈ 1:N
                        denominator += α[i, t] * A[i, j] * pdf(B[j], X[t+1]) * β[j, t+1]
                    end
                end
                
                for i ∈ 1:N
                    for j ∈ 1:N
                        ξ[i, j, t] = (α[i, t] * A[i, j] * pdf(B[j], X[t+1]) * β[j, t+1]) / denominator
                    end
                end
            end
            
            # Update initial distribution
            for i ∈ 1:N
                π₀[i] = γ[i, 1]
            end
            
            # Update transition matrix
            for i ∈ 1:N
                denominator = sum(γ[i, t] for t ∈ 1:(T-1))
                for j ∈ 1:N
                    numerator = sum(ξ[i, j, t] for t ∈ 1:(T-1))
                    A[i, j] = denominator > 0 ? numerator / denominator : A[i, j]
                end
            end
            
            # Update emission distributions
            # This part depends on the specific emission distribution type
            # Here, I'll provide a generic update for discrete observations
            # For continuous distributions like Gaussian, you would update mean and variance
            
            # Example assuming B contains discrete distributions:
            for i ∈ 1:N
                # This update depends on the type of distribution
                # For Gaussian distributions:
                if typeof(B[i]) <: Distribution
                    # For normal distributions, update mean and variance
                    if typeof(B[i]) <: Normal
                        # Update mean
                        numerator = sum(γ[i, t] * X[t] for t ∈ 1:T)
                        denominator = sum(γ[i, t] for t ∈ 1:T)
                        μ = numerator / denominator
                        
                        # Update variance
                        numerator = sum(γ[i, t] * (X[t] - μ)^2 for t ∈ 1:T)
                        σ² = numerator / denominator
                        
                        # Create new distribution with updated parameters
                        B[i] = Normal(μ, sqrt(σ²))
                    end
                    # Add other distribution types as needed
                end
            end
        end
        
        # Return updated model
        return HMM(A, B, N, π₀)
    end







    function GetLikelihood(λ, C, X)
        # Given λ, C, what is the probability of emitting X?
        p=nothing

        intitial_state = C[1]
        trans_like = λ.π₀[intitial_state]
        emission_like = pdf(λ.B[intitial_state], X[1])
        
        loglikelihood = log(trans_like * emission_like)

        
        for i in 2:length(X)
            prev_state = C[i-1]
            curr_state = C[i]
            
            trans_like = λ.A[prev_state, C[i]]
            emission_like = pdf(λ.B[curr_state], X[i])
            loglikelihood += log(trans_like * emission_like)

        end
        return loglikelihood
    
    end


    function generate_brute_force_sequences(λ, X)
        # Function to generate all possible state sequences and their likelihoods
        
        N = λ.N  # Number of states
        T = length(X)  # Length of observation sequence
        
        # Initialize array to hold {state_sequence, likelihood} pairs
        brute_force = Vector{Tuple{Vector{Int64}, Float64}}()
        
        # Generate all possible state sequences
        # For N states and T observations, there are N^T possible sequences
        total_sequences = N^T
        
        # Check if this is feasible (might be too large for long sequences)
        if total_sequences > 10^6
            println("Warning: There are $total_sequences possible sequences, which may be too many to enumerate.")
            println("Limiting to first 10^6 sequences for practicality.")
            total_sequences = 10^6
        end
        
        # Generate sequences
        for i in 0:(total_sequences-1)
            # Convert number to base-N representation
            sequence = zeros(Int64, T)
            num = i
            for t in 1:T
                sequence[T-t+1] = (num % N) + 1  # +1 because states are 1-indexed
                num = div(num, N)
            end
            
            # Calculate likelihood for this sequence
            log_likelihood = HiddenMM.GetLikelihood(λ, sequence, X)
            
            # Store sequence and its log-likelihood
            push!(brute_force, (sequence, log_likelihood))
        end
        
        # Sort by likelihood (descending)
        sort!(brute_force, by = x -> x[2], rev = true)
        
        return brute_force
    end
end