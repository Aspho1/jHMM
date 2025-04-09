module HiddenMM
    using Distributions
    using Random
    include("DistributionFittingFile.jl")
    using .DistributionFitting

# Enhanced HMM structure to handle multi-dimensional observations
    struct HMM
        # * A=transition matrix
        A::Matrix
        # * B=emission distributions (now a matrix: states × dimensions)
        B::Matrix{<:Distribution}
        # * N=distinct states
        N::Int64
        # * D=observation dimensions
        D::Int64
        # * π₀=initial probabilities
        π₀::Vector
    end
    
    # Helper function to calculate emission probability for multi-dimensional observations
    function emission_prob(distributions::Vector{<:Distribution}, observation::Vector)
        # Calculate joint probability as product of individual dimension probabilities
        # (assuming independence between dimensions)
        prob = 1.0
        for d in 1:length(distributions)
            prob *= pdf(distributions[d], observation[d])
        end
        return prob
    end
    
    function ForwardsAlgorithm(λ, X)
        # X is the series of observations where each X[t] is a vector
        # Given λ, what is the probability of emitting X?
        
        T = length(X)
        α = zeros(λ.N, T)

        for i ∈ 1:λ.N
            α[i, 1] = λ.π₀[i] * emission_prob(λ.B[i,:], X[1])
        end
        for t ∈ 2:T
            for j ∈ 1:λ.N
                α[j, t] = 0.0
                for i ∈ 1:λ.N
                    α[j, t] += α[i, t-1] * λ.A[i, j]
                end
                α[j, t] *= emission_prob(λ.B[j,:], X[t])
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
                    β[i, t] += λ.A[i, j] * emission_prob(λ.B[j,:], X[t+1]) * β[j, t+1]
                end
            end
        end
        
        p = 0.0
        for i ∈ 1:λ.N
            p += λ.π₀[i] * emission_prob(λ.B[i,:], X[1]) * β[i, 1]
        end
        
        return log(p)
    end


    ```
    Finds the sequence of hidden states C with the maximal likelihood 
    of X being emitted given λ
    SUPERVISED
    ```
    function ViterbiAlgorithm(λ, X)
        T = length(X)
        N = λ.N
        
        log_δ = fill(-Inf, N, T)
        ψ = zeros(Int, N, T)
        
        for i ∈ 1:N
            log_δ[i, 1] = log(λ.π₀[i]) + log(emission_prob(λ.B[i,:], X[1]))
            ψ[i, 1] = 0 # NOT USED
        end
        
        # For each index, check both states
        for t ∈ 2:T
            for j ∈ 1:N
                # Find the maximum previous state in log space
                vals = log_δ[:, t-1] .+ log.(λ.A[:, j])
                max_val, max_idx = findmax(vals)
                # Update log-delta and psi
                log_δ[j, t] = max_val + log(emission_prob(λ.B[j,:], X[t]))
                ψ[j, t] = max_idx
            end
        end
        
        # Termination
        log_P_star = maximum(log_δ[:, T])
        q_star_T = findmax(log_δ[:, T])[2]
        
        # Path backtracking
        q_star = zeros(Int, T)
        q_star[T] = q_star_T
        
        for t in (T-1):-1:1
            q_star[t] = ψ[q_star[t+1], t+1]
        end
        
        return q_star, log_P_star
    end


    ```
    Finds the parameters of λ which maximize the likelihood of X
    UNSUPERVISED
    ```
    function BaumWelch(λ, X, max_iterations=100, tolerance=1e-6)
        T = length(X)
        N = λ.N
        D = λ.D  # Observation dimensions
        
        # Current model parameters
        A = copy(λ.A)
        B = deepcopy(λ.B)  # Deep copy for distributions matrix
        π₀ = copy(λ.π₀)
        
        old_log_likelihood = -Inf
        
        for iter in 1:max_iterations
            # E-step: Compute forward and backward variables
            α = zeros(N, T)
            β = zeros(N, T)
            
            # Forward pass
            for i ∈ 1:N
                # println(X[1])
                α[i, 1] = π₀[i] * emission_prob(B[i,:], X[1])
            end
            for t ∈ 2:T
                for j ∈ 1:N
                    α[j, t] = 0.0
                    for i ∈ 1:N
                        α[j, t] += α[i, t-1] * A[i, j]
                    end
                    α[j, t] *= emission_prob(B[j,:], X[t])
                end
            end
            
            # Backward pass
            for i in 1:N
                β[i, T] = 1.0
            end
            for t ∈ (T-1):-1:1
                for i ∈ 1:N
                    β[i, t] = 0.0
                    for j ∈ 1:N
                        β[i, t] += A[i, j] * emission_prob(B[j,:], X[t+1]) * β[j, t+1]
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
                        denominator += α[i, t] * A[i, j] * emission_prob(B[j,:], X[t+1]) * β[j, t+1]
                    end
                end
                
                for i ∈ 1:N
                    for j ∈ 1:N
                        ξ[i, j, t] = (α[i, t] * A[i, j] * emission_prob(B[j,:], X[t+1]) * β[j, t+1]) / denominator
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
            for i ∈ 1:N
                for d ∈ 1:D
                    # Update distribution based on its type
                    B[i,d] = DistributionFitting.update_distribution(B[i,d], i, d, X, γ)
                end
            end
        end
        
        # Return updated model
        return HMM(A, B, N, D, π₀)
    end

    function SupervisedLearning(λ, X, C)
        


    end



    # Constructor for creating an HMM with multi-dimensional observations
    function create_hmm(A::Matrix, B::Matrix{<:Distribution}, π₀::Vector)
        N = size(A, 1)
        D = size(B, 2)
        return HMM(A, B, N, D, π₀)
    end

    function print(λ::HMM)
        println("+",repeat("-",33), " HMM Summary ",repeat("-",33),"+")
        println("Initital Probabilities: ")
        println("π₀ = ", λ.π₀)
        println("Transition Matrix: ")
        for i in 1:λ.N
            println(join(λ.A[i,:], ", "))
        end
        println("Emission Distributions")
        for i in 1:λ.N
            println(join(λ.B[i,:], ", "))
        end
        println("+",repeat("-",78),"+")
        
    end 

end