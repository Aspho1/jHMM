module HiddenMM
using Distributions
using Random

struct HMM
    A::Matrix
    B::Vector
    N::Int64
    π₀::Vector
end


function Forwards(λ, C_prev, C_current, X_current, p)

end

function GetLikelihood(λ, C, X)
    # Given λ, what is the probability of emitting X?
    p=nothing
    for i ∈ 1:λ.N
        # println(i)
        # println(λ.B[i])
        # println(X[1])
        t = λ.π₀[i]  # Transition matrix start
        e = pdf(λ.B[i], X[1])
        
        
        p = t
        println("Transition probability: ", t)
        println("Emission probability: ", e)

    end

    p
end

end