
using Distributions
using Random


using .HiddenMM

Random.seed!(0)
A = [
    .7 .3;
    .6 .4
]

# Rows are states, columns are datastreams
B = [Normal(2,1) Exponential(1/2);
     Normal(3,1) Exponential(1/5)
     ]
π₀ = [.5; .5]
X::Vector{Vector{Float64}} = []
C::Vector{Int64} = [2,2,1,1,2,2,1,1]



for c in C
    push!(X, [rand(B[c,1]), rand(B[c,2])])
end

println(X)
λ = HiddenMM.create_hmm(A,B, π₀)
println("Pre EM parameters: ")
HiddenMM.print(λ)
println("loglikelihood of the proposed sequence `", C, "` : ",HiddenMM.ForwardsAlgorithm(λ, X))

λ = HiddenMM.BaumWelch(λ, X, 1000,1e-7)
println("Post EM parameters: ")
HiddenMM.print(λ)

println("loglikelihood of the EM sequence `", C, "` : ",HiddenMM.ForwardsAlgorithm(λ, X))
println(HiddenMM.BackwardAlgorithm(λ, X))

mostlikely, ll = HiddenMM.ViterbiAlgorithm(λ, X)

println("The most likely sequence is : ", mostlikely)
println("With a loglikelihood of: ", ll)

println(repeat("-",80))