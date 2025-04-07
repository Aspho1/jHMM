
using Distributions
using Random


using .HiddenMM


A = [
    .4 .6;
    .6 .4
]

B = [Exponential(5), Exponential(2)]
π₀ = [.1; .9]


λ = HiddenMM.HMM(A,B,2, π₀)

C = [0,1,1,1,1,1,0,0]
X = [5,4,6,6,6]

HiddenMM.GetLikelihood(λ, C, X)