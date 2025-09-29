##########################################################################################################
#                                               load dependencies
##########################################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("../")
using Revise
using MCMCChains
using Pigeons
using Turing
using PriorityHeuristicModels
using QuantumDynamicInconsistencyModels
using ReferencePointModels
include("data_parsing_functions.jl")
include("turing_models.jl")
##########################################################################################################
#                                               generate data
##########################################################################################################
parms = (α = 1.2, λ = 2, p_rep = 0.10, w₁ = 0.5, θ = 0.80)

model = ReferencePointModel(; parms...)

n_trials = 100

outcome_stage1 =
    [[5.0, -5], [5, -5], [2, -1], [2, -1], [20, -4], [20, -4], [10, -15], [10, -15]]
outcome_stage2 =
    [[5.0, -5], [5, -5], [2, -1], [2, -1], [20, -4], [20, -4], [10, -15], [10, -15]]
won_first = [true, false, true, false, true, false, true, false]

data =
    rand.(
        model,
        outcome_stage1,
        outcome_stage2,
        won_first,
        n_trials
    )
all_data = (outcome_stage1, outcome_stage2, won_first, data)
##########################################################################################################
#                                          setup Turing model 
##########################################################################################################
estimator = reference_point_model(all_data, parms)

pt = pigeons(target = TuringLogPotential(estimator), record = [traces])

samples = Chains(pt)

chains = sample(estimator, NUTS(), MCMCThreads(), 1000, 4; progress = false)
