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
include("data_parsing_functions.jl")
include("turing_models.jl")
##########################################################################################################
#                                               generate data
##########################################################################################################
parms =
    (σmax = 0.20, σmin = 0.20, σprob = 0.20, δoutcome = 0.10, δprob = 0.10, p_rep = 0.30)

model = PriorityModel(; parms...)

n_trials = fill(100, 8)

outcome_stage1 = [-5.0, 0, 2, 0, 20, 0, 10, 0]
outcome1_stage2 = [5, 5, 2, 2, 20, 20, 10, 10]
outcome2_stage2 = [-5, 5, -1, -1, -4, -4, -15, 15]
outcome1_prob_stage2 = 0.50

data =
    rand.(
        model,
        outcome_stage1,
        outcome1_stage2,
        outcome2_stage2,
        outcome1_prob_stage2,
        n_trials
    )
all_data = (outcome_stage1, outcome1_stage2, outcome2_stage2, outcome1_prob_stage2, data)
##########################################################################################################
#                                          setup Turing model 
##########################################################################################################
estimator = priority_model(all_data, parms)

pt = pigeons(target = TuringLogPotential(estimator), record = [traces])

chain = sample(estimator, NUTS(), MCMCThreads(), 1000, 4; progress = false)
