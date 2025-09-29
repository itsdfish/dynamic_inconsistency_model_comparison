##########################################################################################################
#                                               load dependencies
##########################################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("../")
using Revise
using CSV
using DataFrames
using MCMCChains
using Pigeons
using PriorityHeuristicModels
using QuantumDynamicInconsistencyModels
using ReferencePointModels
using Random
using StatsPlots
using TrueAndErrorDependentModels
using TrueAndErrorModels
using Turing
using TuringUtilities
include("data_parsing_functions.jl")
include("true_error_model_utility_functions.jl")
Random.seed!(655)
##########################################################################################################
#                                               load rpph predictions
##########################################################################################################
df_rpph =
    CSV.read("../reference_point_priority_heuristic_analysis/results_table.csv", DataFrame)
# remove practice trial, as it only has one replicate 
filter!(x -> x.gamble_id ≠ 0, df_rpph)
##########################################################################################################
#                                               parse data
##########################################################################################################
# load the data
df = CSV.read("../Barkan_2003_data/Barkan_2003_data.csv", DataFrame)
df_stacked = parse_barkan_data(df)
# remove practice trial, as it only has one replicate 
filter!(x -> x.gamble_id ≠ 0, df_stacked)
# code response categories as equation (response) categories (1-16)
df_responses = combine(
    groupby(df_stacked, [:subject, :gamble_id, :gain, :loss, :stage1_outcome]),
    (x -> code_response_category(x))
)
rename!(df_responses, :x1 => :response_id)

# create a vector of response counts for each gamble
df_counts = combine(
    groupby(df_responses, [:gamble_id, :gain, :loss, :stage1_outcome]),
    (x -> [count_responses(x)])
)
rename!(df_counts, :x1 => :counts)
sort!(df_counts, :stage1_outcome)
# test correct ordering 
if df_counts.gamble_id ≠ df_rpph.gamble_id
    error("not sorted correctly")
end

BFs = fill(0.0, size(df_counts, 1))
cnt = 1
for df_gamble ∈ eachrow(df_counts)
    println("estimating parameters for gamble $cnt")
    data = df_gamble.counts
    pred_idx = get_rpph_pred_index(df_rpph.prediction[cnt])
    pt_rpph = pigeons(
        target = TuringLogPotential(rpph_dependency_model(data, pred_idx)),
        record = [traces],
        multithreaded = true
    )
    # mcmc chain 
    #chain_rpph = Chains(pt_rpph)
    # marginal log likelihood
    mll_rpph = stepping_stone(pt_rpph)

    pt_tet4 = pigeons(
        target = TuringLogPotential(tet4_depedency_model_uniform(data)),
        record = [traces],
        multithreaded = true
    )
    # mcmc chain 
    #chain_tet4 = Chains(pt_tet4)
    # marginal log likelihood
    mll_tet4 = stepping_stone(pt_tet4)
    # Bayes Factor 
    BFs[cnt] = log10.(exp.(mll_rpph .- mll_tet4))
    cnt += 1
end
df_counts.BF = BFs
##########################################################################################################
#                                 compute decision difficulty
##########################################################################################################
df_counts.signal_to_noise = compute_difficulty(df_counts)
df_bf =
    select(df_counts, [:gamble_id, :gain, :loss, :stage1_outcome, :signal_to_noise, :BF])
# relable ids for table in paper. Skip practice trial 9
df_bf.gamble_id .= [1:8..., 10:17...]
CSV.write("dependent_rpph_uniform_bfs.csv", df_bf)
