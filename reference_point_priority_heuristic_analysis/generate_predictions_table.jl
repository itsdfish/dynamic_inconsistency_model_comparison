##########################################################################################################
#                                               load dependencies
##########################################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("..")
using Revise
using CSV
using DataFrames
using LaTeXStrings
using PriorityHeuristicModels
using PriorityHeuristicModels: to_decision_index
using QuantumDynamicInconsistencyModels
using ReferencePointModels
using Statistics
include("../model_comparison/data_parsing_functions.jl")
##########################################################################################################
#                                              load data
##########################################################################################################
df = CSV.read("../Barkan_2003_data/Barkan_2003_data.csv", DataFrame)
df_stacked = parse_barkan_data(df)
# load Bayes factors for each gamble 
df_bf = CSV.read("../true_error_model_analysis/rpph_bfs.csv", DataFrame)
##########################################################################################################
#                                    compute modal joint choice probabilities
##########################################################################################################
# compute joint choice probability across gambles 
df_joint_probs = combine(
    groupby(df_stacked, [:stage1_outcome, :gain, :loss, :gamble_id]),
    [:RR, :RS, :SR, :SS] .=> mean .=> [:RR, :RS, :SR, :SS]
)

function get_modal_choice(df)
    labels = [:RR, :RS, :SR, :SS]
    choice_probs = [df.RR, df.RS, df.SR, df.SS]
    modal_choice, idx = findmax(choice_probs)
    return "$(labels[idx]) ($(round(modal_choice, digits = 2)))"
    #return modal_choice
end

df_joint_probs.modal_choice .= map(r -> get_modal_choice(r), eachrow(df_joint_probs))
##########################################################################################################
#                                  compute marginal probabilities
##########################################################################################################
# compute marginal choice probabilities from joint probabilties 
df_stacked.plan_take .= df_stacked.RR .+ df_stacked.RS
df_stacked.final_take .= df_stacked.RR .+ df_stacked.SR

# compute mean acceptance across gambles
df_summary = combine(
    groupby(df_stacked, [:stage1_outcome, :gain, :loss, :gamble_id]),
    :plan_take => mean,
    :final_take => mean
)
##########################################################################################################
#                                  compute modal joint probability predictions
##########################################################################################################
df_joint_probs.prediction .= ""

model = PriorityHeuristic()
i = 1
for row ∈ eachrow(df_joint_probs)
    planned_decision = decide(model, 0, row.gain, row.loss, 0.5)
    # map to S for safe and R for risky 
    planned_decision = planned_decision == :retreat ? "S" : "R"
    stage1_outcome = row.stage1_outcome == "x_G" ? row.gain : row.loss
    final_decision = decide(model, stage1_outcome, row.gain, row.loss, 0.5)
    # map to S for safe and R for risky 
    final_decision = final_decision == :retreat ? "S" : "R"
    df_joint_probs.prediction[i] = planned_decision * final_decision
    i += 1
end

df_table = select(
    df_joint_probs,
    [:gamble_id, :gain, :loss, :stage1_outcome, :prediction, :modal_choice]
)

sort!(df_table, :stage1_outcome)
# add Bayes factor to results table 
BF = df_bf.BF
idx = findfirst(x -> x.gamble_id == 0, eachrow(df_table))
insert!(BF, idx, NaN)
df_table.BF .= BF
CSV.write("results_table.csv", df_table)
