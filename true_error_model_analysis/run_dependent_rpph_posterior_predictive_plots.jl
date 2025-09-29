##########################################################################################################
#                                               load dependencies
##########################################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("../")
using Revise
using CSV
using DataFrames
using dynamic_inconsistency_model_comparison
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
Random.seed!(4874)
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
##########################################################################################################
#                                 generate posterior predictive distributions 
##########################################################################################################
plots = Plots.Plot[]
gamble_order = 1
for _ ∈ eachrow(df_counts)
    # preference index ordered as pᵣᵣ, pᵣₛ, pₛᵣ, pₛₛ
    pred_idx = get_rpph_pred_index(df_rpph.prediction[gamble_order])
    rpph_est_model = rpph_dependency_model(df_counts.counts[gamble_order], pred_idx)
    # Estimate parameters
    chains = sample(rpph_est_model, NUTS(1000, 0.65), MCMCThreads(), 1000, 4)
    pred_model = predict_distribution(
        TEDM;
        model = rpph_est_model,
        func = x -> x ./ sum(x),
        n_samples = sum(df_counts.counts[gamble_order])
    )

    post_preds = generated_quantities(pred_model, chains)
    post_preds = stack(post_preds, dims = 1)

    gamble_label = gamble_order < 9 ? gamble_order : gamble_order + 1

    ylabel = gamble_label ∈ [1, 5, 10, 14] ? "Response Probability" : ""
    yticks = gamble_label ∈ [1, 5, 10, 14] ? [0.0:0.10:0.50;] : nothing

    labels = get_response_labels()
    xticks = gamble_label ∈ [14:17;] ? (1:length(labels), labels) : nothing

    sub_plot = violin(
        post_preds;
        xticks,
        yticks,
        ylabel,
        xaxis = font(4),
        yaxis = font(5),
        titlefontsize = 6,
        ylims = (-0.01, 0.55),
        leg = false,
        xrotation = 90,
        grid = false,
        title = "trial id $gamble_label"
    )
    scatter!(
        sub_plot,
        1:16,
        df_counts.counts[gamble_order] ./ sum(df_counts.counts[gamble_order]),
        color = :black,
        markersize = 2.5
    )
    push!(plots, sub_plot)
    gamble_order += 1
end

plot(
    plots...,
    layout = (4, 4),
    size = (450, 400),
    left_margin = -1Plots.Measures.mm,
    dpi = 300
)

savefig("dependent_posterior_predictive_grid.png")
