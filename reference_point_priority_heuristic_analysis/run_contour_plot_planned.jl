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
using Plots
include("../model_comparison/data_parsing_functions.jl")
##########################################################################################################
#                                               plot configuration
##########################################################################################################
config = (
    fill = true,
    levels = 3,
    xlabel = L"$x_G$",
    xlims = (0, 250),
    ylims = (-250, 0),
    frame_style = :box,
    colorbar_ticks = (1:4, ["safe", "risky", "next feature", "na"]),
    xaxis = font(5),
    yaxis = font(5),
    titlefontsize = 6,
    colorbarfontsize = 6,
    color = [:black, :green, :yellow, :white],
    colorbar = false
)
##########################################################################################################
#                                              load data
##########################################################################################################
df = CSV.read("../Barkan_2003_data/Barkan_2003_data.csv", DataFrame)
df_stacked = parse_barkan_data(df)
gambles = unique(df_stacked, :gamble_id)
gambles_stage1_loss = filter(x -> x.stage1_outcome == "x_L", eachrow(gambles))
# stage 1 wins
gambles_stage1_win = filter(x -> x.stage1_outcome == "x_G", eachrow(gambles))
##########################################################################################################
#                                        generate predictions
##########################################################################################################
pyplot()
model = PriorityHeuristic()
x_Gs = range(0, 250, length = 500)
x_Ls = range(0, -250, length = 500)
p_outcome = 0.5
decisions_min = [decide_min_outcome(model, 0, x_G, x_L) for x_L ∈ x_Ls, x_G ∈ x_Gs]
decision_idxs_min = map(x -> to_decision_index(x), decisions_min)
decision_idxs_min[1, 1] = 4
##########################################################################################################
#                                        generate min outcome plot
##########################################################################################################
min_outcome_plot = contour(
    x_Gs,
    x_Ls,
    decision_idxs_min,
    title = "min outcome";
    config...,
    ylabel = L"$x_L$"
)
##########################################################################################################
#                              generate predictions min outcome probability
##########################################################################################################
decisions_min_prob =
    [decide_min_outcome_prob(model, 0, x_G, x_L, p_outcome) for x_L ∈ x_Ls, x_G ∈ x_Gs]
decision_idxs_min_prob = map(x -> to_decision_index(x), decisions_min_prob)
decision_idxs_min_prob[decision_idxs_min .≠ 3] .= 4
# needed to properly display color bar
decision_idxs_min_prob[1, 1] = 1
decision_idxs_min_prob[1, 2] = 3
##########################################################################################################
#                                 generate plot min outcome probability
##########################################################################################################
min_prob_plot = contour(
    x_Gs,
    x_Ls,
    decision_idxs_min_prob,
    title = "min outcome probability";
    config...,
    yticks = :none
)
##########################################################################################################
#                                    generate predictions maximum outcome
##########################################################################################################
decisions_max = [decide_max_outcome(model, 0, x_G, x_L) for x_L ∈ x_Ls, x_G ∈ x_Gs]
decision_idxs_max = map(x -> to_decision_index(x), decisions_max)
decision_idxs_max[decision_idxs_min_prob .≠ 3] .= 4

# needed to properly display color bar
decision_idxs_max[1, 1] = 1
decision_idxs_max[1, 2] = 3
decision_idxs_max[2, 1] = 2
##########################################################################################################
#                                      generate plot maximum outcome
##########################################################################################################
max_plot =
    contour(x_Gs, x_Ls, decision_idxs_max, title = "max outcome"; config..., yticks = :none)
##########################################################################################################
#                                    generate predictions for decision
##########################################################################################################
decisions = [decide(model, 0, x_G, x_L, p_outcome) for x_L ∈ x_Ls, x_G ∈ x_Gs]
decision_idxs = map(x -> to_decision_index(x), decisions)
# needed to properly display color bar
decision_idxs[1, 1] = 1
decision_idxs[1, 2] = 3
decision_plot = contour(
    x_Gs,
    x_Ls,
    decision_idxs,
    title = "planned decision";
    config...,
    yticks = :none
)
##########################################################################################################
#                                           add outcomes 
##########################################################################################################
scatter!(
    decision_plot,
    gambles.gain,
    gambles.loss,
    leg = false,
    color = :darkorange,
    markersize = 1,
    markerstrokewidth = 0.0
)
##########################################################################################################
#                                           combine plots 
##########################################################################################################
plot(
    min_outcome_plot,
    min_prob_plot,
    max_plot,
    decision_plot,
    layout = (1, 4),
    size = (350, 100),
    xaxis = font(4),
    titlefontsize = 5
)

savefig("predictions_planned.eps")
