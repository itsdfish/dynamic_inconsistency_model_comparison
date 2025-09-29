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
#                                      generate predictions min outcome
##########################################################################################################
pyplot()
model = PriorityHeuristic()
# probability of first outcome 
p_outcome = 0.10
x_Gs = range(0, 250, length = 500)
x_Ls = range(0, -250, length = 500)
decisions_min_loss = [decide_min_outcome(model, x_L, x_G, x_L) for x_L ∈ x_Ls, x_G ∈ x_Gs]
decision_idxs_min_loss = map(x -> to_decision_index(x), decisions_min_loss)
decision_idxs_min_loss[1, 1] = 4
decisions_min_gain = [decide_min_outcome(model, x_G, x_G, x_L) for x_L ∈ x_Ls, x_G ∈ x_Gs]
decision_idxs_min_gain = map(x -> to_decision_index(x), decisions_min_gain)
decision_idxs_min_gain[1, 1] = 4
##########################################################################################################
#                                        generate min outcome plot
##########################################################################################################
min_outcome_loss_plot = contour(
    x_Gs,
    x_Ls,
    decision_idxs_min_loss,
    title = "min outcome";
    config...,
    xlabel = "",
    ylabel = L"$x_L$"
    #xticks = :none,
)

min_outcome_gain_plot =
    contour(x_Gs, x_Ls, decision_idxs_min_gain, title = ""; config..., ylabel = L"$x_L$")
##########################################################################################################
#                            generate predictions min outcome probability
##########################################################################################################
decisions_min_prob_loss =
    [decide_min_outcome_prob(model, x_L, x_G, x_L, p_outcome) for x_L ∈ x_Ls, x_G ∈ x_Gs]
decision_idxs_min_prob_loss = map(x -> to_decision_index(x), decisions_min_prob_loss)
# 0 indicates parts of the space in which a decision was made in the previous step
decision_idxs_min_prob_loss[decision_idxs_min_loss .≠ 3] .= 4
decision_idxs_min_prob_loss[1, 1] = 1
decision_idxs_min_prob_loss[1, 2] = 3

decisions_min_prob_gain =
    [decide_min_outcome_prob(model, x_G, x_G, x_L, p_outcome) for x_L ∈ x_Ls, x_G ∈ x_Gs]
decision_idxs_min_prob_gain = map(x -> to_decision_index(x), decisions_min_prob_gain)
# 0 indicates parts of the space in which a decision was made in the previous step
decision_idxs_min_prob_gain[decision_idxs_min_gain .≠ 3] .= 4
decision_idxs_min_prob_gain[1, 1] = 1
decision_idxs_min_prob_gain[1, 2] = 3
##########################################################################################################
#                            plot predictions min outcome probability
##########################################################################################################
min_outcome_prob_loss_plot = contour(
    x_Gs,
    x_Ls,
    decision_idxs_min_prob_loss,
    title = "min outcome prob";
    config...,
    xlabel = "",
    ylabel = "",
    #xticks = :none,
    yticks = :none
)

min_outcome_prob_gain_plot = contour(
    x_Gs,
    x_Ls,
    decision_idxs_min_prob_gain,
    title = "";
    config...,
    ylabel = "",
    yticks = :none
)
##########################################################################################################
#                            generate predictions max outcome
##########################################################################################################
decisions_max_loss = [decide_max_outcome(model, x_L, x_G, x_L) for x_L ∈ x_Ls, x_G ∈ x_Gs]
decision_idxs_max_loss = map(x -> to_decision_index(x), decisions_max_loss)
decision_idxs_max_loss[decision_idxs_min_prob_loss .≠ 3] .= 4
decision_idxs_max_loss[1, 1] = 3
decision_idxs_max_loss[1, 2] = 1

decisions_max_gain = [decide_max_outcome(model, x_G, x_G, x_L) for x_L ∈ x_Ls, x_G ∈ x_Gs]
decision_idxs_max_gain = map(x -> to_decision_index(x), decisions_max_gain)
decision_idxs_max_gain[decision_idxs_min_prob_gain .≠ 3] .= 4
decision_idxs_max_gain[1, 1] = 3
decision_idxs_max_gain[1, 2] = 1
##########################################################################################################
#                            plot predictions max outcome
##########################################################################################################
max_outcome_loss_plot = contour(
    x_Gs,
    x_Ls,
    decision_idxs_max_loss,
    title = "max outcome";
    config...,
    xlabel = "",
    ylabel = "",
    #xticks = :none,
    yticks = :none
)

max_outcome_gain_plot = contour(
    x_Gs,
    x_Ls,
    decision_idxs_max_gain,
    title = "";
    config...,
    ylabel = "",
    yticks = :none
)
##########################################################################################################
#                            generate predictions final decision
##########################################################################################################
decisions_loss = [decide(model, x_L, x_G, x_L, p_outcome) for x_L ∈ x_Ls, x_G ∈ x_Gs]
decision_idxs_loss = map(x -> to_decision_index(x), decisions_loss)
decision_idxs_loss[1, 2] = 3
decision_idxs_loss[1, 1] = 4
decisions_gain = [decide(model, x_G, x_G, x_L, p_outcome) for x_L ∈ x_Ls, x_G ∈ x_Gs]
decision_idxs_gain = map(x -> to_decision_index(x), decisions_gain)
decision_idxs_gain[1, 2] = 3
decision_idxs_gain[1, 1] = 4
##########################################################################################################
#                            generate plots final decision
##########################################################################################################
decision_loss_plot = contour(
    x_Gs,
    x_Ls,
    decision_idxs_loss,
    title = "final decision";
    config...,
    xlabel = "",
    ylabel = "",
    #xticks = :none,
    yticks = :none
)

decision_gain_plot = contour(
    x_Gs,
    x_Ls,
    decision_idxs_gain,
    title = "";
    config...,
    ylabel = "",
    yticks = :none
)
##########################################################################################################
#                                           add outcomes 
##########################################################################################################
scatter!(
    decision_gain_plot,
    gambles_stage1_win.gain,
    gambles_stage1_win.loss,
    leg = false,
    color = :darkorange,
    markersize = 1,
    markerstrokewidth = 0.0
)
scatter!(
    decision_loss_plot,
    gambles_stage1_loss.gain,
    gambles_stage1_loss.loss,
    leg = false,
    color = :darkorange,
    markersize = 1,
    markerstrokewidth = 0.0
)
##########################################################################################################
#                                           combine plots 
##########################################################################################################
loss_plots = plot(
    min_outcome_loss_plot,
    min_outcome_prob_loss_plot,
    max_outcome_loss_plot,
    decision_loss_plot,
    layout = (1, 4)
    #left_margin = [5px  5px  5px]

)
gain_plots = plot(
    min_outcome_gain_plot,
    min_outcome_prob_gain_plot,
    max_outcome_gain_plot,
    decision_gain_plot,
    layout = (1, 4)
)
plot(loss_plots, gain_plots, layout = (2, 1), size = (350, 160), xaxis = font(4))
savefig("predictions_final.eps")
