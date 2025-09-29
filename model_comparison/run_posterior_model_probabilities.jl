##########################################################################################################
#                                               load dependencies
##########################################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("../")
using Revise
using CSV
using DataFrames
using LaTeXStrings
using Statistics
using StatsPlots
##########################################################################################################
#                                               load marginal log likelihoods
##########################################################################################################
mll_rpm = CSV.read(
    "reference_point_model_output/marginal_log_likelihood/reference_point_marginal_log_likelihood.csv",
    DataFrame
)
sort!(mll_rpm, :subject)

mll_qdim = CSV.read(
    "qdim_output/marginal_log_likelihood/qdim_marginal_log_likelihood.csv",
    DataFrame
)
sort!(mll_qdim, :subject)

mll_pm = CSV.read(
    "priority_model_output/marginal_log_likelihood/priority_model_marginal_log_likelihood.csv",
    DataFrame
)
sort!(mll_pm, :subject)

mll_rpm_uniform = CSV.read(
    "reference_point_uniform_model_output/marginal_log_likelihood/reference_point_marginal_log_likelihood.csv",
    DataFrame
)
sort!(mll_rpm_uniform, :subject)

mll_qdim_uniform = CSV.read(
    "qdim_uniform_output/marginal_log_likelihood/qdim_marginal_log_likelihood.csv",
    DataFrame
)
sort!(mll_qdim_uniform, :subject)

mll_pm_uniform = CSV.read(
    "priority_uniform_model_output/marginal_log_likelihood/priority_model_marginal_log_likelihood.csv",
    DataFrame
)
sort!(mll_pm_uniform, :subject)
##########################################################################################################
#                                   organize log marginal likelihoods
##########################################################################################################
informed_mlls = [mll_qdim.m_ll mll_rpm.m_ll mll_pm.m_ll]

uniform_mlls = [mll_qdim_uniform.m_ll mll_rpm_uniform.m_ll mll_pm_uniform.m_ll]
function compute_posterior_probs(x)
    v = exp.(x)
    return v ./ sum(v)
end
##########################################################################################################
#                                   plot posterior model probabilities
##########################################################################################################
pyplot()
config = (
    xaxis = font(7),
    yaxis = font(7),
    titlefontsize = 8,
    legendfontsize = 6,
    xlims = (0.75, 100.25),
    xlabel = "Subjects",
    xticks = nothing,
    leg = false,
    grid = false,
    linewidth = 0.5,
    framestyle = :box
)

# model labels 
model_labels = ["QCM" "RPPT" "SRPPH"]
# model colors 
model_colors = [RGB(79 / 255, 110 / 255, 80 / 255) RGB(83 / 255, 35 / 255, 92 / 255) RGB(
    82 / 255,
    84 / 255,
    168 / 255
)]

# compute posterior model probabilities assuming equal prior probabilities
informed_posterior_probs =
    stack(map(x -> compute_posterior_probs(x), eachrow(informed_mlls)), dims = 1)

# sort the models such that highest posterior probability is on the bottom of the bar
model_idx = sortperm(mean(informed_posterior_probs, dims = 1)[:])
informed_posterior_probs = informed_posterior_probs[:, model_idx]

# sort participants with from highest to lowest posterior probability
subj_idx = sortperm(informed_posterior_probs[:, end], rev = true)
informed_posterior_probs = informed_posterior_probs[subj_idx, :]

informed_plot = groupedbar(
    informed_posterior_probs,
    bar_position = :stack;
    color = model_colors[:, model_idx],
    label = model_labels[:, model_idx],
    title = "Informed Priors",
    config...,
    leg = :outerright,
    legendfontsize = 7)

# compute posterior model probabilities assuming equal prior probabilities
uninformed_posterior_probs =
    stack(map(x -> compute_posterior_probs(x), eachrow(uniform_mlls)), dims = 1)

# # sort the models such that highest posterior probability is on the bottom of the bar
# model_idx = sortperm(mean(uninformed_posterior_probs, dims = 1)[:])
uninformed_posterior_probs = uninformed_posterior_probs[:, model_idx]

# sort participants with from highest to lowest posterior probability
subj_idx = sortperm(uninformed_posterior_probs[:, end], rev = true)
uninformed_posterior_probs = uninformed_posterior_probs[subj_idx, :]

uninformed_plot = groupedbar(
    uninformed_posterior_probs,
    bar_position = :stack;
    color = model_colors[:, model_idx],
    label = model_labels[:, model_idx],
    title = "Uninformed Priors",
    config...
)

ylabel_plot = plot(;
    framestyle = :none,
    yaxis = font(8),
    ylabel = "Posterior Model Probability",
    title = ""
)

layout = @layout [a{0.005w} b]
plot(
    ylabel_plot,
    plot(informed_plot, uninformed_plot, layout = (2, 1));
    layout,
    dpi = 300,
    size = (450, 220)
)
savefig("model_posterior_probabilities.eps")
##########################################################################################################
#                                   summarize winning models
##########################################################################################################
_, best_informed_idx = findmax(informed_posterior_probs, dims = 2)
best_informed_idx = map(x -> x[2], best_informed_idx)
informed_win_proportions = map(x -> mean(x .== best_informed_idx), 1:3)
df_informed = DataFrame(
    prior = fill("Informed", 3),
    model = model_labels[:, model_idx][:],
    winner_prop = informed_win_proportions
)

_, best_uninformed_idx = findmax(uninformed_posterior_probs, dims = 2)
best_uninformed_idx = map(x -> x[2], best_uninformed_idx)
uninformed_win_proportions = map(x -> mean(x .== best_uninformed_idx), 1:3)
df_uninformed = DataFrame(
    prior = fill("Uninformed", 3),
    model = model_labels[:, model_idx][:],
    winner_prop = uninformed_win_proportions
)
df_winners = vcat(df_informed, df_uninformed)
