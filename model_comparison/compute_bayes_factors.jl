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
using StatsPlots
##########################################################################################################
#                                               load marginal log likelihoods
##########################################################################################################
mll_rpm = CSV.read(
    "reference_point_model_output/marginal_log_likelihood/reference_point_marginal_log_likelihood.csv",
    DataFrame
)
mll_qdim = CSV.read(
    "qdim_output/marginal_log_likelihood/qdim_marginal_log_likelihood.csv",
    DataFrame
)
mll_pm = CSV.read(
    "priority_model_output/marginal_log_likelihood/priority_model_marginal_log_likelihood.csv",
    DataFrame
)

mll_rpm_uniform = CSV.read(
    "reference_point_uniform_model_output/marginal_log_likelihood/reference_point_marginal_log_likelihood.csv",
    DataFrame
)
mll_qdim_uniform = CSV.read(
    "qdim_uniform_output/marginal_log_likelihood/qdim_marginal_log_likelihood.csv",
    DataFrame
)
mll_pm_uniform = CSV.read(
    "priority_uniform_model_output/marginal_log_likelihood/priority_model_marginal_log_likelihood.csv",
    DataFrame
)
##########################################################################################################
#                                   compute log Bayes factors
##########################################################################################################
# BF: exponentiate difference in marginal log likelihood
log_bf_qdim_rpm = log10.(exp.(mll_qdim.m_ll .- mll_rpm.m_ll))
log_bf_qdim_pm = log10.(exp.(mll_qdim.m_ll .- mll_pm.m_ll))
log_bf_rpm_pm = log10.(exp.(mll_rpm.m_ll .- mll_pm.m_ll))

log_bf_qdim_rpm_uniform = log10.(exp.(mll_qdim_uniform.m_ll .- mll_rpm_uniform.m_ll))
log_bf_qdim_pm_uniform = log10.(exp.(mll_qdim_uniform.m_ll .- mll_pm_uniform.m_ll))
log_bf_rpm_pm_uniform = log10.(exp.(mll_rpm_uniform.m_ll .- mll_pm_uniform.m_ll))
##########################################################################################################
#                                   plot log Bayes factors
##########################################################################################################
pyplot()
config = (
    xaxis = font(7),
    yaxis = font(7),
    titlefontsize = 8,
    color = :grey,
    alpha = 0.60,
    leg = false,
    grid = false,
    ylims = (-6.5, 6.5),
    framestyle = :box,
    markersize = 1.6,
    markerstrokewidth = 0.5
)

ylabel_plot = plot(;
    framestyle = :none,
    yaxis = font(8),
    xlabel = "",
    ylabel = L"\log_{10} \mathrm{ \ Bayes \ Factor}",
    title = ""
)

bf_plot = dotplot(fill("", 100), log_bf_qdim_rpm; title = "Informed", config...)
dotplot!(fill(" ", 100), log_bf_qdim_pm; config...)
dotplot!(fill("  ", 100), log_bf_rpm_pm; config...)
hline!([0], linestyle = :dash, color = :black, linewidth = 0.5)

bf_plot_uniform =
    dotplot(fill("QM vs RPPT", 100), log_bf_qdim_rpm_uniform; title = "Uniform", config...)
dotplot!(fill("QM vs SRPPH", 100), log_bf_qdim_pm_uniform; config...)
dotplot!(fill("RPPT vs SRPPH", 100), log_bf_rpm_pm_uniform; config...)
hline!([0], linestyle = :dash, color = :black, linewidth = 0.5)

layout = @layout [a{0.005w} b]
plot(
    ylabel_plot,
    plot(bf_plot, bf_plot_uniform, layout = (2, 1));
    layout,
    size = (340, 220)
)
savefig("model_comparison_bayes_factor.eps")
