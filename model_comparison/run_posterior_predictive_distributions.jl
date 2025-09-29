##########################################################################################################
#                                               load dependencies
##########################################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("../")
using Revise
using CSV
using DataFrames
using JLD2
using LaTeXStrings
using MCMCChains
using Pigeons
using Plots.Measures
using PriorityHeuristicModels
using QuantumDynamicInconsistencyModels
using ReferencePointModels
using Random
using Statistics
using StatsPlots
include("data_parsing_functions.jl")
include("posterior_predictive_functions.jl")
##########################################################################################################
#                                               load data
##########################################################################################################
Random.seed!(8504)
# load the data
df = CSV.read("../Barkan_2003_data/Barkan_2003_data.csv", DataFrame)
df_stacked = parse_barkan_data(df)

# compute joint probabilities across subjects for each gamble
df_probs = combine(
    groupby(df_stacked, [:stage1_outcome, :gain, :loss]),
    :RR => mean => :RR,
    :RS => mean => :RS,
    :SR => mean => :SR,
    :SS => mean => :SS,
    # number of trials per subject
    :RR => (x -> Int(length(x) / 100)) => :n_trials
)
##########################################################################################################
#                                            generate QM predictions 
##########################################################################################################
n_samples = 1000
# default quantum model parms
default_parms = (α = 0.9, λ = 2, w₁ = 0.5, p_rep = 0.30, γ = 2.5)
# load chain for each subject
chains =
    map(id -> load("qdim_output/pigeons_output/subject_$(id)_output.jld2", "chain"), 1:100)
# remove LL from parameters
map(c -> filter!(p -> p ≠ :LL, c.name_map.parameters), chains)
samples = to_named_tuples.(chains)

qm_preds = map(
    df_row -> generate_post_pred_gamble(
        QDIM,
        samples,
        df_row,
        n_samples;
        sim_func = simulate,
        default_parms
    ),
    eachrow(df_probs)
)

qm_probs = stack(map(p -> p.means, qm_preds), dims = 1)
##########################################################################################################
#                                            generate QM plots 
##########################################################################################################
config = (
    grid = false,
    leg = false,
    color = RGB(112 / 256, 179 / 256, 127 / 256),
    markersize = 2,
    markerstrokewidth = 0.5,
    xaxis = font(6),
    yaxis = font(6),
    alpha = 0.70
)
pyplot()

p_idx = 1
qm_rr_plot = scatter(
    df_probs.RR,
    qm_probs[:, p_idx],
    xlabel = L"\mathrm{data} \ p(R_{p},R_{f})",
    ylabel = L"\mathrm{QCM} \ p(R_{p},R_{f})";
    config...
)
plot!(0.10:0.01:0.75, 0.10:0.01:0.75, color = :black, linewidth = 0.70)
annotate!(
    0.60,
    0.20,
    text("r = $(round(cor(qm_probs[:, p_idx], df_probs.RR), digits = 2))", 5)
)
annotate!(
    0.60,
    0.15,
    text("rmse = $(round(rmse(qm_probs[:, p_idx], df_probs.RR), digits = 2))", 5)
)

p_idx = 2
qm_rs_plot = scatter(
    df_probs.RS,
    qm_probs[:, p_idx],
    xlabel = L"\mathrm{data} \ p(R_{p},S_{f})",
    ylabel = L"\mathrm{QCM} \ p(R_{p},S_{f})";
    config...
)
plot!(0.0:0.01:0.20, 0.0:0.01:0.20, color = :black, linewidth = 0.70)
annotate!(
    0.15,
    0.05,
    text("r = $(round(cor(qm_probs[:, p_idx], df_probs.RS), digits = 2))", 5)
)
annotate!(
    0.15,
    0.03,
    text("rmse = $(round(rmse(qm_probs[:, p_idx], df_probs.RS), digits = 2))", 5)
)

p_idx = 3
qm_sr_plot = scatter(
    df_probs.SR,
    qm_probs[:, p_idx],
    xlabel = L"\mathrm{data} \ p(S_{p},R_{f})",
    ylabel = L"\mathrm{QCM} \ p(S_{p},R_{f})";
    config...
)
plot!(0.0:0.01:0.20, 0.0:0.01:0.20, color = :black, linewidth = 0.70)
annotate!(
    0.15,
    0.05,
    text("r = $(round(cor(qm_probs[:, p_idx], df_probs.SR), digits = 2))", 5)
)
annotate!(
    0.15,
    0.03,
    text("rmse = $(round(rmse(qm_probs[:, p_idx], df_probs.SR), digits = 2))", 5)
)

p_idx = 4
qm_ss_plot = scatter(
    df_probs.SS,
    qm_probs[:, p_idx];
    xlabel = L"\mathrm{data} \ p(S_{p},S_{f})",
    ylabel = L"\mathrm{QCM} \ p(S_{p},S_{f})",
    config...
)
plot!(0.10:0.01:0.75, 0.10:0.01:0.75, color = :black, linewidth = 0.70)
annotate!(
    0.60,
    0.20,
    text("r = $(round(cor(qm_probs[:, p_idx], df_probs.SS), digits = 2))", 5)
)
annotate!(
    0.60,
    0.15,
    text("rmse = $(round(rmse(qm_probs[:, p_idx], df_probs.SS), digits = 2))", 5)
)

qm_plots = plot(qm_rr_plot, qm_rs_plot, qm_sr_plot, qm_ss_plot, layout = (4, 1))
##########################################################################################################
#                                            generate RPPT predictions 
##########################################################################################################
n_samples = 1000
# default reference point priority theory model parms
default_parms = (α = 1.2, λ = 2, p_rep = 0.10, w₁ = 0.5, θ = 0.80)

# load chain for each subject
chains =
    map(
        id -> load(
            "reference_point_model_output/pigeons_output/subject_$(id)_output.jld2",
            "chain"
        ),
        1:100
    )
# remove LL from parameters
map(c -> filter!(p -> p ≠ :LL, c.name_map.parameters), chains)
samples = to_named_tuples.(chains)

rppt_preds = map(
    df_row -> generate_post_pred_gamble(
        ReferencePointModel,
        samples,
        df_row,
        n_samples;
        sim_func = simulate,
        default_parms
    ),
    eachrow(df_probs)
)

rppt_probs = stack(map(p -> p.means, rppt_preds), dims = 1)
##########################################################################################################
#                                            generate RPPT plots 
##########################################################################################################
p_idx = 1
rppt_rr_plot = scatter(
    df_probs.RR,
    rppt_probs[:, p_idx];
    xlabel = L"\mathrm{data} \ p(R_{p},R_{f})",
    ylabel = L"\mathrm{RPPT} \ p(R_{p},R_{f})",
    config...,
    yticks = nothing
)
plot!(0.10:0.01:0.75, 0.10:0.01:0.75, color = :black, linewidth = 0.70)
annotate!(
    0.60,
    0.20,
    text("r = $(round(cor(rppt_probs[:, p_idx], df_probs.RR), digits = 2))", 5)
)
annotate!(
    0.60,
    0.15,
    text("rmse = $(round(rmse(rppt_probs[:, p_idx], df_probs.RR), digits = 2))", 5)
)

p_idx = 2
rppt_rs_plot = scatter(
    df_probs.RS,
    rppt_probs[:, p_idx],
    xlabel = L"\mathrm{data} \ p(R_{p},S_{f})",
    ylabel = L"\mathrm{RPPT} \ p(R_{p},S_{f})",
    yticks = nothing;
    config...
)
plot!(0.0:0.01:0.20, 0.0:0.01:0.20, color = :black, linewidth = 0.70)
annotate!(
    0.15,
    0.05,
    text("r = $(round(cor(rppt_probs[:, p_idx], df_probs.RS), digits = 2))", 5)
)
annotate!(
    0.15,
    0.03,
    text("rmse = $(round(rmse(rppt_probs[:, p_idx], df_probs.RS), digits = 2))", 5)
)

p_idx = 3
rppt_sr_plot = scatter(
    df_probs.SR,
    rppt_probs[:, p_idx],
    xlabel = L"\mathrm{data} \ p(S_{p},R_{f})",
    ylabel = L"\mathrm{RPPT} \ p(S_{p},R_{f})",
    yticks = nothing;
    config...
)
plot!(0.0:0.01:0.20, 0.0:0.01:0.20, color = :black, linewidth = 0.70)
annotate!(
    0.15,
    0.05,
    text("r = $(round(cor(rppt_probs[:, p_idx], df_probs.SR), digits = 2))", 5)
)
annotate!(
    0.15,
    0.03,
    text("rmse = $(round(rmse(rppt_probs[:, p_idx], df_probs.SR), digits = 2))", 5)
)

p_idx = 4
rppt_ss_plot = scatter(
    df_probs.SS,
    rppt_probs[:, p_idx],
    xlabel = L"\mathrm{data} \ p(S_{p},S_{f})",
    ylabel = L"\mathrm{RPPT} \ p(S_{p},S_{f})",
    yticks = nothing;
    config...
)
plot!(0.10:0.01:0.75, 0.10:0.01:0.75, color = :black, linewidth = 0.70)
annotate!(
    0.60,
    0.20,
    text("r = $(round(cor(rppt_probs[:, p_idx], df_probs.SS), digits = 2))", 5)
)
annotate!(
    0.60,
    0.15,
    text("rmse = $(round(rmse(rppt_probs[:, p_idx], df_probs.SS), digits = 2))", 5)
)

rppt_plots = plot(rppt_rr_plot, rppt_rs_plot, rppt_sr_plot, rppt_ss_plot, layout = (4, 1))
##########################################################################################################
n_samples = 1000
# default parameters for srpph intentionally blank
default_parms = ()

# load chain for each subject
chains =
    map(
        id -> load(
            "priority_model_output/pigeons_output/subject_$(id)_output.jld2",
            "chain"
        ),
        1:100
    )
# remove LL from parameters
map(c -> filter!(p -> p ≠ :LL, c.name_map.parameters), chains)
samples = to_named_tuples.(chains)

srpph_preds = map(
    df_row -> generate_post_pred_gamble(
        PriorityModel,
        samples,
        df_row,
        n_samples;
        sim_func = simulate,
        default_parms
    ),
    eachrow(df_probs)
)

srpph_probs = stack(map(p -> p.means, srpph_preds), dims = 1)
##########################################################################################################
#                                            generate SRPPH plots 
##########################################################################################################
p_idx = 1
srpph_rr_plot = scatter(
    df_probs.RR,
    srpph_probs[:, p_idx],
    xlabel = L"\mathrm{data} \ p(R_{p},R_{f})",
    ylabel = L"\mathrm{SRPPH} \ p(R_{p},R_{f})",
    yticks = nothing;
    config...
)
plot!(0.10:0.01:0.75, 0.10:0.01:0.75, color = :black, linewidth = 0.70)
annotate!(
    0.60,
    0.20,
    text("r = $(round(cor(srpph_probs[:, p_idx], df_probs.RR), digits = 2))", 5)
)
annotate!(
    0.60,
    0.15,
    text("rmse = $(round(rmse(srpph_probs[:, p_idx], df_probs.RR), digits = 2))", 5)
)

p_idx = 2
srpph_rs_plot = scatter(
    df_probs.RS,
    srpph_probs[:, p_idx],
    xlabel = L"\mathrm{data} \ p(R_{p},S_{f})",
    ylabel = L"\mathrm{SRPPH} \ p(R_{p},S_{f})",
    yticks = nothing;
    config...
)
plot!(0.0:0.01:0.20, 0.0:0.01:0.20, color = :black, linewidth = 0.70)
annotate!(
    0.15,
    0.05,
    text("r = $(round(cor(srpph_probs[:, p_idx], df_probs.RS), digits = 2))", 5)
)
annotate!(
    0.15,
    0.03,
    text("rmse = $(round(rmse(srpph_probs[:, p_idx], df_probs.RS), digits = 2))", 5)
)

p_idx = 3
srpph_sr_plot = scatter(
    df_probs.SR,
    srpph_probs[:, p_idx],
    xlabel = L"\mathrm{data} \ p(S_{p},R_{f})",
    ylabel = L"\mathrm{SRPPH} \ p(S_{p},R_{f})",
    yticks = nothing;
    config...
)
plot!(0.0:0.01:0.20, 0.0:0.01:0.20, color = :black, linewidth = 0.70)
annotate!(
    0.15,
    0.05,
    text("r = $(round(cor(srpph_probs[:, p_idx], df_probs.SR), digits = 2))", 5)
)
annotate!(
    0.15,
    0.03,
    text("rmse = $(round(rmse(srpph_probs[:, p_idx], df_probs.SR), digits = 2))", 5)
)

p_idx = 4
srpph_ss_plot = scatter(
    df_probs.SS,
    srpph_probs[:, p_idx],
    xlabel = L"\mathrm{data} \ p(S_{p},S_{f})",
    ylabel = L"\mathrm{SRPPH} \ p(S_{p},S_{f})",
    yticks = nothing;
    config...
)
plot!(0.10:0.01:0.75, 0.10:0.01:0.75, color = :black, linewidth = 0.70)
annotate!(
    0.60,
    0.20,
    text("r = $(round(cor(srpph_probs[:, p_idx], df_probs.SS), digits = 2))", 5)
)
annotate!(
    0.60,
    0.15,
    text("rmse = $(round(rmse(srpph_probs[:, p_idx], df_probs.SS), digits = 2))", 5)
)
srpph_plots =
    plot(srpph_rr_plot, srpph_rs_plot, srpph_sr_plot, srpph_ss_plot, layout = (4, 1))
plot(
    qm_plots,
    rppt_plots,
    srpph_plots,
    left_margin = [8mm 0mm],
    layout = (1, 3),
    size = (400, 400),
    dpi = 300
)

savefig("post_pred.png")
##########################################################################################################
#                                     compare standard deviations across gambles
##########################################################################################################
df_stds = combine(df_probs, [:RR, :RS, :SR, :SS] .=> std)
df_stds.source .= "data"
push!(df_stds, [std(qm_probs, dims = 1) "QCM"])
push!(df_stds, [std(rppt_probs, dims = 1) "RPPT"])
push!(df_stds, [std(srpph_probs, dims = 1) "SRPPH"])
