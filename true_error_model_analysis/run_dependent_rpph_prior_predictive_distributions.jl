##########################################################################################################
#                                               load dependencies
##########################################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("../")
using Revise
using Distributions
using LaTeXStrings
using Random
using StatsPlots
using TrueAndErrorDependentModels
using TrueAndErrorModels
##########################################################################################################
#                                 setup posterior predictive distributions 
##########################################################################################################
Random.seed!(50)
labels = get_response_labels()
config = (
    xticks = (1:16, labels),
    ylabel = "Response Probability",
    ylims = (-0.01, 1),
    xrotation = 90,
    xaxis = font(6),
    yaxis = font(6),
    titlefontsize = 8,
    leg = false,
    grid = false,
    dpi = 300
)

n_sim = 1000
##########################################################################################################
#                                               RPPH
##########################################################################################################
preds_rpph = map(
    v -> compute_probs(TEDM(;
        # pᵣᵣ, pᵣₛ, pₛᵣ, pₛₛ
        p = [0, 0, 0, 1],
        ϵ = rand(Uniform(0, 0.5), 4),
        p_rep = rand()
    )),
    1:n_sim
)

preds_rpph = stack(preds_rpph, dims = 1)
rpph_plot = violin(
    preds_rpph;
    config...,
    title =  title = L"\mathcal{M}_{\textrm{RPPH}}" 
)
##########################################################################################################
#                                               DI
##########################################################################################################
preds_di = map(
    v -> compute_probs(TEDM(;
        p = rand(Dirichlet([5, 1, 1, 5])),
        ϵ = rand(Uniform(0, 0.5), 4),
        p_rep = rand()
    )),
    1:n_sim
)

di_plot = plot(preds_di; config..., title = L"\mathcal{M}_{\textrm{DI}}")

preds_di = stack(preds_di, dims = 1)
di_plot = violin(
    preds_di;
    config...,
    title =  title = L"\mathcal{M}_{\textrm{DI}}" 
)
##########################################################################################################
#                                           save plots
##########################################################################################################
plot(rpph_plot, di_plot, layout = (2, 1), size = (240, 300))
savefig("rpph_tedm_prior_predictive_distributions.png")
