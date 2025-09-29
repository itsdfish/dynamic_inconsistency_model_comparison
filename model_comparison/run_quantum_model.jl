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
using MCMCChains
using Pigeons
using PriorityHeuristicModels
using QuantumDynamicInconsistencyModels
using ReferencePointModels
using Random
using StatsPlots
using Turing
include("data_parsing_functions.jl")
include("turing_models.jl")
##########################################################################################################
#                                               load data
##########################################################################################################
# load the data
df = CSV.read("../Barkan_2003_data/Barkan_2003_data.csv", DataFrame)
df_stacked = parse_barkan_data(df)

parms = (α = 0.9, λ = 2, w₁ = 0.5, p_rep = 0.30, γ = 2.5)

df_m_ll = DataFrame(subject = Int[], m_ll = Float64[])
for df_subj ∈ groupby(df_stacked, :subject)
    subj_id = df_subj.subject[1]
    Random.seed!(subj_id)
    println("estimating parameters for subject $subj_id")
    data = format_data(df_subj, QDIM)
    estimator = qdim_model(data, parms)
    pt = pigeons(
        target = TuringLogPotential(estimator),
        record = [traces],
        multithreaded = true
    )
    # mcmc chain 
    chain = Chains(pt)
    # marginal log likelihood
    push!(df_m_ll, [subj_id, stepping_stone(pt)])
    CSV.write(
        "qdim_output/marginal_log_likelihood/qdim_marginal_log_likelihood.csv",
        df_m_ll
    )
    posterior_plots = plot(chain)
    savefig(
        posterior_plots,
        "qdim_output/posterior_plots/subj_$(subj_id)_posterior_plots.png"
    )
    jldsave(
        "qdim_output/pigeons_output/subject_$(subj_id)_output.jld2";
        subj_id,
        chain,
        pt,
        data
    )
end
