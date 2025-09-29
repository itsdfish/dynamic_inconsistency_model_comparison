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
using Random
using StatsPlots
using TrueAndErrorDependentModels
using TrueAndErrorModels
using Turing
include("true_error_model_utility_functions.jl")
include("data_parsing_functions.jl")
Random.seed!(5874)
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
#                                               estimate parameters
##########################################################################################################
df_results = DataFrame(
    id = Int[],
    gain = Int[],
    loss = Int[],
    stage1_outcome = String[],
    HDI = Vector{Tuple{Float64, Float64}}()
)
cnt = 0
gamble_id = 0
for df_gamble ∈ eachrow(df_counts)
    cnt += 1
    gamble_id += 1
    # skip non-applicable gambles 
    df_rpph.prediction[cnt] ≠ "SS" ? continue : nothing
    # skip id 9 in the 
    gamble_id = cnt == 9 ? 10 : gamble_id
    println("estimating parameters for gamble $cnt")
    data = df_gamble.counts
    chains = sample(multinomal_model(data), NUTS(1000, 0.65), MCMCThreads(), 1000, 4)
    diffs = generated_quantities(multinomal_model(data), chains)
    diff_chains = Chains(reshape(diffs, 1000, 1, 4))
    hdi = hpd(diff_chains)
    hdi_results = round.((hdi.nt.lower[1], hdi.nt.upper[1]), digits = 3)
    push!(
        df_results,
        [gamble_id df_gamble.gain df_gamble.loss df_gamble.stage1_outcome hdi_results]
    )
end
# add padding for split columns 
push!(df_results, [999999 0 0 "dfd" (-Inf, -Inf)])
##########################################################################################################
#                                 compute decision difficulty
##########################################################################################################
df_resuts_final = select(df_results, [:id, :HDI])
df_resuts_final =
    hcat(df_resuts_final[1:7, :], df_resuts_final[8:end, :]; makeunique = true)
CSV.write("critical_inequality_test_results.csv", df_resuts_final)
