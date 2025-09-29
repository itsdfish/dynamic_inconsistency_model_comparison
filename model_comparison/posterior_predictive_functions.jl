function to_named_tuples(chain)
    parms = (Symbol.(chain.name_map.parameters)...,)
    samples = Array(chain)
    return [NamedTuple{parms}((r...,)) for r ∈ eachrow(samples)]
end

"""
    posterior_predictive(model, chain, n_samples::Int, f=x -> x)

Returns posterior predictive distribution and optionally applies function to samples on each replication.

# Arguments

- `model`: the data generating function of a model 
- `chain`: an MCMCChains chain object
- `n_samples`: the number of samples 
- `f`: a function that is applied to each sample from posterior predictive distribution
"""
function posterior_predictive(model, samples, n_reps::Int; sim_func, kwargs...)
    return map(_ -> simulate_group(model, samples; sim_func, kwargs...), 1:n_reps)
end

"""
    simulate_group(model, samples; sim_func, kwargs...)

Simulates group mean of joint probability predictions for a given model. For each subject, a NamedTuple is sampled randomly 
from the posterior distribution of parameters. 

# Arguments

- `model`: a model type (e.g., QDIM for quantum cognition)
- `samples`: a nested vector of posterior parameter samples. The first level represents subjects and within each subject level there is a vector of 
    `NamedTuples` corresponding to posterior samples for that subject. 
# Keywords

- `sim_func`: a `simulation` function which returns predicted joint response probabilities 
- `kwargs`: optional keyword arguments 
"""
function simulate_group(model, samples; sim_func, kwargs...)
    return map(s -> sim_func(model, rand(s); kwargs...), samples) |> mean
end

"""
    simulate(
        model_type::Type{<:QDIM},
        parms;
        default_parms,
        outcomes1,
        outcomes2,
        won_first,
        n_trials
    )


Generates predictions for joint probabilities of a given gamble (trial). 

# Arguments

- `model_type::Type{<:QDIM}`: a model type for reference point prospect theory
- `parms`: a `NamedTuple` of model parameters

# Keywords

- `default_parms`: default parameters which can be overwritten by `parms`
- `outcomes1::Vector{<:Real}`: outcomes for the first gamble
- `outcomes2::Vector{<:Real}`: outcomes for the second gamble
- `won_first`: set true if first gamble was won, in which case evaluation of final decision
    is conditioned on winning first gamble 
- `n_trials`: number of trials (i.e. reps to simulation)
"""
function simulate(
    model_type::Type{<:QDIM},
    parms;
    default_parms,
    outcomes1,
    outcomes2,
    won_first,
    n_trials
)
    model = model_type(; default_parms..., parms...)
    data = rand(model, outcomes1, outcomes2, won_first, n_trials)
    return data ./ n_trials
end

"""
    simulate(
        model_type::Type{<:ReferencePointModel},
        parms;
        default_parms,
        outcomes1,
        outcomes2,
        won_first,
        n_trials
    )


Generates predictions for joint probabilities of a given gamble (trial). 

# Arguments

- `model_type::Type{<:ReferencePointModel}`: a model type for reference point prospect theory
- `parms`: a `NamedTuple` of model parameters

# Keywords

- `default_parms`: default parameters which can be overwritten by `parms`
- `outcomes1::Vector{<:Real}`: outcomes for the first gamble
- `outcomes2::Vector{<:Real}`: outcomes for the second gamble
- `won_first`: set true if first gamble was won, in which case evaluation of final decision
    is conditioned on winning first gamble 
- `n_trials`: number of trials (i.e. reps to simulation)
"""
function simulate(
    model_type::Type{<:ReferencePointModel},
    parms;
    default_parms,
    outcomes1,
    outcomes2,
    won_first,
    n_trials
)
    model = model_type(; default_parms..., parms...)
    data = rand(model, outcomes1, outcomes2, won_first, n_trials)
    return data ./ n_trials
end

"""
    simulate(
        model_type::Type{<:PriorityModel},
        parms;
        default_parms,
        outcome_stage1,
        outcome1_stage2,
        outcome2_stage2,
        outcome1_prob_stage2,
        n_trials
    )


Generates predictions for joint probabilities of a given gamble (trial). 

# Arguments

- `model_type::Type{<:PriorityModel}`: a model type for SRPPH
- `parms`: a `NamedTuple` of model parameters

# Keywords

- `default_parms`: default parameters which can be overwritten by `parms`
- `outcome_stage1`: the observed or hypothetical outcome from stage 1
- `outcome1_stage2`: the first outcome for attacking 
- `outcome2_stage2`: the second outcome for attacking 
- `outcome1_prob_stage2`: the probability of outcome 1 in stage 2
- `n_trials`: number of trials (i.e. reps to simulation)
"""
function simulate(
    model_type::Type{<:PriorityModel},
    parms;
    default_parms,
    outcome_stage1,
    outcome1_stage2,
    outcome2_stage2,
    outcome1_prob_stage2,
    n_trials
)
    mod_parms = (
        σmin = parms.σ,
        σmax = parms.σ,
        σprob = parms.σ,
        δprob = parms.δprob,
        δoutcome = parms.δoutcome,
        p_rep = parms.p_rep
    )

    model = model_type(; default_parms..., mod_parms...)
    data = rand(
        model,
        outcome_stage1,
        outcome1_stage2,
        outcome2_stage2,
        outcome1_prob_stage2,
        n_trials
    )
    return data ./ n_trials
end

"""
    generate_post_pred_gamble(
        model_type::Type{<:QDIM},
        samples,
        df_row,
        n_samples;
        default_parms,
        sim_func
    )


Generates posterior predictive distribution of joint probabilities for a given gamble (trial). 

# Arguments

- `model_type::Type{<:QDIM}`: quantum cognition model type
- `samples`: a nested vector of posterior parameter samples. The first level represents subjects and within each subject level there is a vector of 
`NamedTuples` corresponding to posterior samples for that subject. 
- `df_row`: a `DataFrameRow` containing gamble attributes and number of trials (reps) per subject
- `n_samples`: number of posterior predictive samples 

# Keywords

- `default_parms`: a `NamedTuple` of default model parameters which can be overwritten
- `sim_func`: a `simulation` function which returns predicted joint response probabilities 
"""
function generate_post_pred_gamble(
    model_type::Type{<:QDIM},
    samples,
    df_row,
    n_samples;
    default_parms,
    sim_func
)
    outcomes1 = [df_row.gain, df_row.loss] ./ 100
    outcomes2 = outcomes1
    won_first = df_row.stage1_outcome == "x_G"
    n_trials = df_row.n_trials
    post_preds = posterior_predictive(
        model_type,
        samples,
        n_samples;
        sim_func,
        default_parms,
        outcomes1,
        outcomes2,
        won_first,
        n_trials
    )
    matrix_post_preds = stack(post_preds, dims = 1)
    mean_joint_probs = mean(matrix_post_preds, dims = 1)[:]
    chain = Chains(matrix_post_preds)
    hdps = hpd(chain)
    lbs = map((lb, m) -> m - lb, hdps.nt.lower, mean_joint_probs)
    ubs = map((ub, m) -> ub - m, hdps.nt.upper, mean_joint_probs)
    return (; means = mean_joint_probs, lbs, ubs, matrix_post_preds)
end

"""
    generate_post_pred_gamble(
        model_type::Type{<:ReferencePointModel},
        samples,
        df_row,
        n_samples;
        default_parms,
        sim_func
    )


Generates posterior predictive distribution of joint probabilities for a given gamble (trial). 

# Arguments

- `model_type::Type{<:ReferencePointModel}`: reference point prospect theory type
- `samples`: a nested vector of posterior parameter samples. The first level represents subjects and within each subject level there is a vector of 
`NamedTuples` corresponding to posterior samples for that subject. 
- `df_row`: a `DataFrameRow` containing gamble attributes and number of trials (reps) per subject
- `n_samples`: number of posterior predictive samples 

# Keywords

- `default_parms`: a `NamedTuple` of default model parameters which can be overwritten
- `sim_func`: a `simulation` function which returns predicted joint response probabilities 
"""
function generate_post_pred_gamble(
    model_type::Type{<:ReferencePointModel},
    samples,
    df_row,
    n_samples;
    default_parms,
    sim_func
)
    outcomes1 = [df_row.gain, df_row.loss] ./ 100
    outcomes2 = outcomes1
    won_first = df_row.stage1_outcome == "x_G"
    n_trials = df_row.n_trials
    post_preds = posterior_predictive(
        model_type,
        samples,
        n_samples;
        sim_func,
        default_parms,
        outcomes1,
        outcomes2,
        won_first,
        n_trials
    )
    matrix_post_preds = stack(post_preds, dims = 1)
    mean_joint_probs = mean(matrix_post_preds, dims = 1)[:]
    chain = Chains(matrix_post_preds)
    hdps = hpd(chain)
    lbs = map((lb, m) -> m - lb, hdps.nt.lower, mean_joint_probs)
    ubs = map((ub, m) -> ub - m, hdps.nt.upper, mean_joint_probs)
    return (; means = mean_joint_probs, lbs, ubs, matrix_post_preds)
end

"""
    generate_post_pred_gamble(
        model_type::Type{<:PriorityModel},
        samples,
        df_row,
        n_samples;
        default_parms,
        sim_func
    )


Generates posterior predictive distribution of joint probabilities for a given gamble (trial). 

# Arguments

- `model_type::Type{<:PriorityModel}`: priority model (SRRPH) model type
- `samples`: a nested vector of posterior parameter samples. The first level represents subjects and within each subject level there is a vector of 
`NamedTuples` corresponding to posterior samples for that subject. 
- `df_row`: a `DataFrameRow` containing gamble attributes and number of trials (reps) per subject
- `n_samples`: number of posterior predictive samples 

# Keywords

- `default_parms`: a `NamedTuple` of default model parameters which can be overwritten
- `sim_func`: a `simulation` function which returns predicted joint response probabilities 
"""
function generate_post_pred_gamble(
    model_type::Type{<:PriorityModel},
    samples,
    df_row,
    n_samples;
    default_parms,
    sim_func
)
    outcome_stage1 = df.stage1_outcome == "x_G" ? df_row.gain : df_row.loss
    outcome1_stage2 = df_row.gain
    outcome2_stage2 = df_row.loss
    outcome1_prob_stage2 = 0.5
    n_trials = df_row.n_trials

    post_preds = posterior_predictive(
        model_type,
        samples,
        n_samples;
        sim_func,
        default_parms,
        outcome_stage1,
        outcome1_stage2,
        outcome2_stage2,
        outcome1_prob_stage2,
        n_trials
    )

    matrix_post_preds = stack(post_preds, dims = 1)
    mean_joint_probs = mean(matrix_post_preds, dims = 1)[:]
    chain = Chains(matrix_post_preds)
    hdps = hpd(chain)
    lbs = map((lb, m) -> m - lb, hdps.nt.lower, mean_joint_probs)
    ubs = map((ub, m) -> ub - m, hdps.nt.upper, mean_joint_probs)
    return (; means = mean_joint_probs, lbs, ubs, matrix_post_preds)
end

rmse(x, y) = sqrt(mean((x .- y) .^ 2))
