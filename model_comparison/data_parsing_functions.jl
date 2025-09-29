"""
    format_data(df, ::Type{<:QDIM})

Formats subject-level data for the QDIM. 

# Arguments 

- `df`: a dataframe in the form below
- `::Type{<:QDIM}`: QDIM model Type

# DataFrame 

# Returns 

Returns a tuple in which elements correspond to 

1. `outcomes1`: outcomes for stage 1 [win, loss]
2. `outcomes2`: outcomes for stage 2 [win, loss] 
3. `win_gamble1`: for a given gamble indicators whether the first gamble was won
4. `responses`: a vector of response frequencies
"""
function format_data(df, ::Type{<:QDIM})
    group = combine(
        groupby(df, [:stage_pay, :gain, :loss]),
        :RR => sum => :RR,
        :RS => sum => :RS,
        :SR => sum => :SR,
        :SS => sum => :SS
    )

    outcomes1 = map(r -> [r.gain, r.loss], eachrow(group)) ./ 100
    # same as outcomes1 in this experiment
    outcomes2 = map(r -> [r.gain, r.loss], eachrow(group)) ./ 100
    win_gamble1 = map(r -> r.stage_pay > 0, eachrow(group))
    responses = map(r -> [r.RR, r.RS, r.SR, r.SS], eachrow(group))
    return outcomes1, outcomes2, win_gamble1, responses
end

"""
    format_data(df, ::Type{<:ReferencePointModel})

Formats subject-level data for the ReferencePointModel. 

# Arguments 

- `df`: a dataframe in the form below
- `::Type{<:ReferencePointModel}`: ReferencePointModel model Type

# DataFrame 

# Returns 

Returns a tuple in which elements correspond to 

1. `outcomes1`: outcomes for stage 1 [win, loss]
2. `outcomes2`: outcomes for stage 2 [win, loss] 
3. `win_gamble1`: for a given gamble indicators whether the first gamble was won
4. `responses`: a vector of response frequencies
"""
function format_data(df, ::Type{<:ReferencePointModel})
    group = combine(
        groupby(df, [:stage_pay, :gain, :loss]),
        :RR => sum => :RR,
        :RS => sum => :RS,
        :SR => sum => :SR,
        :SS => sum => :SS
    )

    outcomes1 = map(r -> [r.gain, r.loss], eachrow(group)) ./ 100
    # same as outcomes1 in this experiment
    outcomes2 = map(r -> [r.gain, r.loss], eachrow(group)) ./ 100
    win_gamble1 = map(r -> r.stage_pay > 0, eachrow(group))
    responses = map(r -> [r.RR, r.RS, r.SR, r.SS], eachrow(group))
    return outcomes1, outcomes2, win_gamble1, responses
end

"""
    format_data(df, ::Type{<:PriorityModel})

Formats subject-level data for the PriorityModel. 

# Arguments 

- `df`: a dataframe in the form below
- `::Type{<:PriorityModel}`: PriorityModel model Type

# DataFrame 

# Returns 

Returns a tuple in which elements correspond to 

1. `outcomes1`: outcomes for stage 1 [win, loss]
2. `outcomes2`: outcomes for stage 2 [win, loss] 
3. `win_gamble1`: for a given gamble indicators whether the first gamble was won
4. `responses`: a vector of response frequencies
"""
function format_data(df, ::Type{<:PriorityModel})
    group = combine(
        groupby(df, [:stage_pay, :gain, :loss]),
        :RR => sum => :RR,
        :RS => sum => :RS,
        :SR => sum => :SR,
        :SS => sum => :SS
    )

    outcome_stage1 = map(r -> r.stage_pay, eachrow(group))
    outcome1_stage2 = map(r -> r.gain, eachrow(group))
    outcome2_stage2 = map(r -> r.loss, eachrow(group))
    outcome1_prob_stage2 = 0.5
    responses = map(r -> [r.RR, r.RS, r.SR, r.SS], eachrow(group))
    return outcome_stage1, outcome1_stage2, outcome2_stage2, outcome1_prob_stage2, responses
end

"""
    parse_barkan_data(df)

Parses Barkan et al. (2003) data by stacking the loss and gain trials. 

# Returns 

Returns a DataFrame in which each row is a joint response for a planned and final decision of a given gamble. The columns are defined as follows:

- `subject`: a subject id ranging from 1 to 100
- `trial_id`: a trial indicator ranging from 1 to 33, in the order presented in the experiment. Each subject 
    received the same randomized order.
- `gain`: the potential gain for both stages 
- `loss`: the potential loss for both stages
- `trial_pay`: the amount awarded for each trial 
- `stage_pay`: the amount awarded for each stage 
- `RR`: indicates whether the risky option was selected for both the planned and final decision 
- `RS`: indicates whether the risky option selected for the planned decision, and the safe option for was selected for the final decison.
- `SR`: indicates whether the safe option selected for the planned decision, and the risky option for was selected for the final decison.
- `SS`: indicates whether the safe option was selected for both the planned and final decision 
- `stage1_outcome`: indicates whether the outcome experienced in the first stage was a gain (x_G), or a loss (x_L)
- `gamble_id`: a gamble id ranging from 0 to 16. Gamble id 0 corresponds to a practice trial which was repeated once.
- `replicate_id`: indexes the replicate order of each gamble. 

"""
function parse_barkan_data(df)
    # data parsing strategy: stack wins and losses using common joint response variables
    # identify gambles in which a loss occurred in stage 1
    get_outcome_type(x) = x > 0 ? "x_L" : "x_G"
    df.stage1_outcome =
        get_outcome_type.(
            df.lose_plan_take_final_take +
            df.lose_plan_take_final_reject +
            df.lose_plan_reject_final_take +
            df.lose_plan_reject_final_reject
        )

    # dataframe of trails with win in first stage 
    df_win = filter(x -> x.stage1_outcome == "x_G", df)
    df_win = df_win[:, [1:10..., end]]
    # new column names 
    # take gamble -> R (risky option)
    # reject gamble -> S (safe option)
    # data for model
    # 1. probability of planning to accept second gamble and accepting second gamble (RR)
    # 2. probability of planning to accept second gamble and rejecting second gamble (RS)
    # 3. probability of planning to reject second gamble and accepting second gamble (SR)
    # 4. probability of planning to reject second gamble and rejecting second gamble (SS)
    win_map = Dict(
        :win_plan_take_final_take => :RR,
        :win_plan_take_final_reject => :RS,
        :win_plan_reject_final_take => :SR,
        :win_plan_reject_final_reject => :SS
    )
    # rename columns, remove "win"
    rename!(df_win, win_map)

    # dataframe containing only trial with a loss in the first stage
    df_lose = filter(x -> x.stage1_outcome == "x_L", df)
    df_lose = df_lose[:, [1:6..., 11:end...]]
    # new column names 
    # take gamble -> R (risky option)
    # reject gamble -> S (safe option)
    lose_map = Dict(
        :lose_plan_take_final_take => :RR,
        :lose_plan_take_final_reject => :RS,
        :lose_plan_reject_final_take => :SR,
        :lose_plan_reject_final_reject => :SS
    )
    rename!(df_lose, lose_map)
    # combine losses and gains 
    df_combined = vcat(df_win, df_lose)
    sort!(df_combined, [:subject, :trial_id])
    df_combined.gamble_id = repeat(vcat(0:16, 1:16), outer = 100)
    df_combined.replicate_id = repeat(vcat(fill(1, 17), fill(2, 16)), outer = 100)
    return df_combined
end
