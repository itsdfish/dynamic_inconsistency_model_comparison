function code_response_category(df)
    if (df.replicate_id[1] ≠ 1) || (df.replicate_id[2] ≠ 2)
        println(df)
        error("replicate ids are not sorted")
    end
    # 1.  RR,RR
    if (df.RR[1] == 1) && (df.RR[2] == 1)
        return 1
        # 2.  RR,RS
    elseif (df.RR[1] == 1) && (df.RS[2] == 1)
        return 2
        # 3.  RR,SR
    elseif (df.RR[1] == 1) && (df.SR[2] == 1)
        return 3
        # 4.  RR,SS
    elseif (df.RR[1] == 1) && (df.SS[2] == 1)
        return 4
        # 5.  RS,RR
    elseif (df.RS[1] == 1) && (df.RR[2] == 1)
        return 5
        # 6.  RS,RS
    elseif (df.RS[1] == 1) && (df.RS[2] == 1)
        return 6
        # 7.  RS,SR
    elseif (df.RS[1] == 1) && (df.SR[2] == 1)
        return 7
        # 8.  RS,SS
    elseif (df.RS[1] == 1) && (df.SS[2] == 1)
        return 8
        # 9.  SR,RR
    elseif (df.SR[1] == 1) && (df.RR[2] == 1)
        return 9
        # 10. SR,RS
    elseif (df.SR[1] == 1) && (df.RS[2] == 1)
        return 10
        # 11. SR,SR
    elseif (df.SR[1] == 1) && (df.SR[2] == 1)
        return 11
        # 12. SR,SS
    elseif (df.SR[1] == 1) && (df.SS[2] == 1)
        return 12
        # 13. SS,RR
    elseif (df.SS[1] == 1) && (df.RR[2] == 1)
        return 13
        # 14. SS,RS
    elseif (df.SS[1] == 1) && (df.RS[2] == 1)
        return 14
        # 15. SS,SR
    elseif (df.SS[1] == 1) && (df.SR[2] == 1)
        return 15
        # 16. SS,SS
    elseif (df.SS[1] == 1) && (df.SS[2] == 1)
        return 16
    end
end

count_responses(df) = map(x -> sum(df.response_id .== x), 1:16)

@model function tet4_model_small_effect(data::Vector{<:Integer})
    # [pᵣᵣ, pᵣₛ, pₛᵣ, pₛₛ]
    p ~ Dirichlet([5, 1, 1, 5])
    # ϵₛ₁, ϵₛ₂, ϵᵣ₁, ϵᵣ₂
    ϵ ~ filldist(Uniform(0, 0.5), 4)
    data ~ TrueErrorModel(; p, ϵ)
end

@model function tet4_depedency_model_small_effect(data::Vector{<:Integer})
    # [pᵣᵣ, pᵣₛ, pₛᵣ, pₛₛ]
    p ~ Dirichlet([5, 1, 1, 5])
    # ϵₛ₁, ϵₛ₂, ϵᵣ₁, ϵᵣ₂
    ϵ ~ filldist(Uniform(0, 0.5), 4)
    p_rep ~ Uniform(0, 1)
    data ~ TEDM(; p, ϵ, p_rep)
    return (; p, ϵ, p_rep)
end

@model function tet4_depedency_model_uniform(data::Vector{<:Integer})
    # [pᵣᵣ, pᵣₛ, pₛᵣ, pₛₛ]
    p ~ Dirichlet([1, 1, 1, 1])
    # ϵₛ₁, ϵₛ₂, ϵᵣ₁, ϵᵣ₂
    ϵ ~ filldist(Uniform(0, 0.5), 4)
    p_rep ~ Uniform(0, 1)
    data ~ TEDM(; p, ϵ, p_rep)
    return (; p, ϵ, p_rep)
end

@model function rpph_model(data::Vector{<:Integer}, pred_idx)
    # [pᵣᵣ, pᵣₛ, pₛᵣ, pₛₛ]
    p = fill(0.0, 4)
    p[pred_idx] = 1
    # ϵₛ₁, ϵₛ₂, ϵᵣ₁, ϵᵣ₂
    ϵ ~ filldist(Uniform(0, 0.5), 4)
    data ~ TrueErrorModel(; p, ϵ)
    return (; p, ϵ)
end

@model function rpph_dependency_model(data::Vector{<:Integer}, pred_idx)
    # [pᵣᵣ, pᵣₛ, pₛᵣ, pₛₛ]
    p = fill(0.0, 4)
    p[pred_idx] = 1
    p_rep ~ Uniform(0, 1)
    ϵ ~ filldist(Uniform(0, 0.5), 4)
    # ϵₛ₁, ϵₛ₂, ϵᵣ₁, ϵᵣ₂
    data ~ TEDM(; p, ϵ, p_rep)
    return (; p, ϵ, p_rep)
end

# used to test the critical inequality of the RPPH TEDM
@model function multinomal_model(data)
    k = length(data)
    n = sum(data)
    θ ~ Dirichlet(fill(1, k))
    data ~ Multinomial(n, θ)
    return (ρ = θ[1] - θ[end])
end

function get_rpph_pred_index(x)
    # [pᵣᵣ, pᵣₛ, pₛᵣ, pₛₛ]
    if x == "RR"
        return 1
    elseif x == "RS"
        return 2
    elseif x == "SR"
        return 3
    elseif x == "SS"
        return 4
    end
end

"""
    compute_difficulty(df)

Computes the difficulty of a decision based on a signal to noise comparison between two gambles. This is valid when the safe option is 
zero. On final decisions, a constant is added to each outcome in both options, and therefore does not affect the calculation. 
"""
function compute_difficulty(df)
    return map(r -> _compute_difficulty(r), eachrow(df))
end

function _compute_difficulty(df_row)
    # for final decisions, the outcome is added to all outcomes in both options. Therefore, 
    # it's effect is canceled in the ev difference, and does not change the variance
    ev_risky = 0.50 * df_row.gain + 0.50 * df_row.loss
    sd = sqrt(0.50 * (df_row.gain - ev_risky)^2 + 0.50 * (df_row.loss - ev_risky)^2)
    return ev_risky / sd
end
