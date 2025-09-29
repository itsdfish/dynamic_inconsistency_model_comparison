"""
    qdim_model(data, default_parms)

A Turing model for the Quantum Dynamic Inconsistency Model 

# Arguments 

- `data::Tuple`: a tuple containing the following data elements: outcomes1, outcomes2, win_gamble1, responses
- `default_parms`: a `NamedTuple` of default pararmeter values which can be over written 
"""
@model function qdim_model(data::Tuple, default_parms)
    α ~ truncated(Normal(1, 1), 0, 3)
    λ ~ truncated(Normal(1, 2), 1, 6)
    γ ~ Normal(0, 10)
    p_rep ~ Beta(1, 1)
    #println("α $α λ $λ γ $γ p_rep $p_rep")
    data ~ QDIM(; default_parms..., γ, λ, α, p_rep)
end

"""
    qdim_uniform_model(data, default_parms)

A Turing model for the Quantum Dynamic Inconsistency Model.  This model uses uniform prior distributions.

# Arguments 

- `data::Tuple`: a tuple containing the following data elements: outcomes1, outcomes2, win_gamble1, responses
- `default_parms`: a `NamedTuple` of default pararmeter values which can be over written 
"""
@model function qdim_uniform_model(data::Tuple, default_parms)
    α ~ Uniform(0, 3)
    λ ~ Uniform(1, 6)
    γ ~ Uniform(-20, 20)
    p_rep ~ Beta(1, 1)
    #println("α $α λ $λ γ $γ p_rep $p_rep")
    data ~ QDIM(; default_parms..., γ, λ, α, p_rep)
end

"""
    reference_point_model(data, default_parms)

A Turing model for the Reference Point Model 

# Arguments 

- `data::Tuple`: a tuple containing the following data elements: outcomes1, outcomes2, win_gamble1, responses
- `default_parms`: a `NamedTuple` of default pararmeter values which can be over written 
"""
@model function reference_point_model(data::Tuple, default_parms)
    α ~ truncated(Normal(1, 1), 0, 3)
    λ ~ truncated(Normal(1, 2), 1, 6)
    θ ~ truncated(Normal(1, 0.5), 0, 2)
    p_rep ~ Beta(1, 1)
    #dist = ReferencePointModel(; default_parms..., α, θ, p_rep)
    #println("α $α λ $λ θ $θ p_rep $p_rep LL $(loglikelihood(dist, data))")
    data ~ ReferencePointModel(; default_parms..., α, λ, θ, p_rep)
end

"""
    priority_model(data::Tuple, default_parms = ())

A Turing model for the Reference Point Priority Heuristic model.

# Arguments 

- `data::Tuple`: a tuple containing the following data elements: outcomes1, outcomes2, win_gamble1, responses
- `default_parms`: a `NamedTuple` of default pararmeter values which can be over written 
"""
@model function priority_model(data::Tuple, default_parms = ())
    σ ~ truncated(Normal(1, 2), 0, Inf)
    δoutcome ~ Beta(1, 9)
    δprob ~ Beta(1, 9)
    p_rep ~ Beta(1, 1)
    #println("(; σ=$σ, σprob=$σprob, δoutcome=$δoutcome, δprob=$δprob, p_rep=$p_rep) LL $(loglikelihood(PriorityModel(; default_parms..., p_rep, σmin = σ, σmax = σ, σprob, δprob, δoutcome), data))")
    data ~ PriorityModel(;
        default_parms...,
        σmin = σ,
        σmax = σ,
        σprob = σ,
        δprob,
        δoutcome,
        p_rep
    )
end

"""
    priority_uniform_model(data::Tuple, default_parms = ())

A Turing model for the reference point priority heuristic model. This model uses uniform prior distributions. 

# Arguments 

- `data::Tuple`: a tuple containing the following data elements: outcomes1, outcomes2, win_gamble1, responses
- `default_parms`: a `NamedTuple` of default pararmeter values which can be over written 
"""
@model function priority_uniform_model(data::Tuple, default_parms = ())
    σ ~ Uniform(0, 10)
    δoutcome ~ Beta(1, 1)
    δprob ~ Beta(1, 1)
    p_rep ~ Beta(1, 1)
    #println("(; σ=$σ, σprob=$σprob, δoutcome=$δoutcome, δprob=$δprob, p_rep=$p_rep) LL $(loglikelihood(PriorityModel(; default_parms..., p_rep, σmin = σ, σmax = σ, σprob, δprob, δoutcome), data))")
    data ~ PriorityModel(;
        default_parms...,
        σmin = σ,
        σmax = σ,
        σprob = σ,
        δprob,
        δoutcome,
        p_rep
    )
end

"""
    reference_point_uniform_model(data, default_parms)

A Turing model for the Reference Point Prospect Theory Model. This model uses uniform prior distributions.  

# Arguments 

- `data::Tuple`: a tuple containing the following data elements: outcomes1, outcomes2, win_gamble1, responses
- `default_parms`: a `NamedTuple` of default pararmeter values which can be over written 
"""
@model function reference_point_uniform_model(data::Tuple, default_parms)
    α ~ Uniform(0, 3)
    λ ~ Uniform(1, 6)
    θ ~ Uniform(0, 2)
    p_rep ~ Beta(1, 1)
    #dist = ReferencePointModel(; default_parms..., α, θ, p_rep)
    #println("α $α λ $λ θ $θ p_rep $p_rep LL $(loglikelihood(dist, data))")
    data ~ ReferencePointModel(; default_parms..., α, λ, θ, p_rep)
end
