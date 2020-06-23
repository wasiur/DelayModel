"""
# module DelayModel

- Julia version: 1.4.1
- Author: Wasiur KhudaBukhsh
- Email: khudabukhsh.2@osu.edu
- Date: 2020-05-07

# Examples

```jldoctest
julia>
```
"""

module DelayModel
    using Random
    using Distributions
    using StatsBase
    using DataFrames


    hazard_function(d, t) = Distributions.pdf(d,t)/(1 - Distributions.cdf(d,t))
    survival_function(d, t) = 1 - Distributions.cdf(d,t)
    riemann_integral(f, t0=0.0, T=10.0, dt=0.1) = dt * sum(f.(t0:dt:T))

    function sample_categorical(prob_vector::Array, r::Float64 = rand())
        idx::Int64 = 1
        cum_sum::Float64 = prob_vector[1]
        while r > cum_sum
            idx == size(prob_vector,1) ? break : nothing
            idx += 1
            cum_sum += prob_vector[idx]
        end
        return idx
    end

    function hazard_survival(dist)
        nReaction = size(dist,1)
        r = Array{Any}(nothing,nReaction)
        surv = Array{Any}(nothing, nReaction)
        for i in 1:nReaction
            r[i] = t -> DelayModel.hazard_function(dist[i],t)
            surv[i] = t -> DelayModel.survival_function(dist[i],t)
        end
        return r, surv
    end


    function str2parm(s, sep=",")
        res = replace(s, "[" => "")
        res = replace(res, "]" => "")
        res = split(res, sep)
        return parse.(Float64, res)
    end

    function distparm2str(s)
        res = replace(s, "[" => "")
        res = replace(res, "]" => "")
        res = replace(res, "," => "_")
        return res
    end
end
