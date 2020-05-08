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
using Revise


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
end


