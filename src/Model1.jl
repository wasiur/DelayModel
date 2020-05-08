"""
# module Model1

- Julia version: 1.4.1
- Author: Wasiur KhudaBukhsh
- Email: khudabukhsh.2@osu.edu
- Date: 2020-05-07

# Examples

```jldoctest
julia>
```
"""

module Model1
    using Random
    using Distributions
    using StatsBase
    using QuadGK

    mutable struct System
        nSpecies::Int64 #number of species
        nReaction::Int64 #number of reactions
        input_mtrx::Array{Int64,2} #matrix of input coefficients
        output_mtrx::Array{Int64,2} #matrix of output coefficients
        r::Array{Any,1} # Hazard functions for the reactions
        surv::Array{Any,1} #Survival functions for the reactions
        function System(nSpecies::Int64, nReaction::Int64, input_mtrx::Array{Int64,2}, output_mtrx::Array{Int64,2}, r::Array{Any,1}, surv::Array{Any,1} )
            dim_checks(nSpecies, nReaction, input_mtrx, output_mtrx, r, surv)
            return new(nSpecies, nReaction, input_mtrx, output_mtrx, r, surv)
        end
    end

    function dim_checks(nSpecies::Int64, nReaction::Int64, input_mtrx::Array{Int64,2}, output_mtrx::Array{Int64,2}, r::Array{Any,1}, surv::Array{Any,1})
        @assert size(input_mtrx,2) == nSpecies "Number of columns does not match the number of species"
        @assert size(input_mtrx,1) == nReaction "Number of rows does not match the number of reactions"
        @assert size(output_mtrx,2) == nSpecies "Number of columns does not match the number of species"
        @assert size(output_mtrx,1) == nReaction "Number of rows does not match the number of reactions"
        @assert size(r,1) == nReaction "Number of rows does not match number of reactions"
        @assert size(surv,1) == nReaction "Number of rows does not match number of reactions"
    end

    function initialize(network::System, n)
        x0 = Array{Any}(nothing,network.nSpecies)
        Threads.@threads for s in 1:1:network.nSpecies
            x0[s] = n[s] == 0 ? [] : rand(Exponential(),n[s]);
        end
        return x0
    end

    mutable struct Trajectory
        t::Array{Float64,1} #firing times
        x::Array{Any,1} #system states
        function Trajectory(t0, x0::Array{Any,1}, maxevents::Int64)
            t = zeros(Float64,maxevents+1)
            t[1] = float(t0)
            x = Array{Any}(nothing, maxevents+1)
            x[1] = x0
            return new(t, x)
        end
    end

    mutable struct CountTrajectory
        t::Array{Float64,1} #firing times
        y::Array{Any,1} #molecule counts of each species
        function CountTrajectory(path::Trajectory)
            t = path.t
            y = create_counts(path)
            return new(t,y)
        end
    end

    time_index(find_time, times) = isempty(findall(t -> t < find_time, times)) ? 1 : maximum(findall(t -> t < find_time, times))

    function create_counts(path::Trajectory)
        y = Array{Any}(nothing, size(path.x,1))
        Threads.@threads for t in eachindex(path.t)
            y[t] = measure_to_count(path.x[t])
        end
        return y
    end

    function create_counts(path::Trajectory, times)
        y = create_counts(path)
#         z = Array{Any}(nothing, size(times,1))
        z = zeros(Int64,size(times,1),size(y[1],1))
        Threads.@threads for i in eachindex(times)
            idx::Int64 = time_index(times[i], path.t)
#             z[i] = y[idx]
            z[i,:] = y[idx]
        end
        return z
    end

    function measure_to_count(x::Array{Any,1})
        nSpecies = size(x, 1)
        z = zeros(Int64,nSpecies)
        Threads.@threads for s in 1:1:nSpecies
            z[s] = isnothing(x[s]) ? 0 : size(x[s],1)
        end
        return z
    end

    function simulate_path(network::System, x0::Array{Any,1}, t0 = 0.0, maxT = 20.0, maxevents::Int64 = 1000)
        path = Trajectory(t0, x0, maxevents)
        time = t0
        curr_state = copy(x0)
        jump_counter = 1
        flag = true
        while flag
            hazards = compute_hazards(network.input_mtrx, curr_state, network.r)
            jump_time, reaction_idx = next_reaction(hazards)
            if jump_time == Inf || jump_counter == maxevents || time > maxT
                flag = false
                truncate_path!(path,jump_counter)
                break
            else
                 time += jump_time
                 jump_counter += 1
                 evolve_state!(path.t, path.x, jump_time, reaction_idx,jump_counter)
                 curr_state = copy(path.x[jump_counter])
             end
        end
        return path
    end

    function simulate_model1(network::System, n0 = [1000, 0]; nSim::Int64 = 100, t0 = 0.0, dt = 0.1, maxT = 10.0)
        t = t0:dt:maxT
        z = zeros(Int64, size(t,1), network.nSpecies, nSim)
        Threads.@threads for j in 1:1:nSim
            x0 = initialize(network, n0)
            path = simulate_path(network, x0)
            z[:,:,j] = create_counts(path,t)
        end
        return t, z
    end

    function compute_moments(sims)
        m = mean(sims, dims=3)
        m = m[:,:, 1]
        s = std(sims, mean = m, dims=3)
        s = s[:,:,1]
        return m, s
    end

    function truncate_path!(path::Trajectory, l)
        if l < size(path.t,1)
            del_indices = (l+1):1:size(path.t,1)
            deleteat!(path.t,del_indices)
            deleteat!(path.x,del_indices)
        end
    end

    function evolve_state!(t, x, jump_time, reaction_idx, jump_counter)
        t[jump_counter] = t[jump_counter-1] + jump_time
        curr_state = x[jump_counter-1]
        if reaction_idx[1] == 1
            x[jump_counter] = jump_1(curr_state, jump_time)
        elseif reaction_idx[1] == 2
            x[jump_counter] = jump_2(curr_state, jump_time, reaction_idx)
        else
            x[jump_counter] = jump_3(curr_state, jump_time, reaction_idx)
        end
    end

    function advance_age(x, amount)
        y = Array{Any}(nothing,size(x,1))
        Threads.@threads for s in 1:1:size(x,1)
            y[s] = x[s] .+ amount
        end
        return y
    end

    function jump_1(x, jump_time)
        y = advance_age(x, jump_time)
        push!(y[1], 0.0)
        return y
    end

    function jump_2(x, jump_time, reaction_idx)
        y = advance_age(x, jump_time)
        deleteat!(y[1], reaction_idx[2])
        push!(y[2], 0.0)
        return y
    end

    function jump_3(x, jump_time, reaction_idx)
        y = advance_age(x, jump_time)
        deleteat!(y[1], reaction_idx[2])
        return y
    end

    function compute_hazards(input_mtrx::Array{Int64,2}, x::Array{Any,1}, r)
        hazards = Array{Any}(nothing,3)
        hazards[1] = r[1].(1) #correponds to the first reaction
        hazards[2] = size(x[1],1) == 0 ? 0.0 : r[2].(x[1]) #corresponds to the second reaction
        hazards[3] = size(x[1],1) == 0 ? 0.0 : r[3].(x[1]) #corresponds to the third reaction
        return hazards
    end

    total_hazard(hazards) = sum(sum(hazards[i]) for i in 1:1:size(hazards,1))

    function next_reaction(hazards)
        jump_time = next_reaction_time(hazards)
        reaction_idx = which_reaction(hazards)
        return jump_time, reaction_idx
    end

    next_reaction_time(hazards) = total_hazard(hazards) > 0.0 ? -1/total_hazard(hazards)*log(rand()) : Inf

    function which_reaction(hazards)
        prob_vector = vcat(hazards[1], hazards[2], hazards[3]) ./ total_hazard(hazards)
        idx =  sample_categorical(prob_vector)
        if idx == 1
            return 1
        elseif idx <= 1 + size(hazards[2],1)
            offset = idx - 1
            return [2, offset]
        else
            offset = idx - 1 - size(hazards[2],1)
            return [3, offset]
        end
    end

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

    function plot_hazards(network::System, t = 0.0:0.1:10.0)
        PyPlot.plot(t,r[1].(t))
        PyPlot.plot(t,r[2].(t))
        PyPlot.plot(t,r[3].(t))
        legend(["r1", "r2", "r3"])
    end

    function solve_PDEs(network::System, n, t0=0.0, dt=0.1, T=10.0)
        surv = network.surv
        r = network.r
        y_a(t,s) = s >= t ? (exp(-(s-t)) * surv[2](s) * surv[3](s) / (surv[2](s-t) * surv[3](s-t))) : ( (r[1](1)/n)*surv[2](s)*surv[3](s) )
        times = t0:dt:T
        sol_a = solve_a(times, y_a)
        ds = 0.01
        bbb(t) = sum(sum(r[2](u)*y_a(s,u) for u in 0.0:ds:50.0) for s in 0.0:ds:t)*(ds^2)
        sol_b = bbb.(times)
        return times, sol_a, sol_b
    end

    function solve_a(t, a)
        sol_a = zeros(Float64,size(t,1));
        Threads.@threads for i in eachindex(t)
            ti = t[i]
            sol_a[i], = quadgk(s -> a(ti,s), 0.0, 50.0)
        end
    return sol_a
    end

end


