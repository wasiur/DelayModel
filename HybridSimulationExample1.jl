#=
InferenceExample1:
- Julia version: 1.4.1
- Author: khudabukhsh.2
- Date: 2020-05-09
=#

using Pkg
Pkg.activate(".")
using DelayModel
using Revise
using ArgParse
using Random
using Distributions
using StatsBase
using LaTeXStrings
using Plots
using StatsPlots
using Colors
using Tables
using CSV
using DataFrames
include("src/MyColours.jl")
include("src/Model1.jl")

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--final-time"
            help = "Final time"
            arg_type = Float64
            default = 3.00
        "-n"
            help = "Initial copy number of A molecules"
            arg_type = Int64
            default = 5000
        "-b"
            help = "Birth rate of A molecules"
            arg_type = Float64
            default = 100.0
        "--conv-dist-name"
            help = "Name of the distribution characterized by tau"
            arg_type = String
            default = "InverseGamma"
        "--conv-dist-parms"
            help = "Conversion distribution parameters"
            arg_type = String
            default = "1.75,4.25"
        "--death-dist-name"
            help = "Name of the distribution characterized by d"
            arg_type = String
            default = "BetaPrime"
        "--death-dist-parms"
            help = "Death distribution parameters"
            arg_type = String
            default = "1.75,1.25"
        "--nSim"
            help = "Number of simulations"
            arg_type = Int64
            default = 100
        "--dt"
            help = "Time grid size to return simulated trajectory"
            arg_type = Float64
            default = 0.1
        "--backend"
            help = "Backend for plotting figures"
            arg_type = String
            default = "GR"
    end
    return parse_args(s)
end


function plot_hazards(r , t = 0.0:0.1:10.0)
    pl = plot(t,r[1].(t), color=cyans[3], linewidth=4,
              label="b", line=:dashdot, marker=:o, alpha=0.5, grid=false, legend=:best)
    plot!(t,r[2].(t), color=purplybrown[5], linewidth=5, label=L"\tau", line=:solid)
    plot!(t,r[3].(t), color=reds[3], linewidth=4, label="d", line=:dash, marker=:*)
    xlabel!("Time")
    ylabel!("Hazard functions")
    return pl
end

function main()
    parsed_args = parse_commandline()
    println("Parsed args:")
    for (arg,val) in parsed_args
        println("  $arg  =>  $val")
    end

    n = parsed_args["n"]
    nSim = parsed_args["nSim"]
    final_time = parsed_args["final-time"]
    dt = parsed_args["dt"]
    b = parsed_args["b"]
    conv_dist_name = parsed_args["conv-dist-name"]
    conv_dist_parms = parsed_args["conv-dist-parms"]
    death_dist_name = parsed_args["death-dist-name"]
    death_dist_parms = parsed_args["death-dist-parms"]
    nSpecies = 2
    nReaction = 3
    input_mtrx = [0 0; 1 0; 1 0]
    output_mtrx= [1 0; 0 1; 0 0]

    d = Array{Any}(nothing, nReaction)
    d[1] = Exponential(b)
    if conv_dist_name == "Weibull"
        dist_parms = DelayModel.str2parm(parsed_args["conv-dist-parms"])
        d[2] = Weibull(dist_parms[1],dist_parms[2])
    elseif conv_dist_name == "Gamma"
        dist_parms = DelayModel.str2parm(parsed_args["conv-dist-parms"])
        d[2] = Gamma(dist_parms[1],dist_parms[2])
    elseif conv_dist_name == "GeneralizedExtremeValue"
        dist_parms = DelayModel.str2parm(parsed_args["conv-dist-parms"])
        d[2] = GeneralizedExtremeValue(dist_parms[1], dist_parms[2], dist_parms[3])
    elseif conv_dist_name == "BetaPrime"
        dist_parms = DelayModel.str2parm(parsed_args["conv-dist-parms"])
        d[2] = BetaPrime(dist_parms[1], dist_parms[2])
        println("Assigning a Beta prime distribution with parameters ", dist_parms[1], " and ", dist_parms[2])
    elseif conv_dist_name == "InverseGamma"
        dist_parms = DelayModel.str2parm(parsed_args["conv-dist-parms"])
        d[2] = InverseGamma(dist_parms[1], dist_parms[2])
        println("Assigning an Inverse Gamma distribution with parameters ", dist_parms[1], " and ", dist_parms[2])
    elseif conv_dist_name == "Exponential"
        dist_parms = DelayModel.str2parm(parsed_args["conv-dist-parms"])
        d[2] = Exponential(dist_parms[1])
        println("Assigning an Exponential distribution with parameter ", dist_parms[1])
    end

    if death_dist_name == "Weibull"
        dist_parms = DelayModel.str2parm(parsed_args["death-dist-parms"])
        d[3] = Weibull(dist_parms[1],dist_parms[2])
    elseif death_dist_name == "Gamma"
        dist_parms = DelayModel.str2parm(parsed_args["death-dist-parms"])
        d[3] = Gamma(dist_parms[1],dist_parms[2])
    elseif death_dist_name == "GeneralizedExtremeValue"
        dist_parms = DelayModel.str2parm(parsed_args["death-dist-parms"])
        d[3] = GeneralizedExtremeValue(dist_parms[1], dist_parms[2], dist_parms[3])
    elseif death_dist_name == "BetaPrime"
        dist_parms = DelayModel.str2parm(parsed_args["death-dist-parms"])
        d[3] = BetaPrime(dist_parms[1], dist_parms[2])
        println("Assigning a Beta prime distribution with parameters ", dist_parms[1], " and ", dist_parms[2])
    elseif death_dist_name == "InverseGamma"
        dist_parms = DelayModel.str2parm(parsed_args["death-dist-parms"])
        d[3] = InverseGamma(dist_parms[1], dist_parms[2])
        println("Assigning an Inverse Gamma distribution with parameters ", dist_parms[1], " and ", dist_parms[2])
    elseif death_dist_name == "Exponential"
        dist_parms = DelayModel.str2parm(parsed_args["death-dist-parms"])
        d[3] = Exponential(dist_parms[1])
        println("Assigning an Exponential distribution with parameters ", dist_parms[1])
    end



    r, surv = DelayModel.hazard_survival(d)

    println("The mean holding times are ", mean(d[1]), ", ", mean(d[2]), ", and ", mean(d[3]))
    model1 = Model1.System(nSpecies, nReaction, input_mtrx, output_mtrx, r, surv)

    @time t, sims = Model1.simulate_model1(model1, [n, 0], nSim = nSim, t0=0.0, dt=dt, maxT=final_time);
    m, s = Model1.compute_moments(sims);
    b_std = s[:,2];


    @time t, hsims = Model1.hybrid_simulation(model1, n; nSim=nSim, t0=0.0, dt=dt, maxT=final_time)
    mh, sh = Model1.compute_moments_hybrid(hsims)

    pl1 = plot_hazards(r, t)
    savefig(pl1, "plots/hybrid_hazards.pdf")

    pl2 = plot(t, m[:,2], ribbon=b_std, alpha=0.3, color=maroons[2], marker=:o, linewidth=2, label="", grid=false )
    plot!(t, mh, ribbon=sh, alpha=0.3, color=cyans[3], linestyle=:dash, linewidth=2, label="", grid=false )
    plot!(t, m[:,2], color=maroons[2], marker=:o, linewidth=2, label="Full stochastic simulation", grid=false )
    plot!(t, mh,color=cyans[5], linestyle=:dash, linewidth=3, label="Hybrid simulation", grid=false )
    xlabel!("Time")
    ylabel!("Copy numbers of B")
    fname = "hybrid_sim" * conv_dist_name * "_" * DelayModel.distparm2str(conv_dist_parms) * "_" * death_dist_name * "_" * DelayModel.distparm2str(death_dist_parms) * "_N" * string(n)
    savefig(pl2, "plots/" * fname * ".pdf")
    savefig(pl2, "plots/" * fname * ".svg")
    if parsed_args["backend"] == "PGFPlotsX"
        savefig(pl2, "plots/" * fname * ".tikz")
    end

    fname = "combined_hybrid_" * conv_dist_name * "_" * DelayModel.distparm2str(conv_dist_parms) * "_" * death_dist_name * "_" * DelayModel.distparm2str(death_dist_parms) * "_N" * string(n)
    l = @layout [a{0.5w} b{0.5w}]
    pl3 = plot(pl1, pl2, layout = l, size=(600, 250))
    # plot!(size=(12,7.5))
    savefig(pl3, "plots/" * fname * ".pdf")
    savefig(pl3, "plots/" * fname * ".svg")
    if parsed_args["backend"] == "PGFPlotsX"
        savefig(pl3, "plots/" * fname * ".tikz")
    end

    println("Figures plotted\n")

end


main()
