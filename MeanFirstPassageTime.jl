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
            default = 1.0
        "-n"
            help = "Initial copy number of A molecules"
            arg_type = Int64
            default = 2000
        "-b"
            help = "Birth rate of A molecules"
            arg_type = Float64
            default = 10.0
        "--conv-dist-name"
            help = "Name of the distribution characterized by tau"
            arg_type = String
            default = "GeneralizedExtremeValue"
        "--conv-dist-parms"
            help = "Conversion distribution parameters"
            arg_type = String
            default = "4.166,1.25,0.3"
        "--death-dist-name"
            help = "Name of the distribution characterized by d"
            arg_type = String
            default = "Weibull"
        "--death-dist-parms"
            help = "Death distribution parameters"
            arg_type = String
            default = "0.5, 1.5"
        "--nSim"
            help = "Number of simulations"
            arg_type = Int64
            default = 1000
        "--dt"
            help = "Time grid size to return simulated trajectory"
            arg_type = Float64
            default = 0.1
        "--backend"
            help = "Backend for plotting figures"
            arg_type = String
            default = "GR"
        "--end-time"
            help = "End time for integrals"
            arg_type = Float64
            default = 100.0
    end
    return parse_args(s)
end


function plot_hazards(r , t = 0.0:0.1:10.0)
    pl = plot(t,r[1].(t), color=cyans[3], linewidth=4,
              label="b", line=:dashdot, alpha=0.5, grid=false, legend=:topleft)
    plot!(t,r[2].(t), color=purplybrown[5], linewidth=5, label=L"\tau", line=:solid)
    plot!(t,r[3].(t), color=reds[3], linewidth=4, label="d", line=:dash)
    xlabel!("Time")
    ylabel!("Hazard functions")
    return pl
end


function theoretical_mfpt(dist, n; endT=50.0)
    r, surv = DelayModel.hazard_survival(dist)
    # f(s) = s * n * r[2](s) * exp(-s) / (r[2](s) + r[3](s))
    # ds = 0.01
    # sgrids = 0.01:ds:endT
    # res = sum(f.(sgrids)) * ds
    # res, = quadgk(s -> n * r[2](s) * exp(-s) / (r[2](s) + r[3](s)), 0.0, 12.0)
    res, = quadgk(s -> exp(-s)*r[2](s), 0.0, endT)
    return 1.0/(n*res)
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
    endT = parsed_args["end-time"]
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

    n0 = [n, 0]

    fpts = zeros(nSim)
    Threads.@threads for i in 1:nSim
        x0 = Model1.initialize(model1, n0)
        fpts[i] = Model1.first_passage_time(model1, x0)
    end
    println("Simulations done")

    println("Mean of the simulated FPTs is ", mean(fpts))

    df = DataFrame(Dict("FPT" => fpts))
    fname = "mfpt_" * conv_dist_name * "_" * DelayModel.distparm2str(conv_dist_parms) * "_" * death_dist_name * "_" * DelayModel.distparm2str(death_dist_parms) * "_N" * string(n) * ".csv"
    CSV.write("out/" * fname, df)

    theo_t1 = Model1.theoretical_mfpt(model1, n, endT=endT)
    # theo_t1 = theoretical_mfpt(d, n, endT=50.0)
    println("Theoretical MFPT is ", theo_t1)


    # fpts_theo = zeros(nSim)
    # Threads.@threads for i in 1:nSim
    #     fpts_theo[i] = rand(Exponential(theo_t1))
    # end
    fpts_theo = rand(Exponential(theo_t1), nSim)
    # fpts_theo = rand(Exponential(mean(fpts)), nSim)

    println("The difference between theoretical MFPT and mean of simulated FPTs is ", theo_t1 - mean(fpts))

    pl1 = plot_hazards(r)
    savefig(pl1, "plots/mfpt_hazards.pdf")

    # palette = distinguishable_colors(5)
    #
    # pl2 = density(fpts, density=true, fillrange = 0, fillalpha = 0.25, fillcolor = palette[1], color=palette[2], linewidth=2.5, grid=false, label="Simulated FPTs", legend=:best)
    # density!(fpts_theo, density=true, line=:dash, fillrange = 0, fillalpha = 0.25, fillcolor = palette[3], color=palette[4], linewidth=2.5, grid=false, label="Simulated approximate FPTs")
    # vline!([theo_t1], color=palette[5], linewidth=3.5,  label="Theoretical MFPT")
    # xlabel!("First passage time")
    # ylabel!("Density")


    # pl2 = density(fpts, density=true, fillrange = 0, fillalpha = 0.25, fillcolor = browns[1], color=browns[1], linewidth=4.0, grid=false, label="FPTs", legend=:best)
    # density!(fpts_theo, density=true, line=:dash, fillrange = 0, fillalpha = 0.25, fillcolor = cyans[1], color=cyans[3], linewidth=2.5, grid=false, label="Approximate FPTs")
    # # vline!([theo_t1], color=reds[5], linewidth=3.5,  label="Theoretical MFPT")
    # xlabel!("First passage time")
    # ylabel!("Density")
    # xlims!(0,0.04)

    pl2 = histogram(fpts, density=true, color=browns[1], alpha=0.25, linewidth=0.25, grid=false, label="FPTs", legend=:best)
    histogram!(fpts_theo, density=true, color=cyans[3], alpha=0.25, linewidth=0.25, grid=false, label="Approximate FPTs")
    # vline!([theo_t1], color=reds[5], linewidth=3.5,  label="Theoretical MFPT")
    xlabel!("First passage time")
    ylabel!("Density")
    xlims!(0,0.04)

    fname = "mfpt_" * conv_dist_name * "_" * DelayModel.distparm2str(conv_dist_parms) * "_" * death_dist_name * "_" * DelayModel.distparm2str(death_dist_parms) * "_N" * string(n)
    savefig(pl2, "plots/" * fname * ".pdf")
    savefig(pl2, "plots/" * fname * ".svg")
    if parsed_args["backend"] == "PGFPlotsX"
        savefig(pl2, "plots/" * fname * ".tikz")
    end

    if parsed_args["backend"] == "GR"
        display(pl2)
    end

    fname = "combined_mfpt_" * conv_dist_name * "_" * DelayModel.distparm2str(conv_dist_parms) * "_" * death_dist_name * "_" * DelayModel.distparm2str(death_dist_parms) * "_N" * string(n)
    l = @layout [a{0.5w} b{0.5w}]
    pl3 = plot(pl1, pl2, layout = l, size=(800, 350))
    # plot!(size=(12,7.5))
    savefig(pl3, "plots/" * fname * ".pdf")
    savefig(pl3, "plots/" * fname * ".svg")
    if parsed_args["backend"] == "PGFPlotsX"
        savefig(pl3, "plots/" * fname * ".tikz")
    end

    println("Figures plotted\n")


end


main()
