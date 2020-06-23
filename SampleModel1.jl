#=
SampleModel1:
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
using Colors
using JuMP
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
            default = 500
        "--fast-simulation"
            help = "An option to run the hybrid simulation algorithm"
            action = :store_true
        "--mpi"
            help = "Option to indicate if MPI should be used for parallelization"
            action = :store_false
        "-b"
            help = "Birth rate of A molecules"
            arg_type = Float64
            default = 0.1
        "--conv-dist-name"
            help = "Name of the distribution characterized by tau"
            arg_type = String
            default = "Weibull"
        "--conv-dist-parms"
            help = "Conversion distribution parameters"
            arg_type = String
            default = "1.25,1.75"
        "--death-dist-name"
            help = "Name of the distribution characterized by d"
            arg_type = String
            default = "Weibull"
        "--death-dist-parms"
            help = "Death distribution parameters"
            arg_type = String
            default = "1.5,2.5"
        "--nSim"
            help = "Number of simulations"
            arg_type = Int64
            default = 100
        "--dt"
            help = "Time grid size to return simulated trajectory"
            arg_type = Float64
            default = 0.1
        "-p"
            help = "Boolean variable indicating whether to plot the simulated trajectories"
            action = :store_true
        "--backend"
            help = "Backend for plotting figures"
            arg_type = String
            default = "GR"
    end
    return parse_args(s)
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
    ifPlot = parsed_args["p"]
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
    end

    if death_dist_name == "Weibull"
        dist_parms = DelayModel.str2parm(parsed_args["death-dist-parms"])
        d[3] = Weibull(dist_parms[1],dist_parms[2])
    elseif death_dist_name == "Gamma"
        dist_parms = DelayModel.str2parm(parsed_args["death-dist-parms"])
        d[3] = Gamma(dist_parms[1],dist_parms[2])
    end

    r = Array{Any}(nothing,nReaction)
    r[1] = t -> DelayModel.hazard_function(d[1],t)
    r[2] = t -> DelayModel.hazard_function(d[2], t)
    r[3] = t -> DelayModel.hazard_function(d[3],t)

    surv = Array{Any}(nothing, nReaction)
    surv[1] = t -> DelayModel.survival_function(d[1],t)
    surv[2] = t -> DelayModel.survival_function(d[2],t)
    surv[3] = t -> DelayModel.survival_function(d[3],t)

    println("The mean holding times are ", mean(d[1]), ", ", mean(d[2]), ", and ", mean(d[3]))
    model1 = Model1.System(nSpecies, nReaction, input_mtrx, output_mtrx, r, surv)

    t, sims = Model1.simulate_model1(model1, [n, 0], nSim = nSim, t0 = 0.0, dt = dt, maxT=final_time);

    df = DataFrame(sims[:,2,:])
    df.times = collect(t)
    fname = "b_count_data_" * conv_dist_name * "_" * DelayModel.distparm2str(conv_dist_parms) * "_" * death_dist_name * "_" * DelayModel.distparm2str(death_dist_parms) * "_N" * string(n) * ".csv"
    CSV.write("data/" * fname, df)

    println("Simulations done")

    if ifPlot
        if parsed_args["backend"] == "PlotlyJS"
            plotlyjs()
            println("Using Plotly JS backend\n")
        elseif parsed_args["backend"] == "PGFPlotsX"
            pgfplotsx()
            println("Using PGFPlotsX backend\n")
        else
            gr()
            println("Using GR backend\n")
        end

        m, s = Model1.compute_moments(sims);
        a_std = s[:,1] ./ n;
        b_std = s[:,2] ./ n;

        pl = plot(t, m[:,1] ./ n, ribbon=a_std,
                          color=cyans[3], linewidth=2, label="", grid=false)
        plot!(t, m[:,1] ./ n, color=cyans[3], marker=:d, linewidth=1, label="A simulation", grid=false )
        plot!(t, m[:,2] ./ n, ribbon=a_std,
                          color=maroons[3], linewidth=2, label="", grid=false)
        plot!(t, m[:,2] ./ n, color=maroons[3], marker=:o, linewidth=2, label="B simulation", grid=false )
        xlabel!("Time")
        ylabel!("Concentration")
        fname = "sim_" * conv_dist_name * "_" * DelayModel.distparm2str(conv_dist_parms) * "_" * death_dist_name * "_" * DelayModel.distparm2str(death_dist_parms) * "_N" * string(n)
        savefig(pl, "plots/" * fname * ".pdf")
        savefig(pl, "plots/" * fname * ".svg")
        if parsed_args["backend"] == "PGFPlotsX"
            savefig(pl, "plots/" * fname * ".tikz")
        end
        println("Figures plotted\n")
    end

end

main()
