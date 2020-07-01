#=
Example1:
- Julia version: 1.4.1
- Author: khudabukhsh.2
- Date: 2020-05-07
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
include("src/MyColours.jl")
include("src/Model1.jl")


function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--final-time", "-t"
            help = "Final time"
            default = 20.0
        "--n-molecules", "-n"
            help = "Initial copy number of A molecules"
            arg_type = Int
            default = 500
        "--fast-simulation"
            help = "An option to run the hybrid simulation algorithm"
            action = :store_true
        "--mpi"
            help = "Option to indicate if MPI should be used for parallelization"
            action = :store_false
        "--backend"
            help = "Backend for plotting figures"
            arg_type = String
            default = "GR"
    end

    return parse_args(s)
end

function plot_hazards(r , t = 0.0:0.1:10.0)
    pl = plot(t,r[1].(t), color=cyans[3], linewidth=4,
              label="b", line=:dashdot,  alpha=0.5, grid=false, legend=:bottomright)
    plot!(t,r[2].(t), color=purplybrown[5], linewidth=5, label=L"\tau", line=:solid)
    plot!(t,r[3].(t), color=reds[3], linewidth=4, label="d", line=:dash)
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

    n = 500
    nSpecies = 2
    nReaction = 3
    input_mtrx = [0 0; 1 0; 1 0]
    output_mtrx= [1 0; 0 1; 0 0]

    d = Array{Any}(nothing, nReaction)
    d[1] = Exponential(2.50)
    d[2] = GeneralizedExtremeValue(1.25/0.30,1.250,0.30)
    d[3] = Gamma(2.5, 1.75)

    r = Array{Any}(nothing,nReaction)
    r[1] = t -> DelayModel.hazard_function(d[1],t)
    r[2] = t -> DelayModel.hazard_function(d[2], t)
    r[3] = t -> DelayModel.hazard_function(d[3],t)

    pl = plot_hazards(r)
    savefig(pl, "plots/hazards.pdf")


    surv = Array{Any}(nothing, nReaction)
    surv[1] = t -> DelayModel.survival_function(d[1],t)
    surv[2] = t -> DelayModel.survival_function(d[2],t)
    surv[3] = t -> DelayModel.survival_function(d[3],t)

    println("The mean holding times are ", mean(d[1]), ", ", mean(d[2]), ", and ", mean(d[3]))
    model1 = Model1.System(nSpecies, nReaction, input_mtrx, output_mtrx, r, surv)

    t, sims = Model1.simulate_model1(model1, [n, 0], nSim = 100);
    println("Simulations done\n")
    m, s = Model1.compute_moments(sims);
    t, sol_a, yb = Model1.solve_PDEs(model1, n);
    a_std = s[:,1] ./ n;
    b_std = s[:,2] ./ n;

    limit_plot = plot(t, m[:,1] ./ n, ribbon=a_std,
                      color=cyans[3], linewidth=2, label="", grid=false)
    plot!(t, sol_a, color=cyans[5], line=:solid, linewidth=2, label="A theoretical", grid=false )
    plot!(t, m[:,1] ./ n, color=cyans[3], markersize=2.0, marker=:d, linewidth=1, label="A simulation", grid=false )
    plot!(t, m[:,2] ./ n, ribbon=a_std,
                      color=maroons[3], linewidth=2, label="", grid=false)
    plot!(t, yb, color=maroons[5], line=:dash, linewidth=2, label="B theoretical", grid=false )
    plot!(t, m[:,2] ./ n, color=maroons[3], marker=:o, markersize=2.0, linewidth=2, label="B simulation", grid=false )
    xlabel!("Time")
    ylabel!("Concentration")
    savefig(limit_plot, "plots/model1_lln.pdf")
    savefig(limit_plot, "plots/model1_lln.svg")
    # savefig(limit_plot, "plots/model1_lln.eps")
    if parsed_args["backend"] == "PGFPlotsX"
        savefig(limit_plot, "plots/model1_lln.tikz")
    end


    fname = "combined_model1_lln"
    l = @layout [a{0.5w} b{0.5w}]
    pl3 = plot(pl, limit_plot, layout = l, size=(800, 350))
    # plot!(size=(12,7.5))
    savefig(pl3, "plots/" * fname * ".pdf")
    savefig(pl3, "plots/" * fname * ".svg")
    if parsed_args["backend"] == "PGFPlotsX"
        savefig(pl3, "plots/" * fname * ".tikz")
    end


    println("Figures plotted\n")

end

main()
