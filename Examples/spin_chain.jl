StartComputeTime = time()

import JLD2 as jld2
import FileIO as FIO
using LinearAlgebra
using Distributions
using Random

# if running interactively
if isinteractive()
    using MathTeXEngine # required for texfont in Makie
    using CairoMakie
    using LaTeXStrings
    import MakieExtra as ME
end

import uTEBD as uT

# get walltime and redirect output to file if wanted
if ARGS!=String[] && length(ARGS)>1
    walltime = parse(Float64,ARGS[1])
    if ARGS[2] == "-"
        uT.set_io!(stdout)
    else
        uT.set_io!(open(ARGS[2],"a"))
    end
end

uT.wrt("Time spent on loading modules: $(round(time()-StartComputeTime,digits=3)) seconds ")


#########################################  Set up parameters  ###################################################
# define Hamiltonian (notice factors of 1/2 in all single-site terms that avoids double counting in TEBD evolution)
# From the perspective of the computation the energy is refered to as <H>, which is not quantitatively identical to the physical energy (due ot the factors of 1/2)
Ham_str = """
function Ham(τ,ϵ,Ω,δ,sites)
    hopping = -τ*(ϵ*uT.op("X",sites[1])*uT.op("X",sites[2])+ϵ*uT.op("Y",sites[1])*uT.op("Y",sites[2])+uT.op("Z",sites[1])*uT.op("Z",sites[2]))
    transverse = τ*Ω*(uT.op("X",sites[1])*uT.op("Id",sites[2])+uT.op("X",sites[2])*uT.op("Id",sites[1]))
    longitudinal = τ*δ*(uT.op("Z",sites[1])*uT.op("Id",sites[2])+uT.op("Z",sites[2])*uT.op("Id",sites[1]))  
    return hopping + 0.5*transverse + 0.5*longitudinal
end
"""
eval(Meta.parse(Ham_str))

# define parameters for time evolution
parameters = let
    τ = 1 # Overall energy scale 
    T = 6 # total simulation time
    step = 0.005 # step size 
    Nsteps = Int(round(T/step)) 
    
    # Define observables
    # flag = true means observables are being calculated at every n * Δ'th step aka. Δ=1 means at every step. 
    # H saves the energy density
    # λ saves the singular values as a vector
    # ρ2/ρ3 is wether or not to compute the two and three-body density matrices
    # TwoSite is a list of the two-site observables that should be computed. They must all be connected to operators defined for the siteType
    # 2p-corr is a list of all the two-point spatial correlation functions. Each element of the list should be of the form [o1,o2,d] which computes the correlation <o1_i o2_i_j> up to j=d. Again the operators should defined for the siteType  
    Obs = Dict{String,Any}("flag"=>true,"Δ"=>10,"λ"=>true,"H"=>true,"ρ1"=>false,"ρ2"=>false,"ρ3"=>false,"OneSite"=>["Z","X","Y"],"2p-corr"=>[["Z","Z",50],["X","X",50]])
    
    # observables to be monitored during the evolution. 
    # flag: turns monitoring on/off
    # pm: monitoring done through the ProgressMonitor (true) or written to uT.io . On HPC do NOT use true 
    # MonitorObs is the observables that are monitored. Only observables present in Obs can be monitored
    MonitorObs = filter!(!isempty,vcat([Obs["H"] ? "H" : [], Obs["λ"] ? "λ" : []],(length(Obs["OneSite"])>0 ? Obs["OneSite"] : [])))  
    monitor = Dict{String,Any}("flag"=>true, "pm"=>false && !Sys.islinux(),"Obs"=>MonitorObs)
    
    # Convergence stops the evolution if steady state is reached
    # Standard use case is for automatically stopping an imaginary time evolution 
    convergence = Dict{String,Any}("flag"=>false)
    
    Dict{String,Any}("Total_time"=>T,"τ"=>τ, "Ω"=>0.8, "ϵ"=>0.025641025641025647, "δ"=>0.2,"δ0"=>0.0,"Hamiltonian"=>Ham_str,
        "maxBD"=>200, "cutoff"=>1E-16, "step"=>step,"Nsteps"=>Nsteps, "Obs"=>Obs,"monitor"=>monitor,"convergence"=>convergence)
end

# print parameters for computation
uT.wrt("Parameters: "*join(["$k = $(v)" for (k, v) in Iterators.filter(((k,v),)->!(k in ["Obs","monitor","Hamiltonian"]), parameters)], ", "))
uT.wrt("Observables: "*join([d[1]*" = $(d[2])" for d in pairs(parameters["Obs"])],", "))
uT.wrt("Monitoring during evolution: "*join([d[1]*" = $(d[2])" for d in pairs(parameters["monitor"])],", "))
uT.wrt("Convergene criteria: "*join([d[1]*" = $(d[2])" for d in pairs(parameters["convergence"])],", "))


########################### Initial state ##################################
# product state input for initial state calculation. The two site state is has the structure 
# -β-ΓA-α-ΓB-β-
#    |     |
#
# The bond indices are α and β and the two physical tensors are ΓA and ΓB. within the uniform MPS ansatzs this pattern is then repeated infinitely many times
Init_state = let 
    sites=uT.siteinds("Qubit",2)
    α = uT.Index(1,"A_bond")
    β = uT.Index(1,"B_bond")

    # start each site as the first eigenstate of the σZ operator + a small random fluctuation 
    EW, EV = eigen(uT.array(uT.op(sites[1],"Z")))
    c = EV[2,:].+ 0.1 .*(rand(Uniform(-1,1),2)+1im*rand(Uniform(-1,1),2))
    c = c./sqrt(sum(abs.(c).^2))
    coef = [[1,c[1]],[2,c[2]]]
    
    ΓA = uT.ITensor(β,sites[1],α)
    ΓB = uT.ITensor(α',sites[2],β')
    for i in coef
        ΓA[sites[1]=>Int(i[1]),α=>1,β=>1] = i[2]
        ΓB[sites[2]=>Int(i[1]),α'=>1,β'=>1] = i[2]
    end
    
    λA = uT.ITensor(α,α')
    λA[α=>1,α'=>1] = 1.0 
    
    
    λB = uT.ITensor(β,β')
    λB[β=>1,β'=>1] = 1.0 
    
    Dict{String,Any}("ΓA"=>ΓA, "λA"=>λA, "ΓB"=>ΓB,"λB"=>λB,"sites"=>sites)
end
H0 = Ham(parameters["τ"],parameters["ϵ"],parameters["Ω"],parameters["δ0"],Init_state["sites"])
uT.wrt("_______Product state used for state preparation_______")
uT.wrt("Normalization: $(round(uT.twoBodyObs(Init_state,"Id"),sigdigits=3))\nZ-mag: $(round(uT.twoBodyObs(Init_state,"Z"),sigdigits=3))\nX-mag: $(round(uT.twoBodyObs(Init_state,"X"),sigdigits=3))\nY-mag: $(round(uT.twoBodyObs(Init_state,"Y"),sigdigits=3))\nEnergy: $(round(uT.twoBodyObs(Init_state,H0),sigdigits=3))")

################## compute correlated initial state #####################################
gs_param = copy(parameters)
gs_param["step"] = parameters["step"]/parameters["τ"]
gs_param["Nsteps"] = 100000
gs_param["Obs"] = Dict{String,Any}("flag"=>true,"Δ"=>1000,"λ"=>true,"H"=>true)
gs_param["monitor"] = Dict{String,Any}("flag"=>false)
gs_param["convergence"] = Dict{String,Any}("flag"=>true,"state"=>true,"stateΔ"=>1E-10,"SV"=>true,"SVΔ"=>1E-10)
# to initialize the calculation build the dictionary with SysInit.
# The first argument is the initial state for the calculation
# The second argument is a NamedTuple with three elements: ([time_dep, Hevo,H), if the Hamiltonian is time-dependent then set time_dep = true, 
# otherwise set it to false. If time_dep = true then Hevo and H has to be functions that takes as only input a time value.
# The third argument is the parameters dictionary 
# Additional optional arguments can be found in the source code. 
setupInit=uT.SysInit(Init_state,(time_dep = false, H = H0),gs_param,imag=true);
t0 = time()

uT.wrt("Preparing correlated initial state")
Init = uT.TEBD(setupInit);
setupInit = uT.SysUpd(setupInit,["state"], [uT.CanonGauge(Init["state"])])
uT.wrt("Preparation of initial state took: $(round(time()-t0,sigdigits=2)) seconds")
uT.wrt("Normalization: $(round(uT.twoBodyObs(Init["state"],"Id"),sigdigits=3))\nZ-mag: $(round(uT.twoBodyObs(Init["state"],"Z"),sigdigits=3))\nX-mag: $(round(uT.twoBodyObs(Init["state"],"X"),sigdigits=3))\nY-mag: $(round(uT.twoBodyObs(Init["state"],"Y"),sigdigits=3))\nEnergy: $(round(uT.twoBodyObs(Init["state"],H0),sigdigits=3))")

###################### setup for real-time evolution ###########################################
setup = let 
        state = setupInit["state"]
        # Hamiltonian. If it is time-dependent define the function H(t)
        if false
            function Hfct(t)
                # linear ramp length
                τ = 1
                if t<=τ
                    return Ham(parameters["τ"],parameters["ϵ"],parameters["Ω"],t/τ*parameters["δ"],state["sites"])
                else
                    return Ham(parameters["τ"],parameters["ϵ"],parameters["Ω"],parameters["δ"],state["sites"])
                end
            end

            Ht = (time_dep = true, H = Hfct)
        else
            H = Ham(parameters["τ"],parameters["ϵ"],parameters["Ω"],parameters["δ"],state["sites"])
            Ht = (time_dep = false, H = H)
        end
        # name for files that are being saved
        descriptor = ("uTEBD_spinExample__MB_$(parameters["maxBD"])__tau_"*replace(
                string(round(parameters["τ"],sigdigits=3)),"."=>"_")*"__eps_"*replace(
                string(round(parameters["ϵ"],sigdigits=3)),"."=>"_")*"__Om_"*replace(
                string(round(parameters["Ω"],sigdigits=3)),"."=>"_")*"__del_"*replace(
                string(round(parameters["δ"],sigdigits=3)),"."=>"_")*"__del0_"*replace(string(round(parameters["δ0"],sigdigits=3)),"."=>"_")*"__dt_"*replace(string(round(parameters["step"],sigdigits=3)),"."=>"_"))
        
        # folder to save files in
        if isinteractive()
            dataFolder = pwd()*"/data/"
            if !isdir(dataFolder)
                mkpath(dataFolder)
            end
        else 
            dataFolder = pwd()*"/"
        end
        
        uT.SysInit(state,Ht,parameters,descriptor = descriptor, folder = dataFolder)
    end

################################### Compute thermal state with energy equal to quenched initial correlated state with quenched Hamiltonian. Done by cooling down until a thermal state with energy Einit is reached.
# as both a left and a right acting Hamiltonian is need a Hamiltonian function must be supplied that only takes the sites as an argument
if true
    tth= time()
    function Hfct(sites)
        return Ham(parameters["τ"],parameters["ϵ"],parameters["Ω"],parameters["δ"],sites)
    end
    state = setupInit["state"]
    Einit = real(uT.twoBodyObs(state,Hfct(state["sites"])))
    thermal = uT.iρβ(Hfct,Einit,string(uT.tags(state["sites"][1])[1]),"Z","Id")
    uT.wrt("Computed thermal state with energy equal to quenched state. Took: $(round(time()-tth,digits=2)) seconds")
else
    thermal = []
end


########################## Real-time evolution #####################################
# set time limit (in seconds) for real time evolution

uT.wrt("------------------------- Time evolution --------------------")
if @isdefined(walltime)
    tlim = walltime-(time()-StartComputeTime)
else 
tlim = 400
end
uT.wrt("Time limit for uTEBD calclation is: $(round(tlim-30,digits=2)) seconds")

# if running on cluster move to GPU
if Sys.islinux()
    setup = uT.SysUpd(setup, ["gpuFlag"],[true])
else
    setup = uT.SysUpd(setup, ["gpuFlag"],[false])
end

# Do time evolution
t0 = time()
if isinteractive()
    # do calculation asynchronously to allow make it possible to stop it interactive by setting uT.stop_evo = true
    quenched = @async uT.TEBD(setup,time_lim = tlim);
else
    quenched = uT.TEBD(setup,time_lim = tlim);
    uT.wrt("Computing time-evolution took: $(round(time()-t0,digits=2)) seconds")
end

# if computation was done asynchronously fetch the results
if isinteractive()
    uT.stop_evo = true
    quenched = fetch(quenched)
end


############# save calculation #####################
results=Dict{String,Any}("setup"=>setup,"data"=>quenched)
sN = uT.saveName((setup["folder"]*setup["descriptor"]*"__t0_"*replace(string(quenched["tlist"][begin]),"."=>"_")*"__tf_"
    *replace(string(round(quenched["tlist"][end],digits=5)),"."=>"_")*"__results"),"jld2")
# save in global folder
FIO.save(sN,results)

uT.wrt("Exported results to: "*sN)
