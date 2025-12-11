# -*- coding: utf-8 -*-
StartComputeTime = time()

import JLD2 as jld2
import FileIO as FIO
using LinearAlgebra
using Distributions
using Random

# if running interactively and wanting to plot inside the notebook
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



####################################### define two-component boson Hilbert space and its operators ######################
Nboson = 3 # maximum number of bosons
# mask that is true if a states has less than N bosons and false if it has more.
# It is used to remove all states beyond the chosen cutoff
Nmask = let 
    states = [[Na Nb] for Na in 0:Nboson for Nb in 0:Nboson]
    Nt = sum.(states)
    Nt.<=Nboson
end

# create new Hilbert space and the needed operators
uT.ITensors.space(::uT.ITensorMPS.SiteType"2B") = sum(Nmask)
let 
    a = diagm(1=>sqrt.(1:Nboson))
    adag = diagm(-1=>sqrt.(1:Nboson))
    as = kron(a,one(a))[Nmask,Nmask]
    adags = kron(adag,one(a))[Nmask,Nmask]
    id = one(as)
    bs = kron(one(a),a)[Nmask,Nmask]
    bdags = kron(one(a),adag)[Nmask,Nmask]
    uT.ITensors.op(::uT.ITensorMPS.OpName"a",::uT.ITensorMPS.SiteType"2B") = as
    uT.ITensors.op(::uT.ITensorMPS.OpName"adag",::uT.ITensorMPS.SiteType"2B") = adags
    uT.ITensors.op(::uT.ITensorMPS.OpName"Na",::uT.ITensorMPS.SiteType"2B") = adags*as
    uT.ITensors.op(::uT.ITensorMPS.OpName"b",::uT.ITensorMPS.SiteType"2B") = bs
    uT.ITensors.op(::uT.ITensorMPS.OpName"bdag",::uT.ITensorMPS.SiteType"2B") = bdags
    uT.ITensors.op(::uT.ITensorMPS.OpName"Nb",::uT.ITensorMPS.SiteType"2B") = bdags*bs
    uT.ITensors.op(::uT.ITensorMPS.OpName"Id",::uT.ITensorMPS.SiteType"2B") = id
    uT.ITensors.op(::uT.ITensorMPS.OpName"σz",::uT.ITensorMPS.SiteType"2B") = adags*as-bdags*bs
    uT.ITensors.op(::uT.ITensorMPS.OpName"σx",::uT.ITensorMPS.SiteType"2B") = adags*bs+bdags*as
    uT.ITensors.op(::uT.ITensorMPS.OpName"σy",::uT.ITensorMPS.SiteType"2B") = -1im*(adags*bs-bdags*as)
end
uT.wrt("Number of states in local Hilber space: $(sum(Nmask))")


#########################################  Set up parameters  ###################################################
# define Hamiltonian (notice factors of 1/2 in all single-site terms that avoids double counting in TEBD evolution)
# From the perspective of the computation the energy is refered to as <H>, which is not quantitatively identical to the physical energy (due ot the factors of 1/2)
Ham_str = """
function Ham(U,Uab,Ω,J,δ,μ,sites)
    hopping = -J*(uT.op("a",sites[1])*uT.op("adag",sites[2])+uT.op("adag",sites[1])*uT.op("a",sites[2])
            +uT.op("b",sites[1])*uT.op("bdag",sites[2])+uT.op("bdag",sites[1])*uT.op("b",sites[2]))
    rabi = 2*Ω*((uT.replaceprime(uT.op("a",sites[1])*uT.op("bdag",sites[1])',2,1)+uT.replaceprime(uT.op("adag",sites[1])*uT.op("b",sites[1])',2,1))*uT.op("Id",sites[2])
        +uT.op("Id",sites[1])*(uT.replaceprime(uT.op("adag",sites[2])*uT.op("b",sites[2])',2,1)+uT.replaceprime(uT.op("a",sites[2])*uT.op("bdag",sites[2])',2,1)))
    intra_species_interaction = U*0.5*((uT.replaceprime(uT.op("Na",sites[1])*(uT.op("Na",sites[1])'-uT.op("Id",sites[1])'),2,1)+uT.replaceprime(uT.op("Nb",sites[1])*(uT.op("Nb",sites[1])'-uT.op("Id",sites[1])'),2,1))*uT.op("Id",sites[2])
                            +(uT.replaceprime(uT.op("Na",sites[2])*(uT.op("Na",sites[2])'-uT.op("Id",sites[2])'),2,1)+uT.replaceprime(uT.op("Nb",sites[2])*(uT.op("Nb",sites[2])'-uT.op("Id",sites[2])'),2,1))*uT.op("Id",sites[1]))
    inter_species_interactions = Uab *(uT.replaceprime(uT.op("Na",sites[1])*uT.op("Nb",sites[1])',2,1)*uT.op("Id",sites[2]) 
                                +uT.replaceprime(uT.op("Na",sites[2])*uT.op("Nb",sites[2])',2,1)*uT.op("Id",sites[1]))
    b_chem = -(μ+2*δ) * (uT.op("Nb",sites[1])*uT.op("Id",sites[2])+uT.op("Id",sites[1])*uT.op("Nb",sites[2]))
    a_chem = -(μ-2*δ) * (uT.op("Na",sites[1])*uT.op("Id",sites[2])+uT.op("Id",sites[1])*uT.op("Na",sites[2]))
    return hopping + 0.5*rabi + 0.5*intra_species_interaction + 0.5*inter_species_interactions + 0.5*b_chem + 0.5 *a_chem
end
"""
eval(Meta.parse(Ham_str))

# define parameters for time evolution
parameters = let
    U = 1. # intraspecies repulsion
    Uab = 20. # interspecies repulsion
    μ = .5 # chemical potential
    J = 0.001 # hopping
    Ω = 0.8 # coherent coupling 
    δ = 0.2 # population imbalance
    τ = J^2*(2/U-1/Uab)
    T = 400. # total simulation time
    step = 0.1 # step size 
    Nsteps = Int(round(T/step)) 
    
    # Define observables
    # flag = true means observables are being calculated at every n * Δ'th step aka. Δ=1 means at every step. 
    # H saves the energy
    # λ saves the singular values as a vector
    # ρ2/ρ3 is wether or not to compute the two and three-body density matrices
    # TwoSite is a list of the two-site observables that should be computed. They must all be connected to operators defined for the siteType
    # 2p-corr is a list of all the two-point spatial correlation functions. Each element of the list should be of the form [o1,o2,d] which computes the correlation <o1_i o2_i_j> up to j=d. Again the operators should defined for the siteType  
    Obs = Dict{String,Any}("flag"=>true,"Δ"=>100,"λ"=>true,"H"=>true,"ρ1"=>false,"ρ2"=>false,"ρ3"=>false,"OneSite"=>["Na","Nb","σz"],"2p-corr"=>[["σz","σz",50]])
    
    # observables to be monitored during the evolution. 
    # flag: turns monitoring on/off
    # pm: monitoring done through the ProgressMonitor (true) or written to uT.io . On HPC do NOT use true 
    # MonitorObs is the observables that are monitored. Only observables present in Obs can be monitored
    MonitorObs = filter!(!isempty,vcat([Obs["H"] ? "H" : [], Obs["λ"] ? "λ" : []],(length(Obs["OneSite"])>0 ? Obs["OneSite"] : [])))  
    monitor = Dict{String,Any}("flag"=>true, "pm"=>false && !Sys.islinux(),"Obs"=>MonitorObs)
    convergence = Dict{String,Any}("flag"=>false)
    
    Dict{String,Any}("Total_time"=>T, "U"=>U,"Uab"=>Uab, "J"=>J,"τ"=>τ, "Nboson"=>Nboson,"μ"=>μ,
        "Ω"=>Ω,"δ"=>δ,"δ0"=>0.0, "maxBD"=>128, "cutoff"=>1E-8,
        "step"=>step,"Nsteps"=>Nsteps, "Obs"=>Obs,"monitor"=>monitor,"convergence"=>convergence)
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
# product state input for initial state calculation. The two-site state is has the structure 
# -β-ΓA-α-ΓB-β-
#    |     |
#
# The bond indices are α and β and the two physical tensors are ΓA and ΓB. within the uniform MPS ansatzs this pattern is then repeated infinitely many times
Init_state = let 
    ρa = 0.15
    ρb = 0.1
    N = parameters["Nboson"]
    
    sites = uT.siteinds("2B",2)
    α = uT.Index(1,"A_bond")
    β = uT.Index(1,"B_bond")
    
    ΓA = uT.ITensor(β,sites[1],α)
    ΓB = uT.ITensor(α',sites[2],β')
    #coefficient
    norma = 1/sqrt(sum(ρa.^(0:N)./factorial.(0:N)))
    coha = norma.*ρa.^(0.5 .*(0:N))./sqrt.(factorial.(0:N))
    normb = 1/sqrt(sum(ρb.^(0:N)./factorial.(0:N)))
    cohb = normb.*ρb.^(0.5 .*(0:N))./sqrt.(factorial.(0:N))

    states = [[Na Nb] for Na in 0:N for Nb in 0:N][Nmask]
    # renormalize with cut basis
    coef = [coha[states[n][1]+1]*cohb[states[n][2]+1] for n in 1:length(states)]
    coef .*= 1/sqrt(sum(abs.(coef).^2))
    # product coherent state
    for n in 1:uT.dim(sites[1])
        ΓA[sites[1]=>n,α=>1,β=>1] = coef[n]
        ΓB[sites[2]=>n,α'=>1,β'=>1] = coef[n]
    end
    λA = uT.ITensor(α,α')
    λA[α=>1,α'=>1] = 1.0 
    

    λB = uT.ITensor(β,β')
    λB[β=>1,β'=>1] = 1.0 
    
    Dict{String,Any}("ΓA"=>ΓA, "λA"=>λA, "ΓB"=>ΓB,"λB"=>λB,"sites"=>sites)
end;
H0 = Ham(parameters["U"],parameters["Uab"],parameters["τ"]*parameters["Ω"],parameters["J"],parameters["τ"]*parameters["δ0"],parameters["μ"],Init_state["sites"])
uT.wrt("_______Product state used for state preparation_______")
uT.wrt("Normalization: $(round(uT.twoBodyObs(Init_state,"Id"),sigdigits=3))\nZ-mag: $(round(uT.twoBodyObs(Init_state,"σz"),sigdigits=3))\nX-mag: $(round(uT.twoBodyObs(Init_state,"σx"),sigdigits=3))\nY-mag: $(round(uT.twoBodyObs(Init_state,"σy"),sigdigits=3))\nEnergy: $(round(uT.twoBodyObs(Init_state,H0),sigdigits=3))")

################## compute correlated initial state #####################################
gs_param = copy(parameters)
gs_param["step"] = parameters["step"] * Int(80/parameters["step"])
gs_param["Nsteps"] = 20000
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
uT.wrt("Normalization: $(round(uT.twoBodyObs(Init["state"],"Id"),sigdigits=3))\nZ-mag: $(round(uT.twoBodyObs(Init["state"],"σz"),sigdigits=3))\nX-mag: $(round(uT.twoBodyObs(Init["state"],"σx"),sigdigits=3))\nY-mag: $(round(uT.twoBodyObs(Init["state"],"σy"),sigdigits=3))\nEnergy: $(round(uT.twoBodyObs(Init["state"],H0),sigdigits=3))")

###################### setup for real-time evolution ###########################################
setup = let 
        state = setupInit["state"]
        # Hamiltonian. If it is time-dependent define the function H(t)
        if false
            function Hf(t)
                τ = 1
                if t<=τ
                    return Ham(parameters["U"],parameters["Uab"],parameters["τ"]*parameters["Ω"],parameters["J"],t/τ*parameters["τ"]*parameters["δ"],parameters["μ"],state["sites"])
                else
                    return Ham(parameters["U"],parameters["Uab"],parameters["τ"]*parameters["Ω"],parameters["J"],parameters["τ"]*parameters["δ"],parameters["μ"],state["sites"])
                end
            end
            Ht = (time_dep = true, H = Hf)
        else
            H = Ham(parameters["U"],parameters["Uab"],parameters["τ"]*parameters["Ω"],parameters["J"],parameters["τ"]*parameters["δ"],parameters["μ"],state["sites"])
            Ht = (time_dep = false, H = H)
        end

        # name for files that are being saved
        descriptor = ("uTEBD_2cBosonExample__MB_$(parameters["maxBD"])__U_"*replace(
            string(round(parameters["U"],sigdigits=3)),"."=>"_")*"__Uab_"*replace(string(round(parameters["Uab"],sigdigits=3)),"."=>"_")*
        "__J_"*replace(string(round(parameters["J"],sigdigits=3)),"."=>"_")*"__mu_"*replace(string(round(parameters["μ"],sigdigits=3)),"."=>"_")
        *"__Omega_"*replace(string(round(parameters["Ω"],sigdigits=3)),"."=>"_")*"__delta_"*replace(string(round(parameters["δ"],sigdigits=3)),"."=>"_")
        *"__tau_"*replace(string(round(parameters["τ"],sigdigits=3)),"."=>"_")*"__dt_"*replace(string(round(parameters["step"],sigdigits=3)),"."=>"_"))
        
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


# ######################### Real-time evolution #####################################
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
