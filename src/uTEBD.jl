module uTEBD

# importing packages
using LinearAlgebra
using Arpack
using Measures
using ITensors, ITensorMPS
using ITensors:cpu, Diag
using Suppressor
import CUDA as cuda
if Sys.islinux()
elseif Sys.isapple()
    import Metal as mtl
end
import ProgressMeter as PM

# function to write to a specified output stream
const io_ref = Ref{Union{Nothing,IO}}(nothing)

function set_io!(io::IO)
    io_ref[] = io
end

function wrt(str::String)
    io = io_ref[]
    if io === nothing || !isopen(io)
        io = stdout          # safe fallback
        io_ref[] = io
    end
    println(io, str)
    flush(io)
end

# find the indices of the elements in lst that is closest in absolute value to the elements in x0
function find_closest_ind(lst,x0)
    return [findmin(x->abs(x-i), lst)[2] for i in x0]
end

if Sys.islinux()
    function GPU(a)
            if typeof(a)==Vector{ITensor}
                return cuda.cu.(a)
            else
                return cuda.cu(a)
            end
        end
    function GPUvec(n)
        return cuda.CuArray{ComplexF64}(undef, n)
    end
    function GPUcollect(v)
        return cuda.collect(v)
    end
    function GPUprint()
        out = @capture_out begin
            cuda.pool_status()
        end
        wrt(out)
    end
elseif Sys.isapple()
    function GPU(a)
            if typeof(a)==Vector{ITensor}
                return mtl.mtl.(a)
            else
                return mtl.mtl(a)
            end
        end
    function GPUvec(n)
        v = Vector{ComplexF64}(undef, n)
        return mtl.mtl(v)
    end
    function GPUcollect(v)
        return cpu(v)
    end
    function GPUprint()
    end
end

# check if a file already exists and if so then update the name of
# the generated file
function saveName(name::String,format::String)
    #check if file already exists
    n = 1
    txt = ""
    while isfile(name*txt*"."*format)
        txt="_"*string(n)
        n+=1
    end
    return name*txt*"."*format
end

# extract a numeric value from a string with the value having a 
# left delimiter and right delimiter
function extractNum(str::String,ldelim,rdelim)
    return parse(Float64,replace(split(split(str,rdelim)[1],ldelim)[2],"_"=>"."))
end

# convert between F64 and F32
function con32_64(A;F32::Bool=true)
    Adata = array(A)                             
    Adata = F32 ? ComplexF32.(Adata) : ComplexF64.(Adata)              
    return ITensor(Adata, inds(A)...)
end

#Check canonical form
function CanonGaugeCheck(state; left::Bool=true)
    ΓA = state["ΓA"]
    ΓB = state["ΓB"]
    λA = state["λA"]
    λB = state["λB"]
    sites = state["sites"]
    Γ = ΓA*λA*ΓB
    if left
        α = Index(dim(commonind(ΓB,λB)),tags="α")
        Γl = replaceind(Γ,commonind(ΓB,λB),α)
    else
        α = Index(dim(commonind(ΓA,λB)),tags="α")
        Γl = replaceind(Γ,commonind(ΓA,λB),α)
    end
    temp = λB * Γl
    temp = λB * temp
    temp = array(temp * conj.(Γ))
    return maximum(abs.(temp-one(temp)))
end

# Gauge to cannonical form
function CanonGauge(state;co=1e-16)
    # determine if transfering back to GPU
    if isa(array(state["ΓA"]),cuda.CuArray)
        gpu = true
        ΓA = con32_64(cpu(state["ΓA"]),F32=false)
        ΓB = con32_64(cpu(state["ΓB"]),F32=false)
        λA = con32_64(cpu(state["λA"]),F32=false)
        λB = con32_64(cpu(state["λB"]),F32=false)
    else
        gpu = false
        ΓA = state["ΓA"]
        ΓB = state["ΓB"]
        λA = state["λA"]
        λB = state["λB"]
    end
    
    maxD = maximum((dims(λA)...,dims(λB)...))
    if maxD == 1
        return state
    else
        sites = state["sites"]
        Γ = ΓA*λA*ΓB
        b = commonind(ΓA,λB) 
        a = commonind(ΓA,λA)
        α = Index(dim(b'),tags="α")
        β = Index(dim(b),tags="β")
        αC = combiner(α,α',tags="αC")
        βC = combiner(β,β',tags="βC")
        Nα = dim(α)
        Nβ = dim(β)
        #Right 
        R = replaceind(Γ * replaceind(λB,b,β),b,α)
        R = R * conj.(prime(R,[α,β]))
        # # vectorize grouping α,α' and β,β'
        Rarr = reshape(permutedims(array(R),(1,3,2,4)),(Nα^2,Nβ^2))
        
        λ, ϕ = Arpack.eigs(Rarr, nev=1, which=:LM)
        λ = λ[1]
        if maximum(abs.(Rarr*ϕ .- λ .* ϕ))>1e-10; wrt("Calc. of dominant right eigenvector failed. Error: $(maximum(abs.(Rarr*ϕ .- λ .* ϕ)))"); return state
        else
            # devectorize
            VR = reshape(ϕ*exp(-im*angle(ϕ[1])),(Nα,Nα))
            VR[abs.(VR) .< 1E-10] .= zero(eltype(VR))
            if !(VR≈VR'); wrt("right eigenmatrix not Hermitian"); return state
            else
                vals, vecs = eigen(VR)
                vals =real(vals)
                if any(0 .>= real(vals)); wrt("right eigendecompositions is not non-negative: minimum EV: $(minimum(real(vals)))"); return state
                else
                    if !(vecs*diagm(vals)*vecs'≈VR); wrt("right eigendecomposition failed test 1"); return state
                    else
                        X = vecs*sqrt.(diagm(vals));
                        if !(X*X'≈VR); wrt("right eigendecomposition failed test 2"); return state
                        else
                            #Left 
                            L = replaceind(Γ * replaceind(λB,b',α),b',β)
                            L = L * conj.(prime(L,[α,β]));
                            Larr = reshape(permutedims(array(L),(2,4,1,3)),(Nα^2,Nβ^2))
                            
                            λL, ϕL = Arpack.eigs(transpose(Larr), nev=1, which=:LM)
                            ϕL = transpose(ϕL)
                            λL=λL[1]
                            if maximum(abs.(ϕL*Larr .- λL .* ϕL))>1e-10; wrt("Calc. of dominant left eigenvector failed ");return state 
                            else
                                if (abs(λ-λL)> 1e-10); wrt("mismatch between right and left eigenvalues: |λR-λL|=$(abs(λ-λL))");return state
                                else
                                    # devectorize
                                    VL = reshape(ϕL*exp(-im*angle(ϕL[1])),(Nβ,Nβ))
                                    VL[abs.(VL) .< 1E-10] .= zero(eltype(VL))
                                    if !(VL≈VL'); wrt("left eigenmatrix not Hermitian");return state
                                    else
                                        valsL, vecsL = eigen(VL)
                                        if !(vecsL*diagm(valsL)*vecsL'≈VL); wrt("left eigendecomposition failed test 1");return state
                                        else
                                            if any(0 .>= real(valsL)); wrt("left eigendecomposition is not non-negative");return state
                                            else
                                                Y = (sqrt.(diagm(valsL))*vecsL')'
                                                if !(Y*Y'≈VL); wrt("left eigendecomposition failed test 2"); return state
                                                else
                                                    Y = transpose(Y)
                                                    U, S, VT = svd(Y*array(λB)*X)
                                                    VT = VT';
                                                    if !(U * diagm(S) * VT ≈ Y*array(λB)*X); wrt("SVD failed");return state
                                                    else
                                                        S = S./sqrt(sum(S.^2))
                                                        λB = ITensor(Diag(S),(b,b'))
                                                        Γ = ITensor(VT*inv(X),b,α)*replaceind(Γ,b,α)
                                                        Γ = replaceind(Γ,b',α)*ITensor(inv(Y)*U,α,b')
                                                        ΓA, λA, ΓB = svd(replaceind(λB,b',α)*Γ*
                                                             replaceind(λB,b,β),(α,uniqueind(ΓA,noprime(ΓB,(b',a'))))
                                                             ,cutoff=co, maxdim = maxD)
                                                        u = commonind(ΓA,λA)
                                                        v = commonind(ΓB,λA)
                                                        a = Index(dim(u),tags="A_bond")
                                                        λA = replaceinds(λA/sqrt(sum(λA.^2)),[u,v],[a,a'])
                                                        ΓA = replaceind(replaceind(1 ./λB,b',α)*ΓA,u,a)
                                                        ΓB = replaceind(replaceind(1 ./λB,b,β)*ΓB,v,a')
    
                                                        if gpu
                                                            ΓA = GPU(ΓA)
                                                            ΓB = GPU(ΓB)
                                                            λA = GPU(λA)
                                                            λB = GPU(λB)
                                                        end
                                                        
                                                        return Dict{String,Any}("ΓA"=>ΓA, "λA"=>λA, "ΓB"=>ΓB,"λB"=>λB,"sites"=>sites)
                                                    end
                                                end
                                            end
                                        end
                                    end
                                 end
                            end
                        end
                    end
                end
            end
        end
    end
end

function TEBDstep(ΓA, λA, ΓB, λB, U, sites; maxdim::Int64=512, cutoff::Float64=1e-10)
    # Use ITensor's dimension functions more efficiently
    γ = Index(dim(commonind(ΓA, λB)), "γ_bond")

    # Optimize tensor contraction, avoid intermediate tensor copies if possible
    θ = λB * replaceind(noprime(ΓA * λA * ΓB * U, sites'), commonind(ΓB, λB), γ) *
        replaceind(λB, commonind(ΓB, λB), γ)
    
    # Perform SVD with a cutoff
    X, λ, Y = svd(θ, uniqueind(λB, ΓA), uniqueind(noprime(ΓA), noprime(ΓB)), 
                      cutoff=cutoff, maxdim=maxdim)
    
    # Normalize singular values
    λ_sum_squares = sum(λ.^2)
    if λ_sum_squares > 0
        λ = λ / sqrt(λ_sum_squares)
    end
    
    # Canonicalization step (optimized for clarity)
    ix = commonind(X, λ)
    iy = commonind(Y, λ)
    
    α = Index(dim(ix); tags=tags(commonind(ΓA, λA)), plev=plev(commonind(ΓA, λA)))

    # Replace indices in a more direct way
    replaceinds!(λ, (ix, iy), ifelse(plev(α) == 1, (α, noprime(α)), (α, α')))
    
    # Avoid unnecessary division operation by precomputing
    λB_inv = 1.0 ./ λB
    Γ1 = replaceind(λB_inv * X, ix, α)
    Γ2 = replaceind(λB_inv * Y, iy, ifelse(plev(α) == 1, noprime(α), α'))
    return [Γ1; λ; Γ2]
end

function applyToState(oA,oB,state)
    return Dict{String,Any}("ΓA"=>noprime(op(oA,state["sites"][1])*state["ΓA"],tags="Qubit,Site,n=1"), "λA"=>state["λA"], "ΓB"=>noprime(op(oB,state["sites"][2])*state["ΓB"],tags="Qubit,Site,n=2"),"λB"=>state["λB"],"sites"=>state["sites"])
end

function twoBodyObs(state,o;site::Int=1,gpuFlag::Bool=false)
    ΓA = state["ΓA"]
    ΓB = state["ΓB"]
    λA = state["λA"]
    λB = state["λB"]
    sites = state["sites"]
    Γ = ΓA*λA*ΓB
    α = Index(dim(commonind(ΓB,λB)),tags="α")
    Γl = replaceind(Γ,commonind(ΓB,λB),α)
    if (isa(o,String) || isa(o,Vector{String}))
        if isa(o,String)
            if site == 1
                s2 = 2
            else
                s2 = 1
            end
            operator = op(o, sites[site])*op("Id", sites[s2])
        else
            operator = op(o[1], sites[1])*op(o[2], sites[2])
        end
        #operator = gpuFlag ? GPU(operator) : operator
        #wrt("type of observable: $(typeof(operator)). Is CUDA: $(isa(array(operator),cuda.CuArray))")
    elseif isa(o,ITensor)
        # maybe catch mistake here if o is not a cuda array and gpuflag = true and opposite 
        if Sys.islinux()
            if gpuFlag == isa(array(o),cuda.CuArray)
                operator = o
            else
                operator = []
                error("Operator input for computation of observable is not a GPU array")
            end
        else
            operator = o
        end
    else 
        operator = []
        error("Operator input for computation of observable is faulty. Is type: $(typeof(o))")
    end
    temp = λB * Γl * operator
    temp = λB * temp
    temp = temp * conj.(prime(Γ,sites))
    temp = temp * λB
    temp = temp * replaceind(λB,commonind(ΓB,λB),α)
    return scalar(cpu(temp))
end

# extended observables
function o1o2Corr(state,o,distMax;gpuFlag::Bool=false) 
    ΓA = state["ΓA"]
    ΓB = state["ΓB"]
    λA = state["λA"]
    λB = state["λB"]
    A = commonind(ΓA,λA)
    B = commonind(ΓA,λB)
    α = Index(dim(commonind(ΓA,λA)),tags="α")
    β = Index(dim(commonind(ΓB,λB)),tags="β")
    sites = state["sites"]
    
    # Use in-place operations and transfer tensors to GPU if gpuFlag is true
    o1 = gpuFlag ? GPU(op(o[1], sites[1])) : op(o[1], sites[1])
    o2A = gpuFlag ? GPU(op(o[2], sites[1])) : op(o[2], sites[1])
    o2B = gpuFlag ? GPU(op(o[2], sites[2])) : op(o[2], sites[2])


    # left block starting from site A
    L1 = λB*ΓA*λA
    temp = replaceind(L1*o1,A',α)
    L1 = replaceind(prime(conj.(L1),sites[1]),A',α') * temp

    Atop = replaceinds(ΓA*λA,[B,A'],[β,α])
    Btop = replaceinds(ΓB*λB,[A',B],[α,β])
    
    #right closure when evaluating on B site
    Bend = ΓB*λB
    temp = replaceind(Bend*o2B,A',α)
    Bend = replaceind(prime(conj.(Bend),sites[2]),A',α') * temp

    #right closure when evaluating on A site
    Aend = replaceind(ΓA*λA*ΓB,B,β)*λB
    temp = Aend*o2A
    Aend = prime(conj.(Aend),[sites[1],β])*temp

    # Pre-allocate the result array, ensure it's a CuArray if GPU is enabled
    val = Vector{ComplexF64}(undef, distMax)

    # Compute the first correlation value
    val[1] = scalar(L1 * Bend)
    
    for i in 1:distMax-1
        if iseven(i)
            # i is even
            temp = L1 * Atop
            L1 = temp * prime(conj.(Atop),[α,β])
            val[i+1] = scalar(L1*Bend)
        else
            # i is odd
            temp = L1 * Btop
            L1 = temp * prime(conj.(Btop),[α,β])
            val[i+1] = scalar(L1*Aend)
        end
        
    end
    return val
end

# compute 1-site density matrix
function ρ1fct(state;site::Int64=1)
    ΓA = state["ΓA"]
    ΓB = state["ΓB"]
    λA = state["λA"]
    λB = state["λB"]
    B = commonind(ΓA,λB)
    α = Index(dim(commonind(ΓB,λB)),tags="α")
    β = Index(dim(commonind(ΓB,λB)),tags="β")
    
    Γ = replaceind(λB,B',α)*ΓA*λA*ΓB*replaceind(λB,B,β)
    return array(cpu(Γ*prime(conj.(Γ),tags(state["sites"][site]))))
end

# compute 2-site density matrix
function ρ2fct(state)
    ΓA = state["ΓA"]
    ΓB = state["ΓB"]
    λA = state["λA"]
    λB = state["λB"]
    B = commonind(ΓA,λB)
    α = Index(dim(commonind(ΓB,λB)),tags="α")
    β = Index(dim(commonind(ΓB,λB)),tags="β")
    phys = combiner(state["sites"],tags="phys")
    
    Γ = replaceind(λB,B',α)*ΓA*λA*ΓB*replaceind(λB,B,β)
    Γ = Γ * phys
    return array(cpu(Γ*prime(conj.(Γ),"phys")))
end

# compute 3-site density matrix
function ρ3fct(state)
    ΓA = state["ΓA"]
    ΓB = state["ΓB"]
    λA = state["λA"]
    λB = state["λB"]
    B = commonind(ΓA,λB)
    α = Index(dim(commonind(ΓB,λB)),tags="α")
    β = Index(dim(commonind(ΓB,λB)),tags="β")
    sites = state["sites"]
    s3 = Index(dim(sites[1]),tags=tags(replacetags(sites[1], "n=1" => "n=3")))
    phys = combiner([sites[1],sites[2],s3],tags="phys")

    R = λA*ΓB*λB
    R = R * prime(conj.(R),commonind(ΓA,λA))
    rest = phys*(replaceind(λB,B',α)*ΓA*λA*ΓB*λB*replaceind(ΓA,sites[1],s3))
    R = rest*R
    rest = prime(conj.(rest),"phys")
    R = R*prime(rest,commonind(ΓA,λA))
    return array(cpu(R))
end

# trace distance
function trD(A,B)
    return 0.5*real(tr(sqrt((A-B)'*(A-B))))
end

# extract SV's to julia array
function SV(λ)
    λtemp = cpu(λ)
    return diag(array(λtemp))
end

function TEBD(sysDef;time_lim = Inf)
    timeStart = time()
    global stop_loop = false
    gpuFlag = sysDef["gpuFlag"]
    ΓA = gpuFlag ? GPU(sysDef["state"]["ΓA"]) : sysDef["state"]["ΓA"]
    ΓB = gpuFlag ? GPU(sysDef["state"]["ΓB"]) : sysDef["state"]["ΓB"]
    λA = gpuFlag ? GPU(sysDef["state"]["λA"]) : sysDef["state"]["λA"]
    λB = gpuFlag ? GPU(sysDef["state"]["λB"]) : sysDef["state"]["λB"]
    
    tInit = sysDef["t0"]
    sites = sysDef["state"]["sites"]
    Hilbert_dim = dim(sites[1])
    co = sysDef["cutoff"]
    maxD = sysDef["maxBD"]
    Nsteps = sysDef["Nsteps"]
    step = sysDef["step"]
    tlist = [tInit+n*step for n in 0:Nsteps]
    ObsDef = sysDef["Obs"]
    ObFlag = ObsDef["flag"]
    monitor = sysDef["monitor"]
    mflag = monitor["flag"]
    convergence = sysDef["convergence"]
    cflag = convergence["flag"]

    if mflag
        if Sys.islinux()
            if isa(array(ΓA),cuda.CuArray)
                wrt("Running on gpu")
            else
                wrt("Running on cpu multithreading with $(BLAS.get_num_threads()) threads")
            end
        end
    end

    # Initialize Hamiltonian and gates
    Ht = sysDef["Ht"]
    if Ht.time_dep
        Hcpu = Ht.H(tInit)
        U1, U2 = gpuFlag ? GPU(T2Gates(Hcpu,step,imag=false)) : T2Gates(Hcpu,step,imag=false)
        H = gpuFlag ? GPU(Hcpu) : Hcpu
    else
        H = gpuFlag ? GPU(Ht.H) : Ht.H
        U1 = gpuFlag ? GPU(sysDef["U1"]) : sysDef["U1"]
        U2 = gpuFlag ? GPU(sysDef["U2"]) : sysDef["U2"]
    end
    
    # set up initial state
    state = Dict{String,Any}("ΓA"=>ΓA, "λA"=>λA, "ΓB"=>ΓB,"λB"=>λB,"sites"=>sites)

    # set up observables and arrays to store them
    if ObFlag; 
        obsΔ = ObsDef["Δ"]
        # vector with all the steps where the observables will be calculated
        ObsSteps = [i for i in 0:obsΔ:Nsteps]
        if ObsSteps[end]!=Nsteps
            push!(ObsSteps,Nsteps)
        end

        # build NamedTuple for storing observables
        obs = (;)
        if ObsDef["H"]
            obs = (; obs..., H = Vector{Float64}(undef, length(ObsSteps)))
        end
        if cflag
            if convergence["state"]
                obs = (; obs..., ϵΓ = Vector{Float64}(undef, length(ObsSteps)))
            end
            if convergence["SV"]
                obs = (; obs..., ϵλ = Vector{Float64}(undef, length(ObsSteps)))
            end
            obs = (; obs..., conv = Vector{Dict{String,Any}}(undef, length(ObsSteps)))
        end
        if ObsDef["λ"]
            obs = (; obs..., λ = Vector{Vector{Float64}}(undef, length(ObsSteps)))
        end
         if ObsDef["ρ1"]
            obs = (; obs..., ρ1 = [fill(0.0+0.0im,Hilbert_dim,Hilbert_dim) for n in 1:length(ObsSteps)])
        end
        if ObsDef["ρ2"]
            obs = (; obs..., ρ2 = [fill(0.0+0.0im,Hilbert_dim^2,Hilbert_dim^2) for n in 1:length(ObsSteps)])
        end
        if ObsDef["ρ3"]
            obs = (; obs..., ρ3 = [fill(0.0+0.0im,Hilbert_dim^3,Hilbert_dim^3) for n in 1:length(ObsSteps)])
        end
        if length(ObsDef["OneSite"])>0
            obs = (; obs..., (Symbol(k) => Vector{ComplexF64}(undef, length(ObsSteps)) for k in ObsDef["OneSite"])...)
            OneSiteobs = Dict{String,ITensor}()
            for o in ObsDef["OneSite"]
               if (isa(o,String) || isa(o,Vector{String}))
                    if isa(o,String)
                        operator = op(o, sites[1])*op("Id", sites[2])
                    else
                        operator = op(o[1], sites[1])*op(o[2], sites[2])
                    end 
                else operator = o
                end
                operator = gpuFlag ? GPU(operator) : operator
                OneSiteobs[o]=operator
            end
        end
        if length(ObsDef["2p-corr"])>0
            obs = (; obs..., (Symbol(k[1]*k[2]) => Vector{Vector{ComplexF64}}(undef, length(ObsSteps)) for k in ObsDef["2p-corr"])...)
        end
    
        # function to update observables 
       function obsUpd!(iterObs,H)
            if haskey(obs,:H)
                obs.H[iterObs] = real(twoBodyObs(state,H,gpuFlag=gpuFlag))
            end
            if haskey(obs,:λ)
                obs.λ[iterObs] = SV(state["λA"])
            end
            if haskey(obs,:ρ1)
                obs.ρ1[iterObs] = ρ1fct(state)
            end
            if haskey(obs,:ρ2)
                obs.ρ2[iterObs] = ρ2fct(state)
            end
            if haskey(obs,:ρ3)
                obs.ρ3[iterObs] = ρ3fct(state)
            end
            if length(ObsDef["OneSite"]) >0
                for k in ObsDef["OneSite"]
                    obs[Symbol(k)][iterObs] = twoBodyObs(state,OneSiteobs[k],gpuFlag=gpuFlag)
                end
            end
            if length(ObsDef["2p-corr"]) >0
                for k in ObsDef["2p-corr"]
                    obs[Symbol(k[1]*k[2])][iterObs] = o1o2Corr(state,[k[1],k[2]],k[3],gpuFlag=gpuFlag)
                end
            end
            return iterObs+=1
        end
        iterObs = 1
        #compute observable for initial state
        if ObFlag;
            iterObs = obsUpd!(iterObs,H)
            if mflag
                if sum([haskey(obs,Symbol(monitor["Obs"][i])) for i in 1:length(monitor["Obs"])])!=length(monitor["Obs"])
                    error("Wanted observables can not be moitored as they are not computed observables")
                end
                str = "Initial state::"
                if "λ" ∈ monitor["Obs"]
                    str *= " Bond dim. = $(length(obs.λ[iterObs-1])),"
                end
                for o in Iterators.filter(x -> x != "λ", monitor["Obs"])
                    str *= " <"*o*"> = $(round(obs[Symbol(o)][iterObs-1],sigdigits=3)),"
                end
                str = String(chop(str))
                wrt(str)
            end
        end
        
    end
    
    # do time evolution
    t0 = time()
    Γpre = 0.
    λpre = 0.
    ϵλ = 10
    ϵΓ = 10
    converged = false
    if mflag && monitor["pm"]
        prog=PM.Progress(Int(Nsteps); showspeed=true)
    end
    if cflag
        conv_2step = 0
        conv = Dict{String,Any}()
        if convergence["state"] # one could be more rigorous and use the 2-site density matrix instead and compute the HIlbert-Schmidt overlap, but it is more expensive
            Γpre = array(λB*ΓA*λA*ΓB)
            obs.ϵΓ[1] = ϵΓ
            conv["state"] = 0
        end
        if convergence["SV"]
            λpre = vcat(diag(array(λA)),diag(array(λB)))
            obs.ϵλ[1] = ϵλ
            conv["SV"] = 0
        end
        obs.conv[1] = copy(conv)
    end
    
    for iter in 1:Nsteps
        if Ht.time_dep
            Hcpu = Ht.H(tlist[iter])
            U1, U2 = gpuFlag ? GPU(T2Gates(Hcpu,step,imag=false)) : T2Gates(Hcpu,step,imag=false)
            H = gpuFlag ? GPU(Hcpu) : Hcpu
        end
        # Second order Trotter scheme
        ΓA,λA,ΓB = TEBDstep(ΓA,λA,ΓB,λB,U1,sites,cutoff=co,maxdim=maxD)
        ΓB,λB,ΓA = TEBDstep(ΓB,λB,ΓA,λA,U2,sites,cutoff=co,maxdim=maxD)
        ΓA,λA,ΓB = TEBDstep(ΓA,λA,ΓB,λB,U1,sites,cutoff=co,maxdim=maxD)

        # update state
        state["ΓA"]=ΓA
        state["ΓB"]=ΓB
        state["λA"]=λA
        state["λB"]=λB

        # compute observable at specific time stamps
        if ObFlag;
            if iter == ObsSteps[iterObs];
                iterObs = obsUpd!(iterObs,H)

                # check for convergence of state
                if cflag
                    if convergence["state"]
                        Γt = array(λB*ΓA*λA*ΓB)
                        ΔΓ = Γt-Γpre
                        ϵΓ = sum(abs.(ΔΓ))/sum(abs.(Γpre))
                        obs.ϵΓ[iterObs-1] = ϵΓ
                        if ϵΓ < convergence["stateΔ"]
                            conv["state"] = 1
                        else
                            conv["state"] = 0
                        end
                        Γpre = Γt
                    end
                    if convergence["SV"]
                        λt = vcat(diag(array(λA)),diag(array(λB)))
                        if size(λt)==size(λpre)
                            Δλ = λt-λpre
                            ϵλ = maximum(abs.(Δλ)./abs.(λpre))
                        else
                            ϵλ = 10.
                        end
                        obs.ϵλ[iterObs-1] = ϵλ
                        if ϵλ < convergence["SVΔ"]
                            conv["SV"] = 1
                        else
                            conv["SV"] = 0
                        end
                        λpre = λt
                    end
                    obs.conv[iterObs-1] = copy(conv)
                    if prod(values(conv)) == 1
                        conv_2step += 1
                        if conv_2step>1
                            iterObs += -1
                            # shorten observables to the correct length
                            shrtZ(v) = v[begin:iterObs]
                            obs = (; (k =>shrtZ(v) for (k,v) in zip(keys(obs), obs))...)
                            ObsSteps = ObsSteps[begin:iterObs]
                            ObsSteps[end] = iter+1
                            wrt("Calculation converged")
                            break
                        end
                    else
                        conv_2step = 0
                    end
                end
                # # compare to external cutoff data and stop calculation of deviation exceeds cutoff
                # if DZcut != false
                #     deltaZ = abs(obs.Z[iterObs-1]-DZ[iterObs-1])/abs(obs.Z[iterObs-1])
                #     if deltaZ > DZcut
                #     # compute observables and adapt output
                #         if ObFlag
                #             if iter != ObsSteps[iterObs];
                #                 obsUpd!(iterObs,H,ObType)
                #             elseif iter == ObsSteps[iterObs];
                #                 iterObs += -1 
                #             end
                #             # shorten observables to the correct length
                #             shrtZ(v) = v[begin:iterObs]
                #             obs = (; (k =>shrtZ(v) for (k,v) in zip(keys(obs), obs))...)
                #             ObsSteps = ObsSteps[begin:iterObs]
                #             ObsSteps[end] = iter+1
                #         else
                #             tlist = tlist[iter]
                #         end
                        
                #         break
                #     end
                # end
                
                if mflag && !monitor["pm"]
                    str = "iter: $(iter)/$(Nsteps), sim. time: $(round(tlist[iter],sigdigits=2))/$(round(tlist[end],sigdigits=2)), time for $(obsΔ) steps: $(round(time()-t0,sigdigits=2)) sec,"
                    if "λ" ∈ monitor["Obs"]
                        str *= " Bond dim. = $(length(obs.λ[iterObs-1])),"
                    end
                    if cflag
                        if convergence["state"]
                            str *= " convergence of state = $(ϵΓ),"
                        end
                        if convergence["SV"]
                            str *= " convergence of SV's = $(ϵλ),"
                        end
                        str *= " convergence flags: ("*join([d[1]*" = $(d[2])" for d in pairs(conv)],", ")*"),"
                    end
                    for o in Iterators.filter(x -> x != "λ", monitor["Obs"])
                        str *= " <"*o*"> = $(round(obs[Symbol(o)][iterObs-1],sigdigits=3)),"
                    end
                    str = String(chop(str))
                    wrt(str)
                    if gpuFlag
                        GPUprint()
                    end
                    flush(stdout)
                    t0 = time()
                end
            end
        end
        if !ObFlag && mflag;
            if iter%10==0 
                wrt("iter: $(iter)/$(Nsteps), sim. time: $(round(tlist[iter],sigdigits=2))/$(round(tlist[end],sigdigits=2)), 
                    Bond: $(size(array(λA))[1]) and $(size(array(λB))[1]),
                    time for 10 steps: $(round(time()-t0,sigdigits=2)) sec")
                flush(stdout)
                t0 = time()
            end
        end
        WTbuffer = 30
        if (time()-timeStart + WTbuffer >time_lim && iter>3) || stop_loop
            # compute observables and adapt output
            if ObFlag
                if iter != ObsSteps[iterObs-1]
                    obsUpd!(iterObs,H)
                    if cflag
                        if convergence["state"]
                            Γt = array(λB*ΓA*λA*ΓB)
                            ΔΓ = Γt-Γpre
                            ϵΓ = sum(abs.(ΔΓ))/sum(abs.(Γpre))
                            obs.ϵΓ[iterObs] = ϵΓ
                        end
                        if convergence["SV"]
                            λt = vcat(diag(array(λA)),diag(array(λB)))
                            if size(λt)==size(λpre)
                                Δλ = λt-λpre
                                ϵλ = maximum(abs.(Δλ)./abs.(λpre))
                            else
                                ϵλ = 10.
                            end
                            obs.ϵλ[iterObs] = ϵλ
                        end
                        obs.conv[iterObs] = copy(conv)
                    end
                elseif iter == ObsSteps[iterObs-1]
                    iterObs += -1 
                end
                # shorten observables to the correct length
                shrt(v) = v[begin:iterObs]
                obs = (; (k =>shrt(v) for (k,v) in zip(keys(obs), obs))...)
                ObsSteps = ObsSteps[begin:iterObs]
                ObsSteps[end] = iter+1
            else
                tlist = tlist[iter]
            end
            if stop_loop
                wrt("!!!! computaiton stopped externally !!!!")
            else
                wrt("!!!!Computation stopped with $(WTbuffer)$sec left of wall time !!!!")
            end
            flush(stdout)
            break
        end
        if mflag && monitor["pm"]
            PM.next!(prog)
        end
    end
    # move results to cpu
    if gpuFlag
        state["ΓA"]=cpu(ΓA)
        state["ΓB"]=cpu(ΓB)
        state["λA"]=cpu(λA)
        state["λB"]=cpu(λB)
    end
    
    out = if ObFlag; ObsSteps .+= 1; Dict{String,Any}("state"=>state,"obs"=>obs,"tlist"=>tlist[ObsSteps],"computation time"=>trunc(Int,time()-timeStart)) else 
        Dict{String,Any}("state"=>state,"t_tinal"=>tlist,"computation time"=>trunc(Int,time()-timeStart)) end     
    return out
end

function T2Gates(H::ITensor,step::Float64;imag::Bool=false) 
    if imag; U1 = exp(-0.5*step * H);U2 = exp(-step * H) else U1 = exp(- im*0.5*step * H); U2 = exp(- im*step * H) end
    return [U1;U2]
end

function SysInit(state::Dict{String, Any},Ht,parameters::Dict{String,Any};imag::Bool=false,descriptor::String="genericDescriptor",
        folder::String=pwd(),SavePoints::Int64=0,gpu::Bool=false) 
    par = parameters
    sites=state["sites"]
    if Ht.time_dep
        H = Ht.H(0.0)
    else
        H = Ht.H
    end
    U1, U2 = T2Gates(H,parameters["step"],imag=imag)
    
    # check that monitor dictionary is initialised
    if haskey(par,"monitor")
        if !haskey(par["monitor"],"flag")
            par["monitor"]["flag"] = false
        end
    else
        par["monitor"] = Dict{String,Any}("flag"=>false)
    end
    # check that observeable dictionary is initialised
    if haskey(par,"Obs")
        if !haskey(par["Obs"],"flag")
            par["Obs"]["flag"] = false
        end
    else
        par["Obs"] = Dict{String,Any}("flag"=>false)
    end

    # check that convergence dictionary is initialised
    if haskey(par,"convergence")
        if !haskey(par["convergence"],"flag")
            par["convergence"]["flag"] = false
        end
    else
        par["convergence"] = Dict{String,Any}("flag"=>false)
    end
        
    return merge(Dict{String,Any}("t0"=>0.0,"U1"=>U1,"U2"=>U2,"state"=>state,"H"=>H, "gpuFlag"=>gpu,"descriptor"=>descriptor,
            "folder"=>folder,"SavePoints"=>SavePoints,"imag"=>imag,"Ht"=> Ht),par)
end

function SysUpd(sysInit::Dict{String,Any},keys::Vector{String},vals::Vector{}) 
    out = Dict(sysInit)
    for i in 1:length(keys)
        out[keys[i]]= vals[i]
    end
    return out
end

# function to apply a bra and ket gate to a 2-site density matrix and perform the truncated svd
function ρTEBDstep(ρl,midI,ρr,outerI,Uket,Ubra,sites;maxD::Int64=512,co::Float64=1e-10)
    # Contracted tensor 
    θ = noprime(prime(ρl,outerI) * ρr * Uket*Ubra,prime(sites))
    # svd 
    X, λ, Y = svd(θ,prime(outerI),uniqueinds(ρl,ρr),cutoff=co,maxdim=maxD)
    # normalize SV's
    λ = λ/sum(λ)
    # define new tensors and updated middle index
    midI1 = Index(dim(inds(λ)[1]),tags(midI))
    ρ1 = replaceind(noprime(X*sqrt.(λ),prime(outerI)),commonind(λ,Y),midI1)
    ρ2 = replaceind(Y*sqrt.(λ),commonind(λ,X),midI1)
    [ρ1;midI1;ρ2;λ]
end

# renormalize the infinite 2-site denisty matrix 
function ρiRenorm(ρA,ρB,sites)
    M = array(delta(sites[1],sites[2])*prime(ρA,tags="outerI")*ρB*delta(sites[3],sites[4]))
    λ, ϕ = eigen(M,sortby=x -> -abs(x))
    λ = λ[1]
    ϕ = ϕ[:,1]
    ϕ = ITensor((ϕ' .* ϕ),inds(prime(ρA,tags="outerI"),plev=1)[1],
        noprime(inds(prime(ρA,tags="outerI"),plev=1)[1]))
    [ρA ./sqrt(λ);ρB ./sqrt(λ);ϕ]
end

# two-point correlation function from density matrix iMPS
function ρo1o2Corr(ρAin,ρBin,phys,o1in,o2in,dmax;maxEV_proj=1)
    if typeof(maxEV_proj)!=ITensors.ITensor
        # compute trace projection
        ρA,ρB,maxEV_proj = ρiRenorm(ρAin,ρBin,phys)
    end

     # first tensor
    L = prime(ρA * delta(phys[1]',phys[2])*op(o1in,phys[1]),tags="outerI")
    # identity on site A
    Ia = ρA * delta(phys[1],phys[2])
    # identity on site B
    Ib = ρB * delta(phys[3],phys[4])
    # expectation value of single site
    o1 = scalar(L*Ib*maxEV_proj)
    o2 = scalar(prime(ρA * delta(phys[1]',phys[2])*op(o2in,phys[1]),tags="outerI")*Ib*maxEV_proj)
    # second operator on site A with identity on site B
    o2a = ρA*delta(phys[1]',phys[2])*op(o2in,phys[1])*prime(Ib,tags="outerI")
    # second operator on site B
    o2b = ρB*delta(phys[3]',phys[4])*op(o2in,phys[3])
    
    corr = Vector{ComplexF64}(undef,dmax)
    
    for i in 1:dmax
        if isodd(i)
            corr[i] = scalar(L*o2b*maxEV_proj)
            L *= Ib  
        else
            corr[i] = scalar(mapprime(L*mapprime(o2a,1,2),2,0)*maxEV_proj)
            L *= Ia
        end
    end
    return corr.-o1*o2
end

# compute energy and a two-site observable of the infinite two-site density matrix
function ρobs(ρA,ρB,maxEV_proj,phys,H,o1,o2)
    # observable
    expec = scalar(delta(phys[1]',phys[2])*(op(o1,phys[1])*op(o2,phys[3])*prime(ρA,tags="outerI")*ρB)*delta(phys[3]',phys[4])*maxEV_proj)
    # energy
    En = scalar(delta(phys[1]',phys[2])*(H*prime(ρA,tags="outerI")*ρB)*delta(phys[3]',phys[4])*maxEV_proj)
    [real.(En),expec]
end

# compute compute generic two-site observable of the infinite 2-site density matrix
function ρ_gen_2obs(ρA,ρB,phys,o1,o2;maxEV_proj=1)
    if typeof(maxEV_proj)!=ITensors.ITensor
        # compute trace projection
        ρA,ρB,maxEV_proj = ρiRenorm(ρA,ρB,phys)
    end
    return scalar(delta(phys[1]',phys[2])*(op(o1,phys[1])*
        op(o2,phys[3])*prime(ρA,tags="outerI")*ρB)*delta(phys[3]',phys[4])*maxEV_proj)
end

# compute the reduced 3-site density matrix for an infinite MPO
function ρ3ρ(ρA,ρB,maxEV_proj,phys)
    Iket = combiner([phys[1],phys[3],phys[1]'],tags="row")
    Ibra = combiner([phys[2],phys[4],phys[2]'],tags="column")
    return array(Iket*(replaceprime(prime(ρA,2,tags="outerI")*ρB*prime(ρA,phys)*
            (ρB*delta(phys[3],phys[4])),2,1,tags="outerI")*maxEV_proj)*Ibra)
end

# Starting from infinite temperature, cool down the system until the target energy (EnT) is reached 
function iρβ(Hfct,EnT,SiteType,o1,o2;BD=512,pm=false) 
    # set up legs initial tensors
    phys = siteinds(SiteType,4)
    oI = Index(1,"outerI")
    mI = Index(1,"midI")
    λmid = Vector{Float64}()
    λout = Vector{Float64}()
    Hdim = dim(phys[1])
    
    # site A MPO starting as an identity
    ρA = ITensor(oI,phys[1],phys[2],mI)
    for i in 1:Hdim
        ρA[phys[1]=>i,phys[2]=>i,oI=>1,mI=>1] = 1/Hdim
    end
    
    # site B MPO starting as an identity
    ρB = ITensor(mI,phys[3],phys[4],oI)
    for i in 1:Hdim
        ρB[phys[3]=>i,phys[4]=>i,mI=>1,oI=>1] = 1/Hdim
    end

    ρA,ρB,maxEV_proj = ρiRenorm(ρA,ρB,phys)
    Hket = Hfct([phys[1], phys[3]])
    Hbra = Hfct([phys[2], phys[4]])

    dτ = 1E-1
    β = 0.
    βlst = [β, β]
    obs0 = ComplexF64.(ρobs(ρA,ρB,maxEV_proj,phys,Hket,o1,o2))
    obslst = [obs0, obs0]
    ρVec = [[oI,ρA,mI,ρB], [oI,ρA,mI,ρB]]
    if pm
        prog=PM.Progress(Int(nloop); showspeed=true)
    end
    # loops performed with increasing resolution (set by dτ)
    for i in 1:5
        pop!(obslst)
        obs0 = obslst[end]
        pop!(βlst)
        β = βlst[end]
        oI,ρA,mI,ρB = ρVec[begin]
        ρVec = [[oI,ρA,mI,ρB], [oI,ρA,mI,ρB]]
        dτ *= 0.1
        # construct evolution operators  
        U1ket, U2ket = T2Gates(Hket,dτ/2,imag=true)
        U1bra, U2bra = T2Gates(Hbra,dτ/2,imag=true)
        ρA,ρB,maxEV_proj = ρiRenorm(ρA,ρB,phys)
        while real(obs0[1])>EnT
            β += dτ
            push!(βlst,β)
            # evolve with second order trotter
            ρA,mI,ρB,λmid = ρTEBDstep(ρA,mI,ρB,oI,U1ket,U1bra,phys,maxD=BD)
            ρB,oI,ρA,λout = ρTEBDstep(ρB,oI,ρA,mI,U2ket,U2bra,phys,maxD=BD)
            ρA,mI,ρB,λmid = ρTEBDstep(ρA,mI,ρB,oI,U1ket,U1bra,phys,maxD=BD)
            # normalize
            ρA,ρB,maxEV_proj = ρiRenorm(ρA,ρB,phys)
            
            obs0 = ρobs(ρA,ρB,maxEV_proj,phys,Hket,o1,o2)
            push!(obslst,obs0)
            circshift!(ρVec,1)
            ρVec[end]=[oI,ρA,mI,ρB]
        end
        if pm
            PM.next!(prog)
        end
    end
    obslst =reduce(hcat,obslst)
    ρ3 = ρ3ρ(ρA,ρB,maxEV_proj,phys)
    Dict{String,Any}("Energy"=>obslst[1,:],"expectation_value"=>obslst[2,:],"ρ3"=>ρ3,"β"=>βlst,"BD"=>BD,"EnergyTarget"=>EnT,"ρA"=>ρA,"ρB"=>ρB,"midInd"=>mI,"outerInd"=>oI,
        "λmid"=>λmid,"λout"=>λout,"physInd"=>phys,"maxEV_proj"=>maxEV_proj)
end

# generate a random test state with bond dimension n by evolution with random unitary
function tstState(n::Int,siteType::String;gpuFlag::Bool=false,Ndim=4)
    if siteType =="Boson"
        sites=siteinds(siteType,dim=Ndim,2)
    else
        sites=siteinds(siteType,2)
    end
    α = Index(1,"A_bond")
    β = Index(1,"B_bond")
    d = dim(sites[1])
    #random unitary 
    M = randn(ComplexF64, d^2, d^2)
    Q, R = qr(M)
    D = Diagonal(R) ./ abs.(Diagonal(R))
    # Equivalent more transparent but complicated method
    #T = Q * Diagonal(D)
    # C1 = combiner(sites[1],sites[2],tags="C1")
    # C1i = combinedind(C1)
    # C2i = Index(dim(C1i),tags="C2")
    # U = ITensor(T,C1i,C2i)
    # U = U*C1*replaceind(C1',C1i',C2i)
    
    # code-wise simpler and equivalent form of U 
    T = reshape(Q * Diagonal(D),d,d,d,d)
    U = ITensor(T,sites[1],sites[2],sites[1]',sites[2]')
    
    ΓA = ITensor(β,sites[1],α)
    ΓB = ITensor(α',sites[2],β')

    coef = randn(ComplexF64,d)
    coef *= 1/sqrt(sum(abs.(coef).^2))

    for n in 1:d
        ΓA[sites[1]=>n,α=>1,β=>1] = coef[n]
        ΓB[sites[2]=>n,α'=>1,β'=>1] = coef[n]
    end

    λA = ITensor(α,α')
    λA[α=>1,α'=>1] = 1.0 
    
    
    λB = ITensor(β,β')
    λB[β=>1,β'=>1] = 1.0 

    for i in 1:100 
        ΓA,λA,ΓB = TEBDstep(ΓA,λA,ΓB,λB,U,sites,cutoff=1E-16,maxdim=n)
        ΓB,λB,ΓA = TEBDstep(ΓB,λB,ΓA,λA,U,sites,cutoff=1E-16,maxdim=n)
        if dims(λA)==dims(λB) && dims(λA)[1]==n
            break
        end
    end

    state = Dict{String,Any}("ΓA"=>ΓA, "λA"=>λA, "ΓB"=>ΓB,"λB"=>λB,"sites"=>sites)
    norm = twoBodyObs(state,"Id")
    ΓA *= 1/norm^(1/4)
    ΓB *= 1/norm^(1/4)
    
    ΓA = gpuFlag ? GPU(ΓA) : ΓA
    ΓB = gpuFlag ? GPU(ΓB) : ΓB
    λA = gpuFlag ? GPU(λA) : λA
    λB = gpuFlag ? GPU(λB) : λB
    U = gpuFlag ? GPU(U) : U

    [Dict{String,Any}("ΓA"=>ΓA, "λA"=>λA, "ΓB"=>ΓB,"λB"=>λB,"sites"=>sites),U]
end

end # module uTEBD
