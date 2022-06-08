using LinearAlgebra
using StaticArrays
using PyPlot
using Random
using ProgressBars
using Optim
using StatsBase
using Statistics
using Roots
using Dates
using MAT

struct ray2d
    origin::Union{SVector{2,Float64},Vector{Float64}}
    dir::Union{SVector{2,Float64},Vector{Float64}}
    inv_direction::Union{SVector{2,Float64},Vector{Float64}}
    sign::Union{Array{Int64,1},SVector{2,Int64}}
end


function ray2d(p1::T, p2::T) where {T}
    dir::T = p2 - p1
    invdir::T = 1.0 ./ dir
    sign = (invdir .< 0) .* 1
    return ray2d(p1, dir, invdir, sign)
end

function ray2ddir(p1::T, dir::T) where {T}
    invdir::T = 1.0 ./ dir
    sign = (invdir .< 0) .* 1
    return ray2d(p1, dir, invdir, sign)
end

@inline function intersects(
    r1::ray2d,
    r2::ray2d,
) 
    if (r1.dir[1]*r2.dir[2] == r1.dir[2]*r2.dir[1])
       return SVector(Inf, Inf)
    end

    try
        #M =  [[r1.dir[1], -r2.dir[1]] [r1.dir[2], -r2.dir[2] ] ]'
        M = SMatrix{2,2}(r1.dir[1], r1.dir[2], -r2.dir[1], -r2.dir[2])
        v = M\(-r1.origin + r2.origin)
        # println(r1.origin + r1.dir*v[1])
        return v
        
    catch
        return SVector(-Inf, -Inf)

    end

end





function fancurves(points, Nproj;src_shift_func = a->[0,0.0],translation=[0,0.0],src_to_det_init=[0,1],det_axis_init = nothing,det_shift_func=a->[0,0.0],apart=range(0,stop=pi/2,length=Nproj),det_radius=sqrt(2),src_radius=sqrt(2),dpart=range(-sqrt(2), stop = sqrt(2), length = 100))

    @assert length(apart) == Nproj
    Nps = size(points,2)

    ivals = zeros(Nproj,Nps)

    sdinit = 2*pi*1/4+atan(src_to_det_init[2],src_to_det_init[1]) 
    apart = apart .+ sdinit
    translation = [translation[2], -translation[1]]
    dextra_rot = 0.0

    if (det_axis_init === nothing)
        dextra_rot = 0.0
    else   
        dinit = 2*pi*1/4+atan(det_axis_init[2],det_axis_init[1])
        dextra_rot = mod((2*pi*1/4-(sdinit-dinit)),2*pi)
    end
    
    span = dpart


    for j = 1:Nps

    
        point0 = SVector(points[1,j], points[2,j])

        for i = 1:Nproj

            #rays[i] = Vector{ray2d}(undef,Nrays)

            src_tangent = SVector(-sin(apart[i]), cos(apart[i]))
            src_orth = SVector(cos(apart[i]), sin(apart[i]))
            srcshift = src_shift_func(apart[i])
            source = SVector(cos(apart[i]), sin(apart[i]))*src_radius + srcshift[1]*src_orth + srcshift[2]*src_tangent     
            p2 = source + translation

            detcenter = -SVector(cos(apart[i]), sin(apart[i]))*det_radius  
            det_tangent = SVector(-sin(apart[i]), cos(apart[i]))
            det_orth = SVector(cos(apart[i]), sin(apart[i]))
            detshift = det_shift_func(apart[i])
            totcenter = detcenter - detshift[1]*det_orth - detshift[2]*det_tangent

            
            ray = ray2d(p2, point0)

            aux1 = SVector(-sin(apart[i]+dextra_rot), cos(apart[i]+dextra_rot))*span[1] + totcenter
            d1 = aux1 + translation 
            
            aux2 = SVector(-sin(apart[i]+dextra_rot), cos(apart[i]+dextra_rot))*span[end] + totcenter
            d2 = aux2 + translation

            dray = ray2d(d1,d2)
            
            v = intersects(dray,ray)

            ivals[i,j] = v[1]
                    

        end
    end

    return ivals
end





function ESS(w)
    s = sum(w)
    return 1/(norm(w)^2)*s^2
end



function strati(w)
    W = w/sum(w)
    S = cumsum(W)
    N = length(W)
    ix = zeros(Int64,N)
    palat = range(0,stop=1,length=N+1)
    for i = 1:N
        a = palat[i]
        b = palat[i+1]
        U = a+rand()*(b-a)
        v = findfirst( S .> U)
        ix[i] = v
    end
    return ix
end


function systematic(w)
    W = w/sum(w)
    S =cumsum(W)
    N = length(W)
    ix = zeros(Int64,N)
    palat = range(0,stop=1,length=N+1)
    U = rand()*(palat[2]-palat[1])
    for i = 1:N
        kohta = palat[i] + U
        k = findfirst(S .>= kohta)
        ix[i] = k
    end
    return ix
end

@inbounds function logsumexp(w)
    offset = maximum(w)
    s = 0.0
    N = length(w)
    @simd for i = 1:N
        s = s + exp(w[i] - offset)
    end
    return log(s) + offset
end


@inbounds function softmax!(out,w)
    N = length(w)
    m = maximum(w)
    @simd for i = 1:N
        out[i] = w[i] - m
        out[i] = exp(out[i])
    end
    s = sum(out)
    @simd for i = 1:N
        out[i] = out[i]/s
    end
    return  nothing 
end



log_pdf_prior(x,l,u) = log(all(x .< u))*( all(x .> l))
sample_prior(l,u) = rand(length(l)).*(u-l) + l


import Base.*

function  *(A::Cholesky{Float64, Matrix{Float64}}, b::Float64)
    return Cholesky(A.U*sqrt(b))

end

mutable struct ramtuning  
    n::Int64
    acc::Int64
    gamma::Float64
    Nadapt::Int64
    C::Array{Float64,2}
    Cho::Cholesky{Float64,Array{Float64,2}}
    xm::Vector{Float64}
end


function mcmc(X,xcache,lp,nonTlp0,T,tuning;Niter)
    #pdf0 = lp(X)
    Xret = X# copy(X)
    Xn = xcache#copy(X)
    dim = length(tuning.xm)
    atarg = 0.238
    for _ = 1:Niter
        tuning.n = tuning.n + 1
        u = randn(dim)
        Xn .= Xret + tuning.Cho.L*u
        nonTlpN = lp(Xn)
        if (log(rand()) < (nonTlpN-nonTlp0)*T)
            Xret .= Xn
            nonTlp0 = nonTlpN
            tuning.acc = tuning.acc +1
        end

        if (tuning.n<=tuning.Nadapt)

            #C = R*(diago+eta*(acc/n-atarg)*(u*u')/(u'*u))*R'
            #R = cholesky(Hermitian(C)).L
            #println(C)
            eta =  min(1, dim * tuning.n^(-tuning.gamma))
            u = u/norm(u)
            z = sqrt(eta * abs(tuning.acc/tuning.n - atarg)) * tuning.Cho.L*u
            if (tuning.acc/tuning.n >= atarg)
                lowrankupdate!(tuning.Cho, z)
            else
                lowrankdowndate!(tuning.Cho, z)
            end

        end

    end
    return Xret,nonTlp0
end


function smc_sample(log_pdf,N,lowb,upb;K=100,NMCMC=20)
    dim = length(lowb)
    X = zeros(K,N,dim)
    W = ones(K,N)/N
    w = ones(N)/N
    lw = log.(w)
    lwcache = similar(lw)
    wcache = similar(w)
    essv = zeros(K)
    eta = 0.90
    T = zeros(K)
    X0 = zeros(N,dim)
    xcache = similar(X0)
    xcache2 = similar(X0)
    tunings = Vector{ramtuning}(undef,N)
    tuningsAUX = Vector{ramtuning}(undef,N)
    sample_gamma0() = sample_prior(lowb,upb)
    log_pdf_gamma0(x) = log_pdf_prior(x,lowb,upb)

    for i = 1:N
        X0[i,:] = sample_gamma0()
        X[1,i,:] = X0[i,:]   
        tunings[i] = ramtuning(dim,0,0.55,2^16,-10.0*Matrix(I(dim)),cholesky(Matrix(0.5*I(dim))),randn(dim))
    end
    
    essv[1] = N

    nonTlpX = zeros(N)
    for i = 1:N
        nonTlpX[i] = log_pdf(X[1,i,:])
    end
   


    function initT(wcache,lwcache,lwprev,nonTlp,ESS_prev)   


        function f(Tn)
            lwcache .= +Tn*nonTlp - lwprev
            softmax!(wcache,lwcache)
            essnew = ESS(wcache)
            return essnew -eta*ESS_prev

        end

        # q = range(0,1e-7,1000)
        # plot(q,f.(q))
        Tnew = nothing
        try
            Tnew = find_zero(f,(0,1.0), Bisection(),rtol=1e-14,abstol=1e-14)
        catch
            Tnew = find_zero(f,(0,1.0), Order0(),rtol=1e-14,abstol=1e-14)
        end
        


        return min(1.0,Tnew)
    end


    function newT(wcache,lwcache,lwprev,nonTlp,Tprev,ESS_prev)
        function f(Tn)
            lwcache .= (-Tprev+Tn)*nonTlp + lwprev
            softmax!(wcache,lwcache)
            essnew = ESS(wcache)
            return essnew -eta*ESS_prev

        end

        if abs(f(1.0)) < 0.01 || Tprev > 0.99
            return 1.0
        end


        Tnew = nothing
        try
            
            Tnew = find_zero(f,(Tprev,1.0), Order0(),rtol=1e-14,abstol=1e-14)
        catch
            Tnew = find_zero(f,(Tprev,1.0), Bisection(),rtol=1e-14,abstol=1e-14)
        end


        return min(1.0,Tnew)
    end

    T1 = initT(wcache,lwcache,lw,nonTlpX,essv[1])
    println(T1)
    T[1] = T1

    for i = 1:N
        pn = log_pdf(X[1,i,:])*T[1]#(temp(1,K))
        po = log_pdf_gamma0(X[1,i,:])   
        lw[i] = pn - po + lw[i]
        
    end

    softmax!(w,lw)
    ess = ESS(w)
    essv[1] = ess
    # if (ess < 0.7*N)
    #     println(1)
    #     #a = sample(1:N,Weights(w),N)
    #     a = strati(w)
    #     X[1,1:N,:] = X[1,a,:]
    #     w .= 1/N                
    # end

    w .= w/sum(w)
    lw .= log.(w)  
  

    for k = 2:K   
        X[k,:,:] .= X[k-1,:,:]  
        Tnew = newT(wcache,lwcache,lw,nonTlpX,T[k-1],essv[k-1])
        T[k] = Tnew
        println(k, ": ", T[k])

        for i = 1:N
            pn = nonTlpX[i]*Tnew #pdf_total(X[k,i,:])
            po =  nonTlpX[i]*T[k-1]# pdf_prev(X[k,i,:])            
            lw[i] = pn-po+lw[i]
            tunings[i].Cho =  tunings[i].Cho*(T[k-1]/Tnew)
        end
       
        softmax!(w,lw)     
       

        if (ess < 0.7*N)
            println("Resampling.")
            #a = sample(1:N,Weights(w),N)
            a = strati(w)
            #a = systematic(w)
            X[k,1:N,:] = X[k,a,:]
            nonTlpX = nonTlpX[a]
            #b = sample(1:N,N)
            b = a
            for i = 1:N
                tuningsAUX[i] = deepcopy(tunings[b[i]])
                #tuningsAUX[i] = deepcopy(tunings[a[i]])               
            end
            tunings = tuningsAUX
            w .= 1/N                 
        end
        w .= w/sum(w)
        lw .= log.(w) 
        W[k,:] .=  w

        ess = ESS(w)
        essv[k] = ess

       # 
       @Threads.threads for i = 1:N
           xcache2[i,:] .= X[k,i,:]
           xcache[i,:] .= X[k,i,:]
           Xn, nonTlpN = mcmc(view(xcache2,i,:),view(xcache,i,:),log_pdf,nonTlpX[i],T[k],tunings[i];Niter=NMCMC)
           X[k,i,:] .= Xn
           nonTlpX[i] = nonTlpN 
           
        end
        
        

    end

    return W,X,essv,T,X0,tunings
end


function test_all(;angle=true,a_fixed=1.2)

    no_of_angles = 20

    a_true = pi/3
    a_fixed = a_true

    fun(pts,x) =  fancurves(pts, no_of_angles;src_shift_func = a->[0,x[1]],translation=[0,0.0],src_to_det_init=[0,1.0],det_axis_init = [1,x[2]],det_shift_func=a->[0,x[4]],apart=range(0+x[6],stop=2*pi+x[6],length=no_of_angles),det_radius=x[3],src_radius=x[5],dpart=range(-sqrt(2), stop = sqrt(2), length = 100))

    Random.seed!(1)
    pts = rand(2,6)*0.5
    # Source shift, detector tilt, detector radius, detector shift, source radius, angle
    meas = fun(pts,[-0.5,0.9,8.0,2.8,4.5,a_true])
    #meas =  meas + randn(size(meas))*0.0005

    if angle
        funsim = (points,pars) -> fun(points,pars)
    else
        funsim = (points,pars) -> fun(points,[pars;a_fixed])
    end

    if angle
        lowb = [-4.0,-4.0, 0, -4.0, 0, 0]
        upb = [4.0, 4.0, 10.0, 4.0, 10.0, 2*pi]

    else
        lowb = [-4.0,-4.0, 0, -4.0, 0]
        upb = [4.0, 4.0, 10.0, 4.0, 10.0]
    end


    function log_p(x) 
        if any(x .< lowb) || any(x .> upb)
            return -Inf
        end

        return  -0.5/0.00001^2*( sum( (funsim(pts,x) - meas).^2 ))

    end

  
    out = smc_sample(log_p,10000,lowb,upb; K=300,NMCMC=200)
 
    return out,pts,meas,funsim,log_p

end



function test_without_src_radius(;angle=true,a_fixed=1.2)

    no_of_angles = 20
    src_radius = 4.5

    a_true = pi/3

    fun(pts,x) =  fancurves(pts, no_of_angles;src_shift_func = a->[0,x[1]],translation=[0,0.0],src_to_det_init=[0,1.0],det_axis_init = [1,x[2]],det_shift_func=a->[0,x[4]],apart=range(0+x[5],stop=2*pi+x[5],length=no_of_angles),det_radius=x[3],src_radius=src_radius,dpart=range(-sqrt(2), stop = sqrt(2), length = 100))

    Random.seed!(1)
    pts = rand(2,6)*0.5
    # Source shift, detector tilt, detector radius, detector shift,  angle
    meas = fun(pts,[-0.5,0.9,8.0,2.8,a_true])
    #meas =  meas + randn(size(meas))*0.01
    
    if angle
        funsim = (points,pars) -> fun(points,pars)
    else
        funsim = (points,pars) -> fun(points,[pars;a_fixed])
    end

    if angle
        lowb = [-4.0,-4.0, 0, -4.0, 0]
        upb = [4.0, 4.0, 10.0, 4.0, 2*pi]

    else
        lowb = [-4.0,-4.0, 0, -4.0]
        upb = [4.0, 4.0, 10.0, 4.0]
    end


    function log_p(x) 
        if any(x .< lowb) || any(x .> upb)
            return -Inf
        end


        return  -0.5/0.00001^2*( sum( (funsim(pts,x) - meas).^2 ))

    end


    out = smc_sample(log_p,10000,lowb,upb; K=300,NMCMC=200)
 
    return out,pts,meas,funsim,log_p
    

end



tfun = test_all # OR test_without_src_radius

out,pts,meas,func,logp = tfun(angle=false) # Whether to include the initial angle in the estimated parameters 
partW = out[1]
partX = out[2]
partESS = out[3]
partT = out[4]
matwrite(string(now()),Dict("W"=>partW, "X"=>partX, "ESS"=>partESS,"T"=>partT))

