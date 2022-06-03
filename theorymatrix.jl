using RegionTrees
using StaticArrays
using LinearAlgebra
using SparseArrays
using VectorizedRoutines
using ProgressBars
using MAT
using ArgParse
using Optim


struct ray2d
    origin::Union{SVector{2,Float64},Vector{Float64}}
    dir::Union{SVector{2,Float64},Vector{Float64}}
    inv_direction::Union{SVector{2,Float64},Vector{Float64}}
    sign::Union{Array{Int64,1},SVector{2,Int64}}
end


@inline function dotitself(x)  
    return dot(x,x)
end


function logpisodiffgradi(x,args,cache;both=true)
    noisesigma = args.noisesigma
    scale = args.scale; y  = args.y; F = args.F
    bscale = args.bscale
    Dx = args.Dx; Dy = args.Dy
    Db = args.Db

    logp = 0.0

    res = cache.residual
    Dxprop = cache.Dxprop
    Dyprop = cache.Dyprop
    Dbprop = cache.Dbprop
    Fxprop = cache.Fxprop
    G = cache.gradiprop

    #Fxprop .= F*x
    mul!(Fxprop,F,x)
    storesubst!(res,Fxprop,y)
    #res .= Fxprop - y  
    #Dxprop .= Dx*x
    mul!(Dxprop,Dx,x)
    #Dyprop .= Dy*x
    mul!(Dyprop,Dy,x)
    #Dbprop .= Db*x
    mul!(Dbprop,Db,x)

    #G .= F'*(-((res)./noisesigma.^2))
    mul!(G,F',-((res)/noisesigma.^2))

    den = (scale^2 .+ Dyprop.^2 + Dxprop.^2)

    if both
        logp = -0.5/noisesigma^2*dotitself(res) -3/2*sum(log.(den)) - sum(log.(bscale^2 .+ Dbprop.^2))
    end

    #Gd1 =  Dx'*(-3.0*Lxx./(scale^2 .+ Lxx.^2));
    #G .= G  -3.0*(Dx)'*(Dxprop./(scale^2 .+ Dyprop.^2 + Dxprop.^2))  - 3.0*(Dy)'*(Dyprop./(scale^2 .+ Dyprop.^2 + Dxprop.^2))   -Db'*(2.0*Dbprop./(bscale^2 .+ Dbprop.^2));
    
    mul!(G,-3.0*(Dx)',(Dxprop./den),1,1)
    mul!(G,-3.0*(Dy)',(Dyprop./den),1,1)
    mul!(G,-Db',(2.0*Dbprop./(bscale^2 .+ Dbprop.^2)),1,1)

    return logp, G 
end

function  logpisodiff(x,args,cache)
    noisesigma = args.noisesigma
    scale = args.scale; y  = args.y; F = args.F
    bscale = args.bscale
    Dx = args.Dx; Dy = args.Dy
    Db = args.Db

    res = cache.residual
    Dxprop = cache.Dxprop
    Dyprop = cache.Dyprop
    Dbprop = cache.Dbprop
    Fxprop = cache.Fxprop

    mul!(Fxprop,F,x)
    #res .= Fxprop - y  
    storesubst!(res,Fxprop,y)
    #Dxprop .= Dx*x
    mul!(Dxprop,Dx,x)
    #Dyprop .= Dy*x
    mul!(Dyprop,Dy,x)
    #Dbprop .= Db*x
    mul!(Dbprop,Db,x)

    return   -0.5/noisesigma^2*dotitself(res)  - sum(log.(bscale^2 .+ Dbprop.^2)) -3/2*sum(log.(scale^2 .+ Dxprop.^2 + Dyprop.^2)) 

end

function regmatrices_first(dim)
    reg1d = spdiagm(Pair(0,-1*ones(dim))) + spdiagm(Pair(1,ones(dim-1))) + spdiagm(Pair(-dim+1,ones(1))) ;reg1d[dim,dim] = 0
    #reg1d = reg1d[1:dim-1,:]
    iden = I(dim)
    regx = kron(reg1d,iden)
    regy = kron(iden,reg1d)

    rmxix = sum(abs.(regx) ,dims=2) .< 2
    rmyix = sum(abs.(regy) ,dims=2) .< 2
    boundary = ((rmxix + rmyix)[:]) .!= 0
    q = findall(boundary .== 1)
    regx = regx[setdiff(1:dim^2,q), :] 
    regy = regy[setdiff(1:dim^2,q), :] 
    
    s = length(q)
    bmatrix = sparse(zeros(s,dim*dim))
    for i=1:s
        v = q[i]
        bmatrix[i,v] = 1
    end
    #bmatrix = bmatrix[i,i] .= 1

    return regx,regy,bmatrix
end

@inbounds @inline function storevector!(target,source)
    N = length(target)
    @assert N == length(source)
    @simd for i = 1:N
        target[i] = source[i]
    end
end

@inbounds @inline function storesubst!(target,a1,a2)
    N = length(target)
    @assert N == length(a1) == length(a2)
    @simd for i = 1:N
        target[i] = a1[i] - a2[i]
    end
end

@inbounds @inline function storeadd!(target,a1,a2)
    N = length(target)
    @assert N == length(a1) == length(a2)
    @simd for i = 1:N
        target[i] = a1[i] + a2[i]
    end
end


@inbounds @inline function slicemadd!(target,cols,vec,ix)
    N = length(ix)
    @assert N == length(vec)
    #mul!(target,cols[ix[1]],vec[1])
    for i = 1:N
        mul!(target,cols[ix[i]],vec[i],1,1)
    end
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


@inline @inbounds function intersects(
    r::ray2d,
    b::HyperRectangle{2,Float64},
)::Float64

    if (r.inv_direction[1] >= 0.0)
        #tmin = (parameters[1][1] - r.origin[1]) * r.inv_direction[1]
        tmin = (b.origin[1] - r.origin[1]) * r.inv_direction[1]
        #tmax = (parameters[2][1] - r.origin[1]) * r.inv_direction[1]
        tmax = (prevfloat(b.origin[1] + b.widths[1]) - r.origin[1]) * r.inv_direction[1]
    else
        #tmin = (parameters[2][1] - r.origin[1]) * r.inv_direction[1]
        tmin = ((b.origin[1] + b.widths[1]) - r.origin[1]) * r.inv_direction[1]
        #tmax = (parameters[1][1] - r.origin[1]) * r.inv_direction[1]
        tmax = (b.origin[1] - r.origin[1]) * r.inv_direction[1]
    end

    if (r.inv_direction[2] >= 0.0)
        #tymin = (parameters[1][2] - r.origin[2]) * r.inv_direction[2]
        tymin = (b.origin[2]  - r.origin[2]) * r.inv_direction[2]      
        #tymax = (parameters[2][2]- r.origin[2]) * r.inv_direction[2]
        tymax = (prevfloat(b.origin[2] + b.widths[2]) - r.origin[2]) * r.inv_direction[2]
    else
        #tymin = (parameters[2][2] - r.origin[2]) * r.inv_direction[2]
        tymin = ((b.origin[2] + b.widths[2]) - r.origin[2]) * r.inv_direction[2]      
        #tymax = (parameters[1][2] - r.origin[2]) * r.inv_direction[2]
        tymax = (b.origin[2]  - r.origin[2]) * r.inv_direction[2]
    end
    if ((tmin > tymax) || (tymin > tmax))
        return -1.0
    end
     if (tymin > tmin)
         tmin = tymin
     end
    if (tymax < tmax)
         tmax = tymax
    end

    if (isnan(tmin))
        #error("NaN detected")
       return b.widths[1]
    end

    if (isnan(tymin))
        #error("NaN detected")
       return  b.widths[2]
    end

    return norm(r.dir*(tmax-tmin))


end


@inline function intersects(
    a::HyperRectangle{N,Float64},
    b::HyperRectangle{N,Float64},
) where {N}
    maxA = a.origin + a.widths
    minA = a.origin
    maxB = b.origin + b.widths
    minB = b.origin
    return (all(maxA .>= minB) && all(maxB .> minA))

end


function inittree2d(n;lowx= -1.0, lowy = -1.0, wx = 2.0, wy = 2.0)
    empty::Array{Int64,1} = []
    oct = Cell(SVector(lowx, lowy), SVector(wx, wx), empty)
    for _ = 1:n
        for leaf in allleaves(oct)
            split!(leaf)
        end
    end
    for node in allcells(oct)
        node.data = copy(node.data)
    end
    return oct
end


function inserttotree!(
    box::HyperRectangle{N,Float64},
    tree::Cell,
    index::Int64,
) where {N}
    if (isleaf(tree))
        if (intersects(box, tree.boundary))
            push!(tree.data, index)
        end
    elseif (intersects(box, tree.boundary))
        for i = 1:length(tree.children)
            inserttotree!(box, tree[i], index)
        end
    end
    return nothing
end

function possiblepixels(r::ray2d, tree::Cell)::Vector{Int64}
    if (isleaf(tree) && intersects(r, tree.boundary) > 0.0)
        return tree.data

    elseif (~(tree.parent === nothing) && intersects(r, tree.boundary) > 0.0)
        N = length(tree.children)
        v = Vector{Int64}()
        for i = 1:N
            append!(v, possiblepixels(r, tree[i]))
        end
        return v

    elseif ((tree.parent === nothing) && intersects(r, tree.boundary) > 0.0)
        N = length(tree.children)
        v = Vector{Int64}()
        for i = 1:N
            append!(v, possiblepixels(r, tree[i]))
        end
        return collect(Set(v))

    else
        return []
    end

end


function pixelray(
    r::ray2d,
    vv::Vector{HyperRectangle{2,Float64}},
    checklist::Vector{Int64},
)
    N = length(checklist)
    indices = Vector{Int64}()
    tlens = Vector{Float64}()
    for i = 1:N
        q = intersects(r, vv[checklist[i]])
        if (q>0.0)
            push!(tlens, q)
            push!(indices, checklist[i])
        end
    end
    return (indices, tlens)
end

# function constructmatrix(tree::Cell, vv::Vector{HyperRectangle{2,Float64}}, Nrays::Int64, theta)
#     r = sqrt(2) # Radius of the transform.
#     Nproj = length(theta)
#     if (~isa(theta, Union{SArray,Array}))
#         theta = [theta]
#     end

#     span = range(r, stop = -r, length = Nrays)

#     Ntotal = Nrays * Nproj
#     rows = Vector{Int64}()
#     cols = Vector{Int64}()
#     vals = Vector{Float64}()
#     Nofpixels = length(vv)

#     Nth = Threads.nthreads()
#     rows = Vector{Vector{Int64}}(undef, Nth)
#     cols = Vector{Vector{Int64}}(undef, Nth)
#     vals = Vector{Vector{Float64}}(undef, Nth)
#     for p = 1:Nth
#         rows[p] = []
#         cols[p] = []
#         vals[p] = []
#     end

#     t = time()
#     if (Nproj > 1)
#         pb = ProgressBar(1:Nproj)
#     else
#         pb = 1:1
#     end
#     for a in pb
#         dir = [cos(theta[a]), sin(theta[a])]
#         aux = [-sin(theta[a]), cos(theta[a])]
#         Threads.@threads for i = 1:Nrays
#             rayindex = (a - 1) * Nrays + i
#             p1 = dir + aux * span[i]
#             p2 = p1 - 2 * dir * r
#             or = ray2d(p1, p2)
#             checklist = possiblepixels(or, tree) #Possible pixels.
#             indices, tlens = pixelray(or, vv, checklist)
#             Nel = length(indices)

#             append!(rows[Threads.threadid()],rayindex * ones(Int64, length(tlens)))
#             append!(cols[Threads.threadid()], indices)
#             append!(vals[Threads.threadid()], tlens)
  
#         end
#     end
#     rows = vcat(rows...)
#     cols = vcat(cols...)
#     vals = vcat(vals...)
#     M = sparse(rows, cols, vals, Ntotal, Nofpixels)
#     println("Matrix constructed in ", time() - t, " seconds.")
#     return M
# end



function constructmatrix(tree::Cell, vv::Vector{HyperRectangle{2,Float64}}, rays::Array{Array{ray2d,1},1}, Ny,Nx;columnmajor=true)
    Nproj = length(rays)
    for i = 1:Nproj-1
        @assert length(rays[i]) == length(rays[i+1])
    end
    Nrays = length(rays[1])
    Ntotal = Nrays * Nproj
    rows = Vector{Int64}()
    cols = Vector{Int64}()
    vals = Vector{Float64}()
    Nofpixels = Ny*Nx

    Nth = Threads.nthreads()
    rows = Vector{Vector{Int64}}(undef, Nth)
    cols = Vector{Vector{Int64}}(undef, Nth)
    vals = Vector{Vector{Float64}}(undef, Nth)
    for p = 1:Nth
        rows[p] = []
        cols[p] = []
        vals[p] = []
    end

    t = time()
    if (Nproj > 1)
        pb = ProgressBar(1:Nproj)
    else
        pb = 1:1
    end
    for a in pb
        Threads.@threads for i = 1:Nrays
            rayindex = (a - 1) * Nrays + i
            or = rays[a][i]            
            checklist = possiblepixels(or, tree) #Possible pixels.
            indices, tlens = pixelray(or, vv, checklist)
            Nel = length(indices)

            append!(rows[Threads.threadid()],rayindex * ones(Int64, length(tlens)))
            append!(cols[Threads.threadid()], indices)
            append!(vals[Threads.threadid()], tlens)
  
        end
        # if (a==Nproj)
        #     cols = vcat(cols...)
        #     ix = zeros(256^2,)
        #     ix[cols] .= 1
        #     figure()
        #     imshow(reshape(ix,256,256))
        #     error("")
        # end
    end
    rows = vcat(rows...); #rows=Int32.(rows)
    cols = vcat(cols...); #cols=Int32.(cols)
    if (!columnmajor)
        cols2 = similar(cols)
        ln = length(rows)
        for i=1:ln
            y = mod(cols[i]-1,Ny)+1
            x = div(cols[i]-1,Ny)+1
            cols2[i] = (y-1)*Nx + x
        end
        cols .= cols2
    end
    vals = vcat(vals...)
    M = sparse(rows, cols, vals, Ntotal, Nofpixels)
    println("Matrix constructed in ", time() - t, " seconds.")
    return M
end


function sub2ind(sizes, i::Int64, j::Int64, k::Int64)
    @assert i > 0 && i <= sizes[1]
    @assert j > 0 && j <= sizes[2]
    @assert k > 0 && k <= sizes[3]
    return (k - 1) * sizes[1] * sizes[2] + sizes[1] * (j - 1) + i
end


function setuppixels(sizex::Int64,sizey::Int64,octsize::Int64;lowy = -1.0, lowx = -1.0, widthy = 2, widthx = 2)
    @assert widthy == widthx
    oct = inittree2d(octsize;lowx= lowx, lowy = lowy, wx = widthx, wy = widthy)  
    size = [sizex, sizey]
    cellsize = [widthx,widthy] ./ size
    wds = [widthx,widthy]
    width = @SVector[cellsize[1], cellsize[2]]
    #l = Array([ [0.0, 0] [1, 0] [1, 1] [0, 1.0]]')
    #pm = mean(l, dims = 1)
    # for i = 1:2
    #     l[:, i] = cellsize[i] .* (l[:, i] .- pm[i]) .- cellsize[i] / 2.0 * size[i] .+ 0.5 * cellsize[i]
    # end
    l = [[lowx, lowy] [lowx + cellsize[1], lowy ] [lowx , lowy + cellsize[2]] [lowx + cellsize[1], lowy + cellsize[2]] ]'
    # println(l)
    # error("")
    pixelvector = Vector{HyperRectangle{2,Float64}}(undef, sizex * sizey)
    t = time()

    for j = 1:sizey
        for i = 1:sizex
            ll = l .+ reshape(cellsize .* [i - 1,  sizey - j + 1], 1, 2) # Flip Y-axis so that we have canonical Euclidean coordinate system.
            origin = @SVector[
                minimum(ll[:, 1]),
                minimum(ll[:, 2]),
            ]
            pixel = HyperRectangle(origin, width)
            index = j + (i - 1) * sizey 
            inserttotree!(pixel, oct, index)
            pixelvector[index] = pixel
        end
    end

    #println("Pixels initialized in ", time() - t, " seconds.")
    return (oct, pixelvector)
end

function constructparallelrays(Nrays,Nproj;rotations=range(-pi/2,stop=3/4*2*pi,length=Nproj),dete_plate_span=range(-sqrt(2), stop = sqrt(2), length = Nrays))

    @assert length(dete_plate_span) == Nrays
    @assert length(rotations) == Nproj
    rays = Vector{Vector{ray2d}}(undef,Nproj)

    # Parallel beam.

    for i = 1:Nproj
        span = dete_plate_span#range(r, stop = -r, length = Nrays)
        rays[i] = Vector{ray2d}(undef,Nrays)
        for j = 1:Nrays
            dir = [cos(rotations[i]), sin(rotations[i])]
            #aux = [-sin(rotations[i]), cos(rotations[i])]
            aux = [-sin(rotations[i]), cos(rotations[i])]*span[j] + center
            p1 =  aux #* span[j]
            #p2 = p1 - 2 * dir * r
            #ray = ray2d(p1, p2)
            ray = ray2ddir(p1,dir)
            rays[i][j] = ray
            
            #plot([ray.origin[1], ray.origin[1] +  ray.dir[1]],[ray.origin[2], ray.origin[2] +  ray.dir[2]] )
           # plot([rays[i][j].origin[1], rays[i][j].origin[1] +  rays[i][j].dir[1]],[rays[i][j].origin[2], rays[i][j].origin[2] +  rays[i][j].dir[2]] )

        end
    end

    return rays

end

function constructfanrays(Nrays,Nproj;src_shift_func = a->[0,0.0],translation=[0,0.0],src_to_det_init=[0,1],det_axis_init = nothing,det_shift_func=a->[0,0.0],apart=range(0,stop=pi,length=Nproj),det_radius=sqrt(2),src_radius=sqrt(2),dpart=range(-sqrt(2), stop = sqrt(2), length = Nrays))

    @assert length(dpart) == Nrays
    @assert length(apart) == Nproj
    rays = Vector{Vector{ray2d}}(undef,Nproj)

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

    for i = 1:Nproj

        rays[i] = Vector{ray2d}(undef,Nrays)
        forplots = zeros(2,2)

        src_tangent = [-sin(apart[i]), cos(apart[i])]
        src_orth = [cos(apart[i]), sin(apart[i])]
        srcshift = src_shift_func(apart[i])
        source = [cos(apart[i]), sin(apart[i])]*src_radius + srcshift[1]*src_orth + srcshift[2]*src_tangent     
        p2 = source + translation

        detcenter = -[cos(apart[i]), sin(apart[i])]*det_radius  
        det_tangent = [-sin(apart[i]), cos(apart[i])]
        det_orth = [cos(apart[i]), sin(apart[i])]
        detshift = det_shift_func(apart[i])
        totcenter = detcenter - detshift[1]*det_orth - detshift[2]*det_tangent

        for j = 1:Nrays                    
            aux = [-sin(apart[i]+dextra_rot), cos(apart[i]+dextra_rot)]*span[j] + totcenter
            p1 = aux + translation         
            ray = ray2d(p1, p2)
            rays[i][j] = ray

            # if(j==1)
            #     forplots[1,:] = p1
            # elseif(j==Nrays)
            #     forplots[2,:] = p1
            # end

            # plot([rays[i][j].origin[1], rays[i][j].origin[1] +  rays[i][j].dir[1]],[rays[i][j].origin[2], rays[i][j].origin[2] +  rays[i][j].dir[2]] )
            # scatter(aux[1],aux[2])
            
        end
        ## plot([rays[i][1].origin[1], rays[i][1].origin[1] +  rays[i][1].dir[1]],[rays[i][1].origin[2], rays[i][1].origin[2] +  rays[i][1].dir[2]] )
        ## plot([rays[i][end].origin[1], rays[i][end].origin[1] +  rays[i][end].dir[1]],[rays[i][end].origin[2], rays[i][end].origin[2] +  rays[i][end].dir[2]] ) 
        # plot(forplots[:,1],forplots[:,2])
        # println(forplots)
        # scatter(p2[1],p2[2])
        # xlim([-9,9])
        # ylim([-9,9])
        # axis(:equal)
     end
    #  error("")
    return rays
end

 ## When compared to ODL, the logic of the detector span is the same. The span vector should be increasing, so
    ## detector_partition = odl.uniform_partition(-8, 4, 100) equals dete_plate = range(-8, stop = 4, length = 100)

    ## However, with the coordinate axis are flipped in ODL. Depending on the case, π/2 radians plus a
    ## possbile angle shift must be added to obtain the same 
    ## rotational span due to the inverted coordinate system. 
    ## This is very important if the code is modified, but the current fanbeam function takes care of the axis differences.

    ## The vector src_to_det_init defines the shift to the initial angle:
    ## angle_partition = odl.uniform_partition(0,2*np.pi, 360) 
    ## equals rotations = range(0,stop=2*pi,length=360) .+ 2*pi*1/4+atan(src_to_det_init[2],src_to_det_init[1]) 
    ## Again, this extra angle is added due to the flipped coordinate axis between ODL and this code.

    ## ODL's parameter det_axis_init=[1,-0.5] refers to extra rotation of the detector plate. That is,
    ## det_axis_init=[3,-0.7] equals dextra_rot = atan(-0.7,3)
    ## if the src_to_det_init is left to its default value. 

    ## Source and detector radii have the same logic. 
    ## By default, the center of rotation in ODL is the origin. The vector translation
    ## moves it. This is just added to  initial points of the rays and the detector plate 
    ## position vectors.
    
    ## The function det_shift_func defines a shift for the detector at each projection angle
    ## in the parallel direction of the ray that goes through the COR and in the tangent direction of the ray.
    ## Function src_shift_func works also. It work in the same manner as det_shift_func, it just determines a shift for the source.
    ## Finally, having non-constant shift functions do not seem to behave properly in ODL. Perhaps there is a bug in ODL?

    ## Curved detector plates are not implemented.

    ## The domain of the object is odl_space and its correspondence to ODL should be clear. Since Astra toolbox and ODL do not seem to support
    ## anisotropic pixels together, the  number of pixels of the object and the geometry dimensions should be the same.



function realdatamatrix(;N=64,NPROJ=90,NRAYS=10,SIDE_2=200,DETECTOR_LENGTH_MM=500,SOURCE_RADIUS = 700,DETECTOR_RADIUS = 500,
    SRC_TO_DET_INIT_A1 = 0.0, SRC_TO_DET_INIT_A2 = 1.0, DET_AXIS_INIT_A1 = 1.0, DET_AXIS_INIT_A2 = 0.0, DET_SHIFT_A1 = 0.0, DET_SHIFT_A2 = 0.0, 
    SRC_SHIFT_A1 = 0.0, SRC_SHIFT_A2 = 0.0, INITANGLE = 0.0, COLUMNMAJOR = true
    )

    #close("all")  

    Nx = N; Ny = N;  # Number of X and Y pixels,
    Os = min(max(Int64(floor(log2(N)))-2,1),7) # Splitting factor of the quadtree. Does not affect the theory matrix, only the performance of building it. 6 seems optimal for 256x256.
    Nproj = NPROJ # Number of projections i.e. no. angles
    Nrays = NRAYS # Number of rays in a projection

    SIDE_2 = SIDE_2

    DETECTOR_LENGTH_PX = Nrays
    DETECTOR_LENGTH_MM = DETECTOR_LENGTH_MM
  
    SOURCE_RADIUS = SOURCE_RADIUS
    #SOURCE_DETECTOR_DIST = 1491.28   
    DETECTOR_RADIUS = DETECTOR_RADIUS

    odl_space = (min_pt=[-SIDE_2, -SIDE_2], max_pt=[SIDE_2, SIDE_2]) 
    src_to_det_init  = [SRC_TO_DET_INIT_A1,SRC_TO_DET_INIT_A2] # [0,1] is the default
    det_axis_init =   [DET_AXIS_INIT_A1,DET_AXIS_INIT_A2] #[1,0] is the default
    det_shift_func(angle) = [ DET_SHIFT_A1,   DET_SHIFT_A2]  
    src_shift_func(angle) = [SRC_SHIFT_A1,SRC_SHIFT_A2]
    rotations=  ((range(0,stop=2*pi,length=Nproj) .+  INITANGLE  )  ) 
    translation = [0,0.0]
    dete_radius = DETECTOR_RADIUS
    source_radius = SOURCE_RADIUS
    dete_plate = range(-DETECTOR_LENGTH_MM/2, stop = DETECTOR_LENGTH_MM/2, length = DETECTOR_LENGTH_PX)

    (qt,pixelvector)=setuppixels(Nx,Ny,Os;lowx=odl_space.min_pt[2],lowy=-odl_space.max_pt[1],widthx=abs(odl_space.max_pt[2]-odl_space.min_pt[2]),widthy=abs(odl_space.max_pt[1]-odl_space.min_pt[1]))  
    rays = constructfanrays(Nrays,Nproj;translation=translation, src_shift_func = src_shift_func, src_to_det_init = src_to_det_init, det_axis_init=det_axis_init, det_shift_func=det_shift_func,apart=rotations,det_radius=dete_radius,src_radius=source_radius, dpart=dete_plate)

    M = constructmatrix(qt, pixelvector, rays,Ny,Nx;columnmajor=COLUMNMAJOR)
    
    return M
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--N"
            help = "Number of pixels in one direction"
            arg_type = Int64
            default = 32        
        "--NPROJ"
            help = "Number of angles"
            arg_type = Int64
            default = 360
        "--NRAYS"
            help = "Number of pixels in the detector plate"
            arg_type = Int64
            default = 768
        "--BFGS_ITER"
            help = "Maximum number of of L-BFGS iterations"
            arg_type = Int64
            default = 150
        "--SIDE_2"
            help = "Half of the width and height of the square domain"
            arg_type = Float64
            default = 200.0
        "--DETECTOR_LENGTH_MM"
            help = "Detector length"
            arg_type = Float64
            default = 1154.2
        "--SOURCE_RADIUS"
            help = "Source radius"
            arg_type = Float64
            default = 859.46
        "--DETECTOR_RADIUS"
            help = "Detector radius"
            arg_type = Float64
            default = 7.14683839E+02
        "--SRC_TO_DET_INIT_A1"
            help = "Source to detector initial axis 1 as defined in ODL"
            arg_type = Float64
            default = 0.0
        "--SRC_TO_DET_INIT_A2"
            help = "Source to detector initial axis 2 as defined in ODL"
            arg_type = Float64
            default = 1.0   
        "--DET_AXIS_INIT_A1"
            help = "Detector initial axis 1 as defined in ODL"
            arg_type = Float64
            default = 1.0                     
        "--DET_AXIS_INIT_A2"
            help = "Detector initial axis 2 as defined in ODL"
            arg_type = Float64
            default = 0.0  
        "--DET_SHIFT_A1"
            help = "Detector shift axis 1 as defined in ODL"
            arg_type = Float64
            default = 0.0                     
        "--DET_SHIFT_A2"
            help = "Detector shift axis 2 as defined in ODL"
            arg_type = Float64
            default = 0.0  
        "--SRC_SHIFT_A1"
            help = "Source shift axis 1 as defined in ODL"
            arg_type = Float64
            default = 0.0                     
        "--SRC_SHIFT_A2"
            help = "Source shift axis 2 as defined in ODL"
            arg_type = Float64
            default = 0.0
        "--INITANGLE"
            help = "Initial angle in radians for the scans"
            arg_type = Float64
            default = 0.0 
        "--MATRIX_FILE"
            default = "radonmatrix.mat"
            help = "Filename for the radon matrix to be saved"
        "--MAP_FILE"
            default = "map_estimate.mat"
            help = "If MAP estimate (with isotropic Cauchy prior) is calculated, the MAP is saved in this file"
        "--MAP"
            default = false
            arg_type = Bool
            help = "Save MAP estimate (isotropic Cauchy prior) instead of theory matrix. Requires sinogram."
        "--SINO_FILE"
            default = "sinog.mat"
            help = "Sinogram file for MAP estimation."   
        "--MAP_PAR"
            default = 0.005
            arg_type = Float64
            help = "Parameter for the isotropic Cauchy prior"
        "--LIKELI_VAR"
            default = 0.3^2
            arg_type = Float64  
            help = "Variance of Gaussian likelihood" 
        "--OTHER_MATRIX"
            default = false
            arg_type = Bool
            help = "Use external theory matrix"
        "--OTHER_MATRIX_FILE"
            default = "odlmatrix.mat"
            help = "File of an external theory matrix"
        "--COLUMNMAJOR"
            default = true
            arg_type = Bool
            help = "If false, the theory matrix is constructed with C++ and Python logic (pixels in row-major order in the reconstruction space)"
        
    end

    return parse_args(s)
end


function mappi(M,sino,map_par,noisevar;Niter=150)
   
    y = copy((sino)[:]) 

    N = Int.(sqrt(size(M)[2]))
    Dx,Dy,Db = regmatrices_first(N)
    noisevar = noisevar    
    Niter = Niter
    scale = map_par # Cauchy dist. parameter for differences
    bscale = 0.1 # Cauchy dist. parameter for boundary values
    argi  = (Dx = Dx, Dy = Dy, F = M,noisesigma = sqrt(noisevar), scale = scale, y = y, bscale = bscale, Db = Db )
    cacheiso =(xprop=zeros(N*N),Fxprop=similar(y),Dxprop=zeros(size(Dx)[1]),Dyprop=zeros(size(Dy)[1]),Dbprop=zeros(size(Db)[1]), residual=similar(y),gradiprop=zeros(N*N))
    target1iso(x) = -logpisodiff(x,argi,cacheiso)
    target1isograd(x) = -logpisodiffgradi(x,argi,cacheiso;both=false)[2]
    res = Optim.optimize(target1iso, target1isograd, 0.0.+0.00001*randn(N*N,),Optim.LBFGS(), Optim.Options(allow_f_increases=true,show_trace=true,iterations=Niter); inplace = false)
    MAP = res.minimizer

    return reshape(MAP,(N,N))

end

function main(;manual=nothing)
    args = parse_commandline()
    if ( !isnothing(manual))        
        args = merge(args,manual)
    end
    N = args["N"]
    NRAYS = args["NRAYS"]
    NPROJ = args["NPROJ"]
    SIDE_2 = args["SIDE_2"]; 
    DETECTOR_LENGTH_MM = args["DETECTOR_LENGTH_MM"] 
    SOURCE_RADIUS = args["SOURCE_RADIUS"]
    DETECTOR_RADIUS = args["DETECTOR_RADIUS"]
    SRC_TO_DET_INIT_A1 = args["SRC_TO_DET_INIT_A1"]
    SRC_TO_DET_INIT_A2 = args["SRC_TO_DET_INIT_A2"]
    DET_AXIS_INIT_A1 = args["DET_AXIS_INIT_A1"]
    DET_AXIS_INIT_A2 = args["DET_AXIS_INIT_A2"]
    DET_SHIFT_A1 = args["DET_SHIFT_A1"]
    DET_SHIFT_A2 = args["DET_SHIFT_A2"]
    SRC_SHIFT_A1 = args["SRC_SHIFT_A1"]
    SRC_SHIFT_A2 = args["SRC_SHIFT_A2"]
    INITANGLE = args["INITANGLE"]
    MATRIX_FILE = args["MATRIX_FILE"]
    MAP_FILE = args["MAP_FILE"]
    MAP = args["MAP"]
    MAP_PAR = args["MAP_PAR"]
    LIKELI_VAR = args["LIKELI_VAR"]
    SINO_FILE = args["SINO_FILE"]
    OTHER_MATRIX = args["OTHER_MATRIX"]
    OTHER_MATRIX_FILE = args["OTHER_MATRIX_FILE"]
    COLUMNMAJOR = args["COLUMNMAJOR"]
    BFGS_ITER  = args["BFGS_ITER"]

    println(args)
    if (OTHER_MATRIX == false)
        M = realdatamatrix(;N=N,NPROJ=NPROJ,NRAYS=NRAYS,SIDE_2=SIDE_2,DETECTOR_LENGTH_MM=DETECTOR_LENGTH_MM,SOURCE_RADIUS=SOURCE_RADIUS,DETECTOR_RADIUS=DETECTOR_RADIUS,
        SRC_TO_DET_INIT_A1=SRC_TO_DET_INIT_A1,SRC_TO_DET_INIT_A2=SRC_TO_DET_INIT_A2,DET_AXIS_INIT_A1 = DET_AXIS_INIT_A1, DET_AXIS_INIT_A2 = DET_AXIS_INIT_A2, DET_SHIFT_A1 = DET_SHIFT_A1, DET_SHIFT_A2 = DET_SHIFT_A2,
        SRC_SHIFT_A1 = SRC_SHIFT_A1, SRC_SHIFT_A2 = SRC_SHIFT_A2, INITANGLE = INITANGLE, COLUMNMAJOR = COLUMNMAJOR
        )
        matwrite(string(@__DIR__) *"/"* MATRIX_FILE, Dict("radonmatrix"=>M))
    else
        M = matread(OTHER_MATRIX_FILE)["radonmatrix"]
    end

    if (MAP)
        sinogram = matread(SINO_FILE)["sino"]
        sinogram = sinogram'
        MAP_ESTIMATE = mappi(M,sinogram,MAP_PAR,LIKELI_VAR;Niter=BFGS_ITER)
        matwrite(string(@__DIR__) *"/"* MAP_FILE, Dict("map"=>MAP_ESTIMATE))
    end


    return M

end 


M=main()

# using PyPlot
# using Images
# params = Dict{String, Any}("COLUMNMAJOR" =>false, "SRC_TO_DET_INIT_A1" => 0.0, "SINO_FILE" => "sinog.mat", "NRAYS" => 768, "N" => 100, "DETECTOR_LENGTH_MM" => 1154.2, "DETECTOR_RADIUS" => 714.683839, "SOURCE_RADIUS" => 859.46, "OTHER_MATRIX" => false, "SIDE_2" => 200.0, "LIKELI_VAR" => 0.09, "SRC_TO_DET_INIT_A2" => 1.0, "MAP_PAR" => 0.005, "SRC_SHIFT_A2" => 0*319.943788, "DET_AXIS_INIT_A1" => 1.0, "OTHER_MATRIX_FILE" => "odlmatrix30.mat", "INITANGLE" => 0*2.55023549, "NPROJ" => 30, "MATRIX_FILE" => "64.mat", "DET_SHIFT_A2" => 0*43.6514375, "MAP" => false, "SRC_SHIFT_A1" => 0.0, "DET_AXIS_INIT_A2" => 0*0.280799451, "DET_SHIFT_A1" => 0.0)

# M = main(manual=params)

# M2 = matread("odlmatrix30.mat")["radonmatrix"]

# obj = shepp_logan(100);  ob = obj[:]
# obj2 = obj'; ob2= obj2[:] # Fortran and C++ row/column order issue!

# sino1 = M*ob; sino1 = reshape(sino1,(769,30)); #sino1 = reverse(sino1,dims=1)
# sino2 = M2*ob2; sino2 = reshape(sino2,(769,30))
# close("all")
# imshow(sino1)
# axis("auto")
# figure()
# imshow(sino2)
# axis("auto")
