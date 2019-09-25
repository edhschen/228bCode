
using PyPlot, PyCall, IJulia

"""
    t = delaunay(p)

Delaunay triangulation `t` of N x 2 node array `p`.
"""
function delaunay(p)
    tri = pyimport("matplotlib.tri")
    t = tri[:Triangulation](p[:,1], p[:,2])
    return Int64.(t[:triangles] .+ 1)
end

"""
    edges, boundary_indices = all_edges(t)

Find all unique edges in the triangulation `t` (ne x 2 array)
Second output is indices to the boundary edges.
"""
function all_edges(t)
    etag = vcat(t[:,[1,2]], t[:,[2,3]], t[:,[3,1]])
    etag = hcat(sort(etag, dims=2), 1:3*size(t,1))
    etag = sortslices(etag, dims=1)
    dup = all(etag[2:end,1:2] - etag[1:end-1,1:2] .== 0, dims=2)[:]
    keep = .![false;dup]
    edges = etag[keep,1:2]
    emap = cumsum(keep)
    invpermute!(emap, etag[:,3])
    emap = reshape(emap,:,3)
    dup = [dup;false]
    dup = dup[keep]
    bndix = findall(.!dup)
    return edges, bndix, emap
end

"""
    e = boundary_nodes(t)

Find all boundary nodes in the triangulation `t`.
"""
function boundary_nodes(t)
    edges, boundary_indices = all_edges(t)
    return unique(edges[boundary_indices,:][:])
end

"""
    tplot(p, t, u=nothing)

If `u` == nothing: Plot triangular mesh with nodes `p` and triangles `t`.
If `u` == solution vector: Plot filled contour color plot of solution `u`.
"""
function tplot(p, t, u=nothing)
    clf()
    axis("equal")
    if u == nothing
        tripcolor(p[:,1], p[:,2], t .- 1, 0*t[:,1],
                  cmap="Set3", edgecolors="k", linewidth=1)
    else
        tricontourf(p[:,1], p[:,2], t .- 1, u, 20)
    end
    draw()
end

"""
    inside = inpolygon(p, pv)

Determine if each point in the N x 2 node array `p` is inside the polygon
described by the NE x 2 node array `pv`.
"""
function inpolygon(p::Array{Float64,2}, pv::Array{Float64,2})
    path = pyimport("matplotlib.path")
    poly = path[:Path](pv)
    inside = [poly[:contains_point](p[ip,:]) for ip = 1:size(p,1)]
end


function remove_outside_tris(p, t, pv)
    pmid = dropdims(sum(p[t,:], dims=2), dims=2) / 3
    is_inside = inpolygon(pmid, pv)
    t = t[is_inside,:]
end

function triarea(p, t)
    d12 = @. p[t[:,2],:] - p[t[:,1],:]
    d13 = @. p[t[:,3],:] - p[t[:,1],:]
    @. abs(d12[:,1] * d13[:,2] - d12[:,2] * d13[:,1]) / 2
end

function remove_tiny_tris(p, t)
    t = t[triarea(p,t) .> 1e-14,:]
end

function circumcenter(p)
    dp1 = @. p[2,:] - p[1,:]
    dp2 = @. p[3,:] - p[1,:]

    mid1 = @. (p[1,:] + p[2,:]) / 2
    mid2 = @. (p[1,:] + p[3,:]) / 2

    s = [ -dp1[2] dp2[2]; dp1[1] -dp2[1]] \ (-mid1 .+ mid2)
    pc = @. mid1' + s[1] * [-dp1[2] dp1[1]]
end

function edge_midpoints(p, t)
    pmid = reshape(p[t,:] + p[t[:,[2,3,1]],:], :, 2) / 2
    pmid = unique(pmid, dims=1)
end

function pmesh(pv, hmax, nref)
    p = zeros(Float64, 0, 2)
    for i = 1:size(pv,1) - 1
        pp = pv[i:i+1,:]
        L = sqrt(sum(diff(pp, dims=1).^2, dims=2))[1]
        if L > hmax
            n = ceil(Int, L / hmax)
            ss = (0:n) / n
            pp = [1 .- ss ss] * pp
        end
        p = [p; pp[1:end-1,:]]
    end

    t = zeros(Int64, 0, 3)
    while true
        t = delaunay(p)
        t = remove_tiny_tris(p, t)
        t = remove_outside_tris(p, t, pv)
        # tplot(p,t), pause(1e-3)
        
        area = triarea(p, t)
        maxarea, ix = findmax(area)
        if maxarea < hmax^2 / 2
            break
        end
        pc = circumcenter(p[t[ix,:],:])
        p = [p; pc]
    end

    for iref = 1:nref
        p = [p; edge_midpoints(p, t)]
        t = delaunay(p)
        t = remove_tiny_tris(p, t)
        t = remove_outside_tris(p, t, pv)
        # tplot(p, t), pause(1e-3)
    end
    
    e = boundary_nodes(t)
    p, t, e
end

function pmesh_test()
    pv = [0 0; 1 0; .5 .5; 1 1; 0 1; 0 0]
    p,t,e = pmesh(pv, 0.2, 1);
    tplot(p,t)
end

using SparseArrays
using LinearAlgebra
function coefficients(RHS, p, t, k)
    LHS = [1 p[t[k,1],1] p[t[k,1],2]; 1 p[t[k,2],1] p[t[k,2],2]; 1 p[t[k,3],1] p[t[k,3],2]]
    coeff = LHS\RHS'
    return coeff'
end

function element(coeffmatrix, a, b)
    return coeffmatrix[a,2]*coeffmatrix[b,2] + coeffmatrix[a,3]*coeffmatrix[b,3]
end

function fempoi(p, t, e)
    areas = triarea(p,t)
    n = size(p)[1]
    A = spzeros(n, n); b = zeros(n);
    for k = 1:size(t)[1]
        cm = [coefficients([1 0 0] , p, t, k) ; 
              coefficients([0 1 0] , p, t, k) ; 
              coefficients([0 0 1] , p, t, k) ]
        insert = [element(cm, 1, 1) element(cm, 1, 2) element(cm, 1, 3); 
                  element(cm, 2, 1) element(cm, 2, 2) element(cm, 2, 3); 
                  element(cm, 3, 1) element(cm, 3, 2) element(cm, 3, 3)]
        insert *=  areas[k]

        A[t[k,:],t[k,:]] = A[t[k,:],t[k,:]]+insert
        #aa = areas[k]/3.0
        b[t[k,:]] = b[t[k,:]].+ areas[k]/3.0
    end
    for k = 1:size(e)[1]
        i=e[k]
        A[i,:].=0.0
        A[i,i]=1.0
        b[i]=0.0
    end
    dropzeros!(A)
    println(b)
    println(A)

    return A \ b
end

function poiconv(pv, hmax, nrefmax)
   pmax, tmax, emax = pmesh(pv, hmax, nrefmax)
   umax = fempoi(pmax, tmax, emax)
   errors = zeros(nrefmax)
   for ncur = 0:(nrefmax-1)
       pprox, tprox, eprox = pmesh(pv, hmax, ncur)
       uprox = fempoi(pprox, tprox, eprox)
       errorarray = maximum(abs.(umax[1:size(uprox)[1]] - uprox))
       #errors = [errors ; errorarray]
       errors[ncur+1]=errorarray
   end

   return errors
end

function p2mesh(p,t)
    "npoints = edge_midpoints(p,t)"
    "p2 = [p; npoints]"
    "e2 = zeroes()"
    m = size(p)[1]
    
    elist, bedges, emap = all_edges(t)
    mid_point=Dict()
    for k = 1:size(elist)[1]   
        x=(p[elist[k,1],1]+p[elist[k,2],1])/2.0
        y=(p[elist[k,1],2]+p[elist[k,2],2])/2.0
        p=[p; x y]
        mid_point[elist[k,2],elist[k,1]]=size(p)[1]
        mid_point[elist[k,1],elist[k,2]]=size(p)[1]
    end
    t2=zeros(Int64,0,6)
    for i = 1:size(t)[1]
       t12=mid_point[t[i,1],t[i,2]]
       t23=mid_point[t[i,2],t[i,3]] 
       t31=mid_point[t[i,3],t[i,1]] 
       t2=[t2; t[i,:]' t12 t23 t31]  
    end

    e2 = boundary_nodes(t)
    for i=1:size(bedges)[1]
       e=elist[bedges[i],:]
       md=mid_point[e[1],e[2]]
       e2=vcat(e2,md)
    end 
    return p,t2,e2
end


using SymPy



function gquad(k, i, j, cmx, p2, t2, ak)
    f(x,y) = 4x^2*(cmx[5,i]*cmx[5,j]) + 2x*y*(cmx[5,i]*cmx[4,j] + cmx[5,j]*cmx[4,i]) + y^2*(cmx[4,i]*cmx[4,j]) + 2x*(cmx[5,i]*cmx[2,j] + cmx[5,j]*cmx[2,i]) + y*(cmx[2,i]*cmx[4,j] + cmx[2,j]*cmx[4,i]) + cmx[2,i]*cmx[2,j]                     
    g(x,y) = 4y^2*(cmx[6,i]*cmx[6,j]) + 2x*y*(cmx[6,i]*cmx[4,j] + cmx[6,j]*cmx[4,i]) + x^2*(cmx[4,i]*cmx[4,j]) + 2y*(cmx[6,i]*cmx[3,j] + cmx[6,j]*cmx[3,i]) + x*(cmx[3,i]*cmx[4,j] + cmx[3,j]*cmx[4,i]) + cmx[3,i]*cmx[3,j] 
    
    result = 0
    for n = 1:3
	base = (p2[t2[k,1],:] + p2[t2[k,2],:] + p2[t2[k,3],:])/6
        num = base + p2[t2[k,n],:]/2
        result = result + f(num[1],num[2]) + g(num[1],num[2])
    end
    result = (result*ak)/3
    return result
end

function gquad2(k, i, cmx, p2, t2, ak)
    h(x,y) = cmx[1,i] + x*cmx[2,i] + y*cmx[3,i] + x*y*cmx[4,i] + x^2*cmx[5,i] + y^2*cmx[6,i]
    result = 0
    for n = 1:3
	base = (p2[t2[k,1],:] + p2[t2[k,2],:] + p2[t2[k,3],:])/6
        num = base + p2[t2[k,n],:]/2
        result = result + h(num[1],num[2])
    end
    result = (result*ak)/3
    return result
end
    
function coeff2(RHS, p2, t2, k)
    nm = p2[t2[k,1:6],:]
    LHS = zeros(6,6)
    for n = 1:6
       LHS[n,:] = [1 nm[n,1] nm[n,2] nm[n,1]*nm[n,2] nm[n,1]^2 nm[n,2]^2] 
    end
    return (LHS\RHS), LHS
end

function fempoi2(p2, t2, e2)
    areas = triarea(p2,t2)
    n = size(p2)[1]
    A = spzeros(n,n); b = zeros(n);
    
    for k = 1:size(t2)[1]
        insert = zeros(6,6)
        insert2 = zeros(6)
        cmx, lhss = coeff2(Matrix{Float64}(I, 6, 6), p2, t2, k)
        for i = 1:6
            for j = 1:6
                insert[i,j] = gquad(k,i,j,cmx,p2,t2,areas[k])
            end
            insert2[i] = gquad2(k,i,cmx,p2,t2,areas[k])
        end
        A[t2[k,:],t2[k,:]] = A[t2[k,:],t2[k,:]]+insert
        b[t2[k,:]] = b[t2[k,:]]+ insert2
    end
    for k = 1:size(e2)[1]
        i=e2[k]
        A[i,:].=0.0
        A[i,i]=1.0
        b[i]=0.0
    end
    
    return A\b
    
end


function convtest(pv, hmax, nrefmax)
    p, t, e = pmesh(pv,hmax,nrefmax)
    p2, t2, e2 = p2mesh(p,t)
    umax = fempoi2(p2,t2,e2)
    errors = zeros(nrefmax)
    for ncur = 0:(nrefmax-1)
       pprox, tprox, eprox = pmesh(pv, hmax, ncur)
       pprox2, tprox2, eprox2 = p2mesh(pprox, tprox)
       uprox = fempoi2(pprox2, tprox2, eprox2)
       errorarray = maximum(abs.(umax[1:size(uprox)[1]] - uprox))
       #errors = [errors ; errorarray]
       errors[ncur+1]=errorarray
    end

    return errors
end

"errors = convtest(pv,hmax,nrefmax)

clf()
loglog(hmax ./ [1,2,4,8], errors)
rates = @. log2(errors[end-1,:]) - log2(errors[end,:])

@. log2(errors[end-1,:])"


