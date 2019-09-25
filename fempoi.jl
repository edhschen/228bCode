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
    edges = vcat(t[:,[1,2]], t[:,[2,3]], t[:,[3,1]])
    edges = sortslices(sort(edges, dims=2), dims=1)
    dup = all(edges[2:end,:] - edges[1:end-1,:] .== 0, dims=2)[:]
    keep = .![false;dup]
    edges = edges[keep,:]
    dup = [dup;false]
    dup = dup[keep]
    return edges, findall(.!dup)
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

