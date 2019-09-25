
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

using SparseArrays
using LinearAlgebra

function waveguide_edges(p,t)
    elist, bedgeindex = all_edges(t)
    bnode = boundary_nodes(t)
    ein = []
    eout = []
    ewall = []
    for i = 1:size(bedgeindex)[1]
        if p[elist[bedgeindex[i],1],1] == 0 && p[elist[bedgeindex[i],2],1] == 0
            ein = [ein; bedgeindex[i]]
        elseif p[elist[bedgeindex[i],1],1] == 5 && p[elist[bedgeindex[i],2],1] == 5
            eout = [eout; bedgeindex[i]]
        else
            ewall = [ewall; bedgeindex[i]]
        end
            
    end
    ein = elist[ein,:]
    eout = elist[eout,:]
    ewall = elist[ewall,:]
    return ein, eout, ewall
    
end





using SymPy

function coeffmx(RHS, p, t, k)
    nm = p[t[k,1:3],:]
    LHS = zeros(3,3)
    for n = 1:3
        LHS[n,:] = [1 nm[n,1] nm[n,2]]
    end
    return (LHS\RHS)
end


function kentry(cmx, i, j, ak)
    return ak*(cmx[2,i]*cmx[2,j] + cmx[3,i]*cmx[3,j])
end

function mentry(cmx, i, j, ak, p, t, k)
    f(x,y) = (cmx[1,i] + cmx[2,i]*x + cmx[3,i]*y)*(cmx[1,j] + cmx[2,j]*x + cmx[3,j]*y)
    result = 0.0
    for n = 1:3
        base = (p[t[k,1],:] + p[t[k,2],:] + p[t[k,3],:])/6 + p[t[k,n],:]/2
        result = result + f(base[1],base[2])
    end
    result = (result*ak)/3
    return result
end

function find_bnd(e,t)
  ind=intersect(hcat(getindex.(findall(x -> x==e[1], t),1)),hcat(getindex.(findall(x -> x==e[2],t),1)))
  i=0
  j=0
  for k=1:3
    if t[ind[1],k]==e[1]
      i=k
    elseif t[ind[1],k]==e[2]
      j=k
    end
  end
  return ind[1],i,j
end

function boundsint(p, t, e)
    k, i, j = find_bnd(e, t)
    cmx = coeffmx(Matrix{Float64}(I, 3, 3), p, t, k)
    cross = 0
    f(x,y) = cmx[1,i]*cmx[1,j]*y + x*y*(cmx[1,j]*cmx[2,i] + cmx[1,i]*cmx[2,j]) + (y^2*(cmx[3,i]*cmx[1,j] + cmx[3,j]*cmx[1,i]))/2 + (y^2*x*(cmx[2,i]*cmx[3,j] + cmx[2,j]*cmx[3,i]))/2 + cmx[2,i]*cmx[2,j]*x^2*y + (cmx[3,i]*cmx[3,j]*y^3)/3  
    h(x,y) = cmx[1,i]*cmx[1,i]*y + x*y*(cmx[1,i]*cmx[2,i] + cmx[1,i]*cmx[2,i]) + (y^2*(cmx[3,i]*cmx[1,i] + cmx[3,i]*cmx[1,i]))/2 + (y^2*x*(cmx[2,i]*cmx[3,i] + cmx[2,i]*cmx[3,i]))/2 + cmx[2,i]*cmx[2,i]*x^2*y + (cmx[3,i]*cmx[3,i]*y^3)/3  
    g(x,y) = cmx[1,j]*cmx[1,j]*y + x*y*(cmx[1,j]*cmx[2,j] + cmx[1,j]*cmx[2,j]) + (y^2*(cmx[3,j]*cmx[1,j] + cmx[3,j]*cmx[1,j]))/2 + (y^2*x*(cmx[2,j]*cmx[3,j] + cmx[2,j]*cmx[3,j]))/2 + cmx[2,j]*cmx[2,j]*x^2*y + (cmx[3,j]*cmx[3,j]*y^3)/3  
    if p[t[k,i],2] > p[t[k,j],2]
        cross = f(p[t[k,i],1],p[t[k,i],2]) - f(p[t[k,i],1],p[t[k,j],2])
        id = h(p[t[k,i],1],p[t[k,i],2]) - h(p[t[k,i],1],p[t[k,j],2])
        jd = g(p[t[k,i],1],p[t[k,i],2]) - g(p[t[k,i],1],p[t[k,j],2])
    elseif p[t[k,i],2] < p[t[k,j],2]
        cross = f(p[t[k,i],1],p[t[k,j],2]) - f(p[t[k,i],1],p[t[k,i],2])
        id = h(p[t[k,i],1],p[t[k,j],2]) - h(p[t[k,i],1],p[t[k,i],2])
        jd = g(p[t[k,i],1],p[t[k,j],2]) - g(p[t[k,i],1],p[t[k,i],2])
    end
    return [id cross; cross jd]
end

function loadvec(p, t, e)
    k, i, j = find_bnd(e, t)
    cmx = coeffmx(Matrix{Float64}(I, 3, 3), p, t, k)
    
    f(x,y) = (cmx[1,i] + cmx[2,i]*x)*y + (cmx[3,i]*y^2)/2
    g(x,y) = (cmx[1,j] + cmx[2,j]*x)*y + (cmx[3,j]*y^2)/2
    
    if p[t[k,i],2] > p[t[k,j],2]
        id = f(p[t[k,i],1],p[t[k,i],2]) - f(p[t[k,i],1],p[t[k,j],2])
        jd = g(p[t[k,i],1],p[t[k,i],2]) - g(p[t[k,i],1],p[t[k,j],2])
    else
        id = f(p[t[k,i],1],p[t[k,j],2]) - f(p[t[k,i],1],p[t[k,i],2])
        jd = g(p[t[k,i],1],p[t[k,j],2]) - g(p[t[k,i],1],p[t[k,i],2])
    end
    
    return [id jd]
end
    

function femhelmholtz(p, t, ein, eout)
    areas = triarea(p,t)
    n = size(p)[1]
    K = spzeros(n,n); M = spzeros(n,n); Bin = spzeros(n,n); Bout = spzeros(n,n); bin = zeros(n);
    
    for kn = 1:size(t)[1]
        kinsert = zeros(3,3)
        minsert = zeros(3,3)
        cmx = coeffmx(Matrix{Float64}(I, 3, 3), p, t, kn)
        for i = 1:3, j = 1:3
            kinsert[i,j] = kentry(cmx,i,j,areas[kn])
            minsert[i,j] = mentry(cmx, i, j, areas[kn], p, t, kn)
        end
        K[t[kn,:],t[kn,:]] = K[t[kn,:],t[kn,:]]+kinsert
        M[t[kn,:],t[kn,:]] = M[t[kn,:],t[kn,:]]+minsert
    end
    
    for m = 1:size(ein)[1]
        einsert = boundsint(p,t,ein[m,:])
        Bin[ein[m,:],ein[m,:]] = Bin[ein[m,:],ein[m,:]] + einsert
    end
    
    for m = 1:size(eout)[1]
        einsert = boundsint(p,t,eout[m,:])
        Bout[eout[m,:],eout[m,:]] = Bout[eout[m,:],eout[m,:]] + einsert
    end
    
    for m = 1:size(ein)[1]
        binsert = loadvec(p,t,ein[m,:])
        bin[ein[m,:]] = bin[ein[m,:]] + binsert'
    end
    
    return K , M , Bin , Bout , bin
end
        
    

function helmholtz(pv, hmax, nref, k)
    p, t, e = pmesh(pv, hmax, nref)
    ein, eout, ewall = waveguide_edges(p,t)
    K, M, Bin, Bout, bin = femhelmholtz(p, t, ein, eout)
    A = K-k^2*M+k*im*(Bout + Bin)
    B = bin*2*k*im
    u = A\B
    ur = real.(u)
    "tplot(p,t,ur)"
    return u, ur, Bout 
end

function helmerrors(pv, hmax, nrefmax, k)
   pmax, tmax, emax = pmesh(pv, hmax, nrefmax)
    e=Base.MathConstants.e
    f(x,y) = e.^(-k*x*im)
   umax = f(pmax[:,1],1)
   errors = zeros(nrefmax)
   for ncur = 1:(nrefmax)
       "pprox, tprox, eprox = pmesh(pv, hmax, ncur)"
       uprox = helmholtz(pv, hmax, ncur, k)
       errorarray = maximum(abs.(umax[1:size(uprox)[1]] - uprox))
       #errors = [errors ; errorarray]
       errors[ncur]=errorarray
   end
   return errors
end

