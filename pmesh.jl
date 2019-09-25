using PyPlot, PyCall

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
function inpolygon(p::Array{Any,2}, pv::Array{Float64,2})
    path = pyimport("matplotlib.path")
    poly = path[:Path](pv)
    inside = [poly[:contains_point](p[ip,:]) for ip = 1:size(p,1)]
end



function div_poly(polys_xy0,hmax)
    x0=polys_xy0[1,1]
    y0=polys_xy0[1,2]
    ret = [x0 y0]
    for i = 2:size(polys_xy0)[1]
        #println(polys[i,:])
        x=polys_xy0[i,1]
        y=polys_xy0[i,2]
        len=sqrt((x-x0)^2+(y-y0)^2)
        n=ceil(len/hmax)
        #println("len=",len)
        #println("n=",n)
        #println([x0 y0]) 
        for j=1:n-1
          nx=(x0*(n-j)+x*j)/n   
          ny=(y0*(n-j)+y*j)/n   
          #println(nx,",",ny)
          ret = [ret; nx ny]
        end
        ret = [ret; x y]
        x0=x
        y0=y
    end 
    return ret
end

function centroids(points, tri)
    ret = reshape([],0,2)
    for i = 1:size(tri)[1]
        nx = (points[tri[i,1],1]+points[tri[i,2],1]+points[tri[i,3],1])/3
        ny = (points[tri[i,1],2]+points[tri[i,2],2]+points[tri[i,3],2])/3
        ret = [ret; nx ny] 
    end
    return ret
end

function area(points,tri)
   ret = []
   max = 0.0
   maxindex = -1
   for i = 1:size(tri)[1]
       na = (points[tri[i,1],1]*(points[tri[i,2],2]-points[tri[i,3],2]) 
           + points[tri[i,2],1]*(points[tri[i,3],2]-points[tri[i,1],2]) 
           + points[tri[i,3],1]*(points[tri[i,1],2]-points[tri[i,2],2]))/2
       ret = [ret; na]
       if na>max
           max=na
           maxindex=i
       end
   end
   return ret, maxindex, max
end

function circumcenter(p, t, it)
    ct = t[it,:]
    dp1 = p[ct[2], :] - p[ct[1], :]
    dp2 = p[ct[3], :] - p[ct[1], :]
    
    mid1 = (p[ct[2], :] + p[ct[1], :])/2
    mid2 = (p[ct[3], :] + p[ct[1], :])/2
    
    rhs = mid2-mid1
    s = [-dp1[2] dp2[2] ; dp1[1] -dp2[1]]\rhs
    cpc = mid1 + s[1] * [-dp1[2], dp1[1]]
    
    return cpc
end

function get_new_tri(inside,t)
    t_new=zeros(Int64,0,3)
    for i = 1:size(t)[1]
        if inside[i]
            t_new = [t_new;t[i,:]']
        end
    end
    return t_new
end

function pmesh(pv, hmax, nref)
    pvd = div_poly(pv,hmax)
    t = delaunay(pvd)
    
    centers = centroids(pvd,t)
    inside = inpolygon(centers,pv)
    t = get_new_tri(inside,t)
    areas, maxindex, max = area(pvd,t)
    
    while (max > 0.5*hmax.^2)
        added = circumcenter(pvd,t,maxindex)
        pvd = [pvd; added']
        t = delaunay(pvd)
        centers = centroids(pvd,t)
        
        inside = inpolygon(centers,pv)
        t = get_new_tri(inside,t)
        areas, maxindex, max = area(pvd,t)
    end
    
    for i = 1:nref
        edges = all_edges(t)[1]
        for n = 1:size(edges)[1]
            node = (pvd[edges[n,1],:] + pvd[edges[n,2],:])/2.0
            pvd = [pvd;node']
        end
        t = delaunay(pvd)
        centers = centroids(pvd,t)
        inside = inpolygon(centers,pv)
        t = get_new_tri(inside,t)
    end
    e = boundary_nodes(t)
    
    tplot(pvd,t)
    return pvd,t,e
end

