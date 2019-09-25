# Solve Poisson's equation -(uxx + uyy) = f, bnd cnds u(x,y) = g(x,y)
# on a square grid using the finite difference method.
#
# UC Berkeley Math 228B, Per-Olof Persson <persson@berkeley.edu>

using Pkg
Pkg.add("Laplacians")
using SparseArrays, PyPlot, Laplacians

"""
    A, b, x, y = assemblePoisson(n, f, g)

Assemble linear system Au = b for Poisson's equation using finite differences.
Grid size (n+1) x (n+1), right hand side function f(x,y), Dirichlet boundary
conditions g(x,y).
"""

function D2x(f,x,y,h) 
  return (f(x-h,y)-2*f(x,y)+f(x+h,y))/h^2
end
function D2y(f,x,y,h) 
  return (f(x,y-h)-2*f(x,y)+f(x,y+h))/h^2
end

function z(f,x,y,h)
  return f(x,y) + h^2*(D2x(f,x,y,h)+D2y(f,x,y,h))/12
end

function assemblePoisson(n, f, g)
    h = 1.0 / n #on a grid of 1x1, figure out index spacing
    N = (n+1)^2 #size of all elements, including boundary conditions
    x = h * (0:n) #x coords of the index points
    y = x #equate 2 axis

    umap = reshape(1:N, n+1, n+1)     # Index mapping from 2D grid to vector
    A = Tuple{Int64,Int64,Float64}[]  # Array of matrix elements (row,col,value)
    b = zeros(N)
    

    # Main loop, insert stencil in matrix for each node point
    for j = 1:n+1
        for i = 1:n+1
            row = umap[i,j]
            if i == 1 || i == n+1 || j == 1 || j == n+1
                # Dirichlet boundary condition, u = g
                push!(A, (row, row, 1.0))
                b[row] = g(x[i],y[j])
            else
                # Interior nodes, 5-point stencil
                push!(A, (row, row, 20.0))
                push!(A, (row, umap[i+1,j], -4.0))
                push!(A, (row, umap[i-1,j], -4.0))
                push!(A, (row, umap[i,j+1], -4.0))
                push!(A, (row, umap[i,j-1], -4.0))
                
                push!(A, (row, umap[i+1,j+1], -1.0))
                push!(A, (row, umap[i-1,j+1], -1.0))
                push!(A, (row, umap[i+1,j-1], -1.0))
                push!(A, (row, umap[i-1,j-1], -1.0))
                b[row] = z(f, x[i], y[j], h) * 6*h^2
            end
        end
    end

    # Create CSC sparse matrix from matrix elements
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), N, N)

    return A, b, x, y
end

function testPoisson(n=40)
    uexact(x,y) = exp(-(4(x - 0.3)^2 + 9(y - 0.6)^2)) #truesoln
    f(x,y) = uexact(x,y) * (26 - (18y - 10.8)^2 - (8x - 2.4)^2) #f_ij
    
    A, b, x, y = assemblePoisson(n, f, uexact)

    # Solve + reshape for plotting
    u = reshape(A \ b, n+1, n+1)

    # Plotting
    clf()
    contour(x, y, u, 10, colors="k")
    contourf(x, y, u, 10)
    axis("equal")
    colorbar()
    
    # Compute error in max-norm
    u0 = uexact.(x, y') #solving for lte
    error = maximum(abs.(u - u0))
end
