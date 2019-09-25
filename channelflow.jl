
using SparseArrays, PyPlot

function orig_xy(xbar,ybar, A, B, H)
  y = ybar * H
  x = xbar * ((1.0-y)*B/2.0+y*(B/2.0+A))
  return  x, y
end

function assemblePoisson(n, f, Apara, B, H)
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
            if i == 1 
                # Dirichlet boundary condition, u = g
                push!(A, (row, row, -1.0))
                push!(A, (row, umap[i+1,j], 1.0)) 
                b[row] = 0.0
            elseif j == n+1
                push!(A, (row, row, 1.0))
                push!(A, (row, umap[i,j-1], -1.0)) 
                b[row] = 0.0
            elseif j == 1 || i == n+1
                # Dirichlet boundary condition, u = g
                push!(A, (row, row, 1.0))
                b[row] = 0.0
            else
                # Interior nodes, 5-point stencil
                push!(A, (row, row, 4.0))
                push!(A, (row, umap[i+1,j], -1.0))
                push!(A, (row, umap[i-1,j], -1.0))
                push!(A, (row, umap[i,j+1], -1.0))
                push!(A, (row, umap[i,j-1], -1.0))
                b[row] = 0.5*h^2
            end
        end
    end

    # Create CSC sparse matrix from matrix elements
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), N, N)

    return A, b, x, y
end


function channelflow(L, B, H, n)
    Apara =  sqrt((L-B)^2/4.0-H^2)
    f(x,y)=0.0   
    A, b, x, y = assemblePoisson(n, f, Apara, B, H)

    # Solve + reshape for plotting
    u = reshape(A \ b, n+1, n+1)

    h = 1.0 / n #on a grid of 1x1, figure out index spacing
    Qcap = 0.0    
    X=similar(u)
    Y=similar(u)
    for j = 1:n+1
        for i = 1:n+1 
           X[i,j],Y[i,j]=orig_xy(x[i],y[j], Apara, B, H)
        end
    end
    for j = 1:n+1
        for i = 1:n+1 
           x1,y1=orig_xy(x[i]+h/2,y[j]+h/2, Apara, B, H)
           x0,y0=orig_xy(x[i]-h/2,y[j]-h/2, Apara, B, H)
           Qcap += u[i,j]*abs(x1-x0)*abs(y1-y0)
        end
    end

    # Plotting
    clf()
    contour(X, Y, u, 10, colors="k")
    contourf(X, Y, u, 10)
    axis("equal")
    colorbar()
    
  return Qcap, x, y, u 
end

Q, x, y, u = channelflow(3.0, 0.5, 1.0, 20)

