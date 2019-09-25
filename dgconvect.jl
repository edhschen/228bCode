using LinearAlgebra
using PyPlot

function gauss_quad(p)
    n = ceil((p+1)/2)
    b = 1:n-1
    b = @. b / sqrt(4*b^2 - 1)
    eval, evec = eigen(diagm(1 => b, -1 => b))
    return eval, 2*evec[1,:].^2
end

function legendre_poly(x, p)
    z = zeros(size(x))
    o = ones(size(x))
    y = hcat(o, x, repeat(z, 1, p-1))
    dy = hcat(z, o, repeat(z, 1, p-1))
    for i = 1:p-1
        @. y[:,i+2] = ((2i+1)*x*y[:,i+1] - i*y[:,i]) / (i+1)
        @. dy[:,i+2] = ((2i+1)*(x*dy[:,i+1] + y[:,i+1]) - i*dy[:,i]) / (i+1)
    end
    y, dy
end

function dgconvect(; n=10, p=1, T=1.0, dt=1e-3)
    # Discretization
    h = 1 / n
    s = @.cos((pi*collect(0:p))/p)
    s1 = 0.5*(-s.+1)
    x = @. h*s1 + (0:h:1-h)'

    # Gaussian initial condition (and exact solution if shifted)
    uinit(x) = exp(-(x - 0.5)^2 / 0.1^2)

    # Coefficient Matrix
    lhsc = legendre_poly(-s,p)[1]
    cmx = lhsc \ Matrix{Float64}(I,p+1,p+1)

    # Mass and Stiffness Matrices
   
    Mel = zeros(p+1,p+1)
    Kel = zeros(p+1,p+1)
    gx, gw = gauss_quad(2p)
    #gx = (h/2).*(gx.+1)
    for i = 0:p, j = 0:p
	    mdvec = zeros(p+1)
	    kdvec = zeros(p+1)
	    for k = 0:p
	         lpval, dlpval = legendre_poly([gx[k+1]], p)
		 mdvec[k+1] = dot(lpval,cmx[:,i+1]) * dot(lpval,cmx[:,j+1])
		 kdvec[k+1] = dot(dlpval,cmx[:,i+1]) * dot(lpval,cmx[:,j+1])
	    end
	    Mel[i+1,j+1] = dot(mdvec, gw)
	    Kel[i+1,j+1] = dot(kdvec, gw)
    end

    Mel = Mel.*(h/2)
    Kel = Kel

    # RHS function in semi-discrete system
    function rhs(u)
        r = Kel * u
        r[end,:] = r[end,:] - u[end,:]
        r[1,:] = r[1,:] + u[end, [end; 1:end-1]]
        r = Mel \ r
    end

    # Setup
    u = uinit.(x)
    nsteps = round(Int, T/dt)

    # Setup plotting
    xx = collect(0:0.01:1) # For exact solution
    clf(); axis([0, 1, -0.1, 1.1]); grid(true);
    ph1 = plot(x, u, "b")   # Many line object (since discontinuous)
    ph2, = plot(xx, uinit.(xx), "k") # Single line object

    # Main loop
    for it = 1:nsteps
        # Runge-Kutta 4
        k1 = dt * rhs(u)
        k2 = dt * rhs(u + k1/2)
        k3 = dt * rhs(u + k2/2)
        k4 = dt * rhs(u + k3)
        u += (k1 + 2*k2 + 2*k3 + k4) / 6

        # Plotting
        if mod(it, round(nsteps/100)) == 0
            uexact = @. uinit(mod(xx - dt*it, 1.0))
            for line in ph1 line.remove(); end
            ph1 = plot(x, u, "b")
            ph2.set_data(xx, uexact)
            pause(1e-3)
        end
    end
    ua = uinit.(x)
    xw = zeros(size(x)[1],size(x)[2])
    for j = 1:p
	xw[p-j+2,:] = x[p-j+2,:] - x[p-j+1,:]
    end
    eu = (sum((u.^2).*xw))^0.5
    eua	= (sum((ua.^2).*xw))^0.5
    return u, abs(eua-eu) 
end


function dgconvect_convergence()
    errors = zeros(5,5)
    for i = 0:4, j = 4:8
        p1 = ceil(Int,2^i)
        n1 = ceil(Int,(2^j)/p1)
        println(p1, " ", n1)
	errors[j-3,i+1] = dgconvect( n=n1, p=p1, dt=2e-4)[2]	
    end
    clf()
    loglog([16 32 64 128 256]', errors)
    rates = @. log2(errors[end-1,:]) - log2(errors[end,:])
    return errors, rates
end

function dgconvdiff(; n=10, p=1, T=1.0, dt=1e-3, k=1e-3)
    # Discretization
    h = 1 / n
    s = @.cos((pi*collect(0:p))/p)
    s1 = 0.5*(-s.+1)
    x = @. h*s1 + (0:h:1-h)'

    # Gaussian initial condition (and exact solution if shifted)
    uinit(x) = exp(-(x - 0.5)^2 / 0.1^2)

    # Coefficient Matrix
    lhsc = legendre_poly(-s,p)[1]
    cmx = lhsc \ Matrix{Float64}(I,p+1,p+1)

    # Mass and Stiffness Matrices
   
    Mel = zeros(p+1,p+1)
    Kel = zeros(p+1,p+1)
    gx, gw = gauss_quad(2p)
    #gx = (h/2).*(gx.+1)
    for i = 0:p, j = 0:p
	    mdvec = zeros(p+1)
	    kdvec = zeros(p+1)
	    for k = 0:p
	         lpval, dlpval = legendre_poly([gx[k+1]], p)
		 mdvec[k+1] = dot(lpval,cmx[:,i+1]) * dot(lpval,cmx[:,j+1])
		 kdvec[k+1] = dot(dlpval,cmx[:,i+1]) * dot(lpval,cmx[:,j+1])
	    end
	    Mel[i+1,j+1] = dot(mdvec, gw)
	    Kel[i+1,j+1] = dot(kdvec, gw)
    end

    Mel = Mel.*(h/2)
    Kel = Kel

    u = uinit.(x)
    us = uinit.(x)

    # RHS function in semi-discrete system
    function rhs(u)
        up = vcat(u[end,:]' , u[1:end-1,:])
        un = vcat(u[2:end,:] , u[1,:]')
        xrev = abs.(x.-vcat(x[2,:]' , x[1:end-1,:]))
        u2 = (2u - un - up)./(2xrev)
        
        
        r = -Kel * (u)
        r[end,:] = r[end,:] + u[end,:]
        r[1,:] = r[1,:] - u[end, [end; 1:end-1]]
        r = Mel \ r
        sigma = r
        r = Kel * (u - k*(r)) 
        r[end,:] = r[end,:] - u[end,:] + k*sigma[end,:]
        r[1,:] = r[1,:] + u[end, [end; 1:end-1]] - k*sigma[end, [end; 1:end-1]]
        r = Mel \ r
    end

    # Setup
    
    nsteps = round(Int, T/dt)

    # Setup plotting
    xx = collect(0:0.01:1) # For exact solution
    uex(x,t,i,k) = exp(-100((x-t-0.5+i)^2/(1+400*k*t)))/sqrt(1+400*k*t)
    clf(); axis([0, 1, -0.1, 1.1]); grid(true);
    ph1 = plot(x, u, "b")   # Many line object (since discontinuous)
    ph2, = plot(xx, uinit.(xx), "k") # Single line object

    # Main loop
    for it = 1:nsteps
        # Runge-Kutta 4
        k1 = dt * rhs(u)
        k2 = dt * rhs(u + k1/2)
        k3 = dt * rhs(u + k2/2)
        k4 = dt * rhs(u + k3)
        u += (k1 + 2*k2 + 2*k3 + k4) / 6

        # Plotting
        if mod(it, round(nsteps/100)) == 0
            uexact = @. uex(xx,dt*it,0,k)+uex(xx,dt*it,1,k)+uex(xx,dt*it,-1,k)  
            for line in ph1 line.remove(); end
            ph1 = plot(x, u, "b")
            ph2.set_data(xx, uexact)
            pause(1e-3)
        end
    end
    ua = @. uex(x,1,0,k)+uex(x,1,1,k) + uex(x,1,-1,k)  
    xw = zeros(size(x)[1],size(x)[2])
    for j = 1:p
	xw[p-j+2,:] = x[p-j+2,:] - x[p-j+1,:]
    end
    eu = (sum((u.^2).*xw))^0.5
    eua	= (sum((ua.^2).*xw))^0.5
    return u, abs(eua-eu) 
end


function dgconvdiff_convergence()
    errors = zeros(4,5)
    for i = 0:4, j = 4:7
        p1 = ceil(Int,2^i)
        n1 = ceil(Int,(2^j)/p1)
        println(p1, " ", n1)
	errors[j-3,i+1] = dgconvdiff( n=n1, p=p1, dt=1e-3, k = 1e-3)[2]	
    end
    clf()
    loglog([16 32 64 128]', errors)
    rates = @. log2(errors[end-1,:]) - log2(errors[end,:])
    return errors, rates
end
