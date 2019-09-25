using PyPlot

function traffic(uinit="step", method="godunov", m=400; T=2, kmul=0.8)
    # Discretization
    h = 4 / m
    x = ((collect(1:m).-0.5) ./(m/4)).-2.0

    k = kmul*h
    N = ceil(Int, T/k)
    a = findmin(map(x->abs(x+(h/2)),x))[2]

    # Initial condition, double step function
    if uinit == "step"
        r = @. Float64(x<0.0)*0.8
	r = zeros(size(x)[1])
        r[1] = 0.5
        "u = zeros(size(x)[1])"
        "u = @. 2 * Float64(abs(x-0.5)<0.25) - 0.5"
    else
        error("Unknown initial condition type")
    end

    clf(); axis([-2, 2, -1, 1.5]); grid(true); ph, = plot(x,r)  # Setup plotting
    #N=5
    for it = 1:N
        rr = r
        rl =  r[[end; 1:end-1]] # Periodic boundary conditions

        F = zeros(m)
        if method == "godunov"
           for i = 1:m
                #if i<=4 
                #   println("i=",i," rl[i]=",rl[i]," rr[i]=",rr[i])
                #end
		if rl[i] > rr[i]
		   F[i] = max(rl[i]*(1-rl[i]), rr[i]*(1-rr[i]))
		elseif rl[i] < rr[i]
		   F[i] = min(rl[i]*(1-rl[i]), rr[i]*(1-rr[i]))
		else
		 F[i] = 0
		end

                
           end
        elseif method == "upwind"
           for i = 1:m
               if (ur[i] + ul[i])/2 > 0
                    F[i] = ul[i]^2/2
               else
                    F[i] = ur[i]^2/2
               end
           end
        elseif method == "roe"
           for i = 1:m
              F[i] = 0.5*((rl[i]*(1-rl[i])) + (rr[i]*(1-rr[i]))) - 0.5*(abs(1.01-rl[i]-rr[i])*(rr[i] - rl[i]))
           end
        else
           error("Unknown method type")
        end
        

        if div(it,125)%2 == 0
            F[a] = 0
        end
	if ceil(Int,div(it,237.5))%2 == 0
            F[216] = 0
        end
	println(F[a])
        @. r -= k/h * (F[[2:end; 1]] - F)
        #println("it=",it," k=",k," h=",h,"  r=",r[1:4]," F=",F[1:4])
	
        r[1] = 0.5
        ph.set_data(x,r); pause(1e-3) # Update plot
    end
    return r
end
	
