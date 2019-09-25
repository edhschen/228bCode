using SparseArrays
using LinearAlgebra

function euler_fluxes(r, ru, rv, rE)
    gamma = 7.0/5
    u = @. ru/r
    v = @. rv/r
    p = @. (gamma - 1.0)*(rE-(ru^2/r+rv^2/r)/2.0)

    Frx = ru
    Fry = rv
    Frux= @. r*u^2+p
    Fruy= @. ru*v
    Frvx= @. ru*v
    Frvy= @. r*v^2+p
    FrEx= @. u*(rE+p)
    FrEy= @. v*(rE+p)
    return Frx, Fry, Frux, Fruy, Frvx, Frvy, FrEx, FrEy
end


function compact_div(Fx, Fy, h)
    global m
    m=ceil(Int,1/h)
    #x = h * (1:m)
    #m = 4  # testing purpose
    #println("m=",m)
    LHS = SymTridiagonal(ones(m), ones(m-1)/4);
    #println(LHS)
    RHS = (3 / 4h) * Tridiagonal(-ones(m-1), zeros(m), ones(m-1));
    #println(RHS)
    RHS = sparse(RHS)
    #println(RHS)
    dFx = LHS \ (RHS * Fx)
    dFy = LHS \ (RHS * Fy)
    divF = dFx+dFy
    return divF
end

function compact_filter(u, alpha)
    a = 5/8+3*alpha/4
    b = alpha+1/2
    c = alpha/4-1/8
    global m
    LHS = SymTridiagonal(ones(m), alpha*ones(m-1));
    RHS=sparse([1:m;2:m;1:m-1;3:m;1:m-2], [1:m;1:m-1;2:m;1:m-2;3:m],
           [a*ones(m); b/2.0*ones(2*(m-1)); c/2.0*ones(2*(m-2))])
    u_filt = LHS \ (RHS * u)
    return u_filt
end

function euler_rhs(r, ru, rv, rE, h)
    Frx, Fry, Frux, Fruy, Frvx, Frvy, FrEx, FrEy = euler_fluxes(r, ru, rv, rE)
    #println("Frx=",Frx)
    #println("Fry=",Fry)
    fr  =-compact_div(Frx,  Fry,  h)
    fru =-compact_div(Frux, Fruy, h)
    frv =-compact_div(Frvx, Frvy, h)
    frE =-compact_div(FrEx, FrEy, h)
    return fr, fru, frv, frE
end


function ff(u,h)
    #println("size of u in ff=",size(u))
    #println("u in ff=",u)
    fr, fru, frv, frE = euler_rhs(u[:,1], u[:,2], u[:,3], u[:,4], h)
    return [fr fru frv frE]
end

function euler_rk4step(r, ru, rv, rE, h, k, alpha)
    u = [r ru rv rE]
    #println("u=",u)
    #println("size of u=",size(u))
    k1 = k * ff(u, h)
    k2 = k * ff(u + 0.5k1, h)
    k3 = k * ff(u + 0.5k2, h)
    k4 = k * ff(u + k3, h)
    u = u + (k1 + 2k2 + 2k3 + k4) / 6
    #println(u)
    u = compact_filter(u,alpha)
    return u[:,1], u[:,2], u[:,3], u[:,4]
end

if true
  ru = [1;2;3;4]
  rv = [5;6;7;8]
  r  = [2;3;4;2]
  rE = [1;2;2;2]
  println(euler_rk4step(r, ru, rv, rE, 0.25, 0.2, 0.3))
end


#f(x, y) = x * sqrt(y)
function rk4(f::Function, x0::Float64, y0::Float64, x1::Float64, n)
    vx = Array{Float64}(undef, n+1)
    vy = Array{Float64}(undef, n+1)
    vx[1] = x = x0
    vy[1] = y = y0
    h = (x1 - x0) / n
    for i in 1:n
        k1 = h * f(x, y)
        k2 = h * f(x + 0.5h, y + 0.5k1)
        k3 = h * f(x + 0.5h, y + 0.5k2)
        k4 = h * f(x + h, y + k3)
        vx[i + 1] = x = x0 + i * h
        vy[i + 1] = y = y + (k1 + 2k2 + 2k3 + k4) / 6
    end
    return vx, vy
end
