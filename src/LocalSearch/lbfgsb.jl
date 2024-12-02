using Optim
using LinearAlgebra

export lbfgsb_n
export Evaluaciones
# Define a wrapper function to use L-BFGS-B from Optim.jl
# function lbfgsb(f, x0, lower_bounds, upper_bounds, max_evals)  
#     result = Optim.optimize(
#         f,
#         x0,
#         Optim.LBFGS(),
#         Optim.Options(f_calls_limit = max_evals, store_trace = true, show_trace = true)
#     )
#     minimizer = Optim.minimizer(result)
#     fitness = Optim.minimum(result)
#     num_evals = Optim.f_calls(result)
#     return minimizer, fitness, num_evals
# end

# Example usage
# function rosenbrock(x)
#     return sum(100.0 * (x[2:end] .- x[1:end-1] .^ 2) .^ 2 + (1.0 .- x[1:end-1]) .^ 2)
# end
# 
# x0 = [-4.76, 7.68]
# lower_bounds = [-100.0, -100.0]
# upper_bounds = [100.0, 100.0]
# max_evals = 100
# 
# minimizer, fitness, num_evals = lbfgsb(rosenbrock, x0, lower_bounds, upper _bounds, max_evals)
# println("Minimizer: $minimizer")
# println("Fitness: $fitness")
# println("Number of evaluations: $num_evals")

#=
# CHECKED
function gradient(f, x, evals; h=1e-8)
    n = length(x)
    grad = zeros(n)
    for i in 1:n
        x_forward = copy(x)
        x_backward = copy(x)
        x_forward[i] += h
        x_backward[i] -= h
        grad[i] = (f(x_forward) - f(x_backward)) / (2 * h)
    end

    evals += 2*n

    return grad
end

# CHECKED
function validate_inputs(x0, l, u)
    # Validar las entradas
    if length(x0) != length(l) || length(x0) != length(u)
        error("Los vectores x0, l y u deben tener la misma longitud.")
    end
    # Asegurar que x0 está dentro de los límites
    x0 = max.(min.(x0, u), l)
    return x0
end
 
# CHECKED
# Devuelve la norma del maximo del gradiente
function get_optimality(x, g, l, u)
    projected_g = x - g
    projected_g = max.(min.(projected_g, u), l) - x
    return maximum(abs.(projected_g))
end

# Punto de Cauchy generalizado
function get_breakpoints(x, g, l, u)
    n = length(x)
    t = zeros(n)
    d = -g
    for i in 1:n
        if g[i] < 0
            t[i] = (x[i] - u[i]) / g[i]
        elseif g[i] > 0
            t[i] = (x[i] - l[i]) / g[i]
        else
            t[i] = Inf
        end
        if t[i] < eps()
            d[i] = 0.0
        end
    end
    F = sortperm(t)
    return t, d, F
end

function get_cauchy_point(x, g, l, u, theta, W, M)
    t, d, F = get_breakpoints(x, g, l, u)
    xc = copy(x)
    p = W' * d
    c = zeros(size(W, 2))
    fp = -d' * d
    fpp = -theta * fp - dot(p, M * p)
    fpp0 = -theta * fp
    dt_min = -fp / fpp
    t_old = 0
    b = F[1]
    t = t[b]
    dt = t - t_old

    for j in 2:length(x)
        if dt_min > dt
            if b > length(xc)
                error("Index b out of bounds: $b is greater than length(xc) which is $(length(xc))")
            end
            if d[b] > 0
                xc[b] = u[b]
            elseif d[b] < 0
                xc[b] = l[b]
            end
            zb = xc[b] - x[b]
            c += dt * p
            gb = g[b]
            wbt = W[b, :]
            fp += dt * fpp + gb * gb + theta * gb * zb - gb * dot(wbt, M * c)
            fpp -= theta * gb * gb + 2.0 * gb * dot(wbt, M * p) + gb * gb * dot(wbt, M * wbt')
            fpp = max(eps() * fpp0, fpp)
            p += gb * wbt'
            d[b] = 0.0
            dt_min = -fp / fpp
            t_old = t
            b = F[j]
            if b > length(t)
                error("Index b out of bounds: $b is greater than length(t) which is $(length(t))")
            end
            t = t[b]
            dt = t - t_old
        else
            break
        end
    end

    dt_min = max(dt_min, 0)
    t_old += dt_min
    for j in F
        if j > length(xc)
            error("Index j out of bounds: $j is greater than length(xc) which is $(length(xc))")
        end
        xc[j] = x[j] + t_old * d[j]
    end
    c += dt_min * p

    return xc, c
end

#
function find_alpha(l, u, xc, du, free_vars_idx)
    alpha_star = 1.0
    for i in free_vars_idx
        if du[i] > 0
            alpha_star = min(alpha_star, (u[i] - xc[i]) / du[i])
        else
            alpha_star = min(alpha_star, (l[i] - xc[i]) / du[i])
        end
    end
    return alpha_star
end

function subspace_min(x, g, l, u, xc, c, theta, W, M)
    line_search_flag = true

    free_vars_idx = []
    Z = []
    for i in 1:eachindex(xc)
        if xc[i] != u[i] && xc[i] != l[i]
            push!(free_vars_idx, i)
            unit = zeros(length(x))
            unit[i] = 1
            Z = hcat(Z, unit)
        end
    end

    num_free_vars = length(free_vars_idx)
    if num_free_vars == 0
        return xc, false
    end

    WTZ = W' * Z

    rr = g + theta * (xc - x) - W * (M * c)
    r = rr[free_vars_idx]

    invtheta = 1.0 / theta
    v = M * (WTZ * r)
    N = invtheta * WTZ * WTZ'
    N = I - M * N
    v = N \ v
    du = -invtheta * r - invtheta^2 * WTZ' * v

    alpha_star = find_alpha(l, u, xc, du, free_vars_idx)

    d_star = alpha_star * du
    xbar = copy(xc)
    for i in 1:num_free_vars
        idx = free_vars_idx[i]
        xbar[idx] += d_star[i]
    end

    return xbar, line_search_flag
end

function strong_wolfe(func, x0, f0, g0, p, evals, maxevals)
    c1 = 1e-4
    c2 = 0.9
    alpha_max = 2.5
    alpha_im1 = 0.0
    alpha_i = 1.0
    f_im1 = f0
    dphi0 = g0' * p
    i = 0
    max_iters = 20

    while evals < maxevals
        x = x0 + alpha_i * p
        f_i = func(x)
        evals += 1
        g_i = gradient(func, x, evals)
        evals += 1
        if f_i > f0 + c1 * alpha_i * dphi0 || (i > 1 && f_i >= f_im1)
            return alpha_zoom(func, x0, f0, g0, p, alpha_im1, alpha_i, evals, maxevals)
        end
        dphi = g_i' * p
        if abs(dphi) <= -c2 * dphi0
            return alpha_i
        end
        if dphi >= 0
            return alpha_zoom(func, x0, f0, g0, p, alpha_i, alpha_im1, evals, maxevals)
        end

        alpha_im1 = alpha_i
        f_im1 = f_i
        alpha_i = alpha_i + 0.8 * (alpha_max - alpha_i)

        if i > max_iters
            return alpha_i
        end

        i += 1
    end
    return alpha_i
end

function alpha_zoom(func, x0, f0, g0, p, alpha_lo, alpha_hi, evals, maxevals)
    c1 = 1e-4
    c2 = 0.9
    i = 0
    max_iters = 20
    dphi0 = g0' * p
    alpha_i = 0.5 * (alpha_lo + alpha_hi)

    while evals < maxevals
        alpha_i = 0.5 * (alpha_lo + alpha_hi)
        x = x0 + alpha_i * p
        f_i = func(x)
        evals += 1
        g_i = gradient(func, x, evals)
        x_lo = x0 + alpha_lo * p
        f_lo = func(x_lo)[1]
        if f_i > f0 + c1 * alpha_i * dphi0 || f_i >= f_lo
            alpha_hi = alpha_i
        else
            dphi = g_i' * p
            if abs(dphi) <= -c2 * dphi0
                return alpha_i
            end
            if dphi * (alpha_hi - alpha_lo) >= 0
                alpha_hi = alpha_lo
            end
            alpha_lo = alpha_i
        end
        i += 1
        if i > max_iters
            return alpha_i
        end
    end
    return alpha_i
end

function lbfgsb(func, x0, l, u, maxevals; m = 10, tol = 1e-5, max_iters = 20, display = false, xhistory = false)
    # Validar las entradas
    x0 = validate_inputs(x0, l, u)

    # Inicializar las variables de L-BFGS
    n = length(x0)
    Y = zeros(n, 0)
    S = zeros(n, 0)
    W = zeros(n, 1)
    M = zeros(1, 1)
    theta = 1.0

    # Inicializar las variables del objetivo
    x = copy(x0)
    f = func(x)
    evals = 1
    g = gradient(func, x, evals)
    

    # Inicializar las iteraciones Quasi-Newton
    k = 0
    xhist = []
    if display
        println(" iter        f(x)          optimality")
        println("-------------------------------------")
        opt = get_optimality(x, g, l, u)
        println("$k $f $opt")
    end

    if xhistory
        push!(xhist, x)
    end

    # Realizar las iteraciones Quasi-Newton
    while get_optimality(x, g, l, u) > tol && k < max_iters && evals < maxevals
        # Actualizar la información de búsqueda
        x_old = copy(x)
        g_old = copy(g)

        # Calcular la nueva dirección de búsqueda
        xc, c = get_cauchy_point(x, g, l, u, theta, W, M)
        xbar, line_search_flag = subspace_min(x, g, l, u, xc, c, theta, W, M)

        alpha = 1.0
        if line_search_flag
            alpha = strong_wolfe(func, x, f, g, xbar - x, evals, maxevals)
        end
        x += alpha * (xbar - x)

        # Actualizar las estructuras de datos de L-BFGS
        f = func(x)
        evals += 1
        g = gradient(f, x, evals)
        y = g - g_old
        s = x - x_old
        curv = abs(dot(s, y))
        if curv < eps()
            println(" warning: negative curvature detected")
            println("          skipping L-BFGS update")
            k += 1
            continue
        end
        if k < m
            Y = hcat(Y, y)
            S = hcat(S, s)
        else
            Y = hcat(Y[:, 2:end], y)
            S = hcat(S[:, 2:end], s)
        end
        theta = dot(y, y) / dot(y, s)
        W = hcat(Y, theta * S)
        A = S' * Y #?
        L = tril(A, -1)
        D = -diagm(diag(A))

        theta_SS = theta * S' * S

        # Construir la matriz MM
        top = hcat(D, L')
        bottom = hcat(L, theta_SS)
        MM = vcat(top, bottom)
        M = inv(MM)

        # Actualizar la iteración
        k += 1
        if xhistory
            push!(xhist, x)
        end

        # Mostrar información de iteración
        if display
            opt = get_optimality(x, g, l, u)
            println("$k $f $opt")
        end
    end

    if k == max_iters
        println(" warning: maximum number of iterations reached")
    end

    if get_optimality(x, g, l, u) < tol
        println(" stopping because convergence tolerance met!")
    end

    return x, f, evals
end

# Ejemplo de uso
function objective_function(x::Vector{Float64})
    return sum(x .^ 2)
end

dim = 10
sol = (rand(dim) .-0.5).* 10
fitness = objective_function(sol)
lower = fill(-5.0, dim)
upper = fill(5.0, dim)
maxevals = 1000

println("Inicial solution: ", sol)
solution, fitness, evaluations = lbfgsb(objective_function, sol, lower, upper, maxevals)
println("Best solution: ", solution)
println("Best fitness: ", fitness)
println("Evaluations: ", evaluations)
=#

mutable struct Evaluaciones
    evals::Int
end

# Función para calcular el gradiente numérico 
# Return a vector of the same length as x with the gradient claculated in the positions in group and 0 everywhere else
function gradient(f, x, evals::Evaluaciones, group, h=1e-8)
    nv = length(x)
    n = length(group)
    grad = zeros(nv)
    for i in 1:n
        index = group[i]
        x_forward = copy(x)
        x_backward = copy(x)
        x_forward[index] += h
        x_backward[index] -= h
        grad[index] = (f(x_forward) - f(x_backward)) / (2 * h)
    end
    evals.evals += 2*n
    return grad
end

# Función para validar las entradas iniciales y proyectar dentro de los límites
function validate_inputs(x0, l, u)
    if length(x0) != length(l) || length(x0) != length(u)
        error("Los vectores x0, l y u deben tener la misma longitud.")
    end
    return clamp.(x0, l, u)  # Proyectar x0 dentro de los límites
end

# Función para proyectar las variables dentro de los límites
function project_bounds(x, l, u)
    return clamp.(x, l, u)
end

# Función para calcular la norma de optimalidad (gradiente proyectado)
function get_optimality(x, g, l, u)
    projected_g = clamp.(x - g, l, u) - x
    return maximum(abs.(projected_g))
end

# Función para actualizar las matrices de memoria limitada L-BFGS
function update_lbfgs_memory(S, Y, s, y, m)
    if size(S, 2) < m
        S = hcat(S, s)
        Y = hcat(Y, y)
    else
        S = hcat(S[:, 2:end], s)
        Y = hcat(Y[:, 2:end], y)
    end
    return S, Y
end

# Función para calcular el punto de Cauchy
function get_cauchy_point(x, g, l, u, W, M)
    n = length(x)
    d = -g
    t = fill(Inf, n)
    x_c = copy(x)
    active_set = falses(n)

    # Calculate times t to the bounds
    for i in 1:n
        if d[i] > 0
            t[i] = (u[i] - x[i]) / d[i]
        elseif d[i] < 0
            t[i] = (l[i] - x[i]) / d[i]
        else
            t[i] = Inf
        end
    end

    # Sort indices according to t
    sorted_indices = sortperm(t)
    f_prime = dot(g, d)
    if f_prime >= 0
        return x_c, active_set
    end

    # Initialize variables
    dt_min = 0.0
    z = zeros(n)
    c = W' * d           # c is (2k,)
    w = W * (M * c)      # w is (n,)
    theta = 1.0          # Ensure theta is consistent with W
    f_prime_zero = f_prime
    f_double_prime = theta * norm(d)^2 + dot(c, M * c)

    for j in sorted_indices
        dt = t[j] - dt_min
        f_prime += dt * f_double_prime
        if f_prime >= 0
            tau = -f_prime_zero / f_double_prime
            z += tau * d
            x_c = x + z
            break
        end
        z[j] += dt * d[j]
        dt_min = t[j]
        x_c[j] = x[j] + t[j] * d[j]
        active_set[j] = true
    end

    return x_c, active_set
end

# Función para calcular el punto de Cauchy
function get_cauchy_point_2(x, g, l, u)
    n = length(x)
    d = -g
    t = fill(Inf, n)

    for i in 1:n
        if d[i] > 0
            t[i] = (u[i] - x[i]) / d[i]
        elseif d[i] < 0
            t[i] = (l[i] - x[i]) / d[i]
        end
    end

    # Encontrar el mínimo t que no sea Inf
    alpha = minimum(t)
    x_c = x + alpha * d

    # Determinar el conjunto activo (variables en los límites)
    active_set = falses(n)
    for i in 1:n
        if x_c[i] == l[i] || x_c[i] == u[i]
            active_set[i] = true
        end
    end

    # Índices de variables libres
    free_vars_idx = findall(!, active_set)

    return x_c, free_vars_idx
end


function subspace_min(x_c, x, g, l, u, active_set, W, M)
    free_indices = findall(!, active_set)
    if isempty(free_indices)
        return zeros(length(x))
    end

    # Construimos la matriz Z
    Z = zeros(length(x), length(free_indices))
    for (i, idx) in enumerate(free_indices)
        Z[idx, i] = 1.0
    end

    # Calculamos el gradiente reducido
    g_free = g[free_indices]

    # Calculamos la dirección en el subespacio
    # WZ = W' * Z
    r = -g_free
    N = Z' * W * M * W' * Z
    p_free = N \ r

    # Construimos la dirección completa
    p = zeros(length(x))
    p[free_indices] = p_free

    # Aseguramos que no violamos los límites
    for idx in free_indices
        if p[idx] > 0
            p[idx] = min(p[idx], u[idx] - x_c[idx])
        elseif p[idx] < 0
            p[idx] = max(p[idx], l[idx] - x_c[idx])
        end
    end

    return p
end

# Función para minimizar en el subespacio
function subspace_min_2(x, g, l, u, free_vars_idx, W, M)
    Z = zeros(length(x), length(free_vars_idx))
    
    for (i, idx) in enumerate(free_vars_idx)
        Z[idx, i] = 1.0
    end

    rr = g .+ W * (M * (W' * Z))  # Utilizar broadcasting
    du = -Z' * rr  # du tendrá dimensiones (k, k)
    
    # Para evitar DimensionMismatch, redefinir du como un vector
    # Por ejemplo, tomando la media de las columnas
    du = -sum(rr .* Z, dims=1)[:, 1]  # Ahora du es Vector{Float64} de tamaño k

    alpha_star = 1.0

    for (i, idx) in enumerate(free_vars_idx)
        if du[i] > 0
            alpha_star = min(alpha_star, (u[idx] - x[idx]) / du[i])
        elseif du[i] < 0
            alpha_star = min(alpha_star, (l[idx] - x[idx]) / du[i])
        end
    end

    # Construir la dirección completa en el espacio n-dimensional
    p = zeros(length(x))
    for (i, idx) in enumerate(free_vars_idx)
        p[idx] = alpha_star * du[i]
    end

    return p
end

function two_loop_recursion(g, S, Y)
    q = copy(g)
    alphas = Vector{Float64}(undef, size(S, 2))
    
    # Primer bucle: Iterar en orden inverso usando axes
    for i in reverse(axes(S, 2))
        alphas[i] = dot(S[:, i], q) / dot(Y[:, i], S[:, i])
        q -= alphas[i] * Y[:, i]
    end

    # Inicializar H0 como escala
    if size(S, 2) > 0
        gamma = dot(S[:, end], Y[:, end]) / dot(Y[:, end], Y[:, end])
    else
        gamma = 1.0
    end
    r = gamma * q

    # Segundo bucle: Iterar en orden directo usando axes
    for i in axes(S, 2)
        beta = dot(Y[:, i], r) / dot(Y[:, i], S[:, i])
        r += S[:, i] * (alphas[i] - beta)
    end

    return -r
end

function lbfgs_direction(S, Y, g, k, m)
    q = copy(g)
    num_corrections = min(k-1, m)
    
    if num_corrections > 0
        alpha = zeros(num_corrections)
        p = zeros(num_corrections)
        
        # Primera fase de la actualización inversa del Hessiano
        for i in reverse(1:num_corrections)
            p[i] = 1.0 / (Y[:, i]' * S[:, i])
            alpha[i] = p[i] * (S[:, i]' * q)
            q -= alpha[i] * Y[:, i]
        end
        
        # Inicializar r con el escalar gamma
        gamma = (S[:, end]' * Y[:, end]) / (Y[:, end]' * Y[:, end])
        r = gamma * q
        
        # Segunda fase de la actualización del Hessiano inverso
        for i in 1:num_corrections
            β = p[i] * (Y[:, i]' * r)
            r += S[:, i] * (alpha[i] - β)
        end
    else
        # Si no hay correcciones almacenadas, usar el gradiente directamente
        r = copy(q)
    end
    
    return -r  # Dirección de descenso
end

function strong_wolfe(f, x, f0, g0, p, evals::Evaluaciones, max_evals, group; c1=1e-4, c2=0.9, max_iter=20)
    alpha0 = 0.0
    alpha1 = 1.0
    alphamax = 5.0
    i = 1
    f_prev = f0
    while i <= max_iter
        x_new = x + alpha1 * p
        f_new = f(x_new)
        evals.evals += 1
        g_new = gradient(f, x_new, evals, group)
        if f_new > f0 + c1 * alpha1 * (g0' * p) || (i > 1 && f_new >= f_prev)
            return zoom(f, x, f0, g0, p, alpha0, alpha1, evals, c1, c2, group)
        end
        if abs(g_new' * p) <= -c2 * (g0' * p)
            return alpha1
        end
        if g_new' * p >= 0
            return zoom(f, x, f0, g0, p, alpha1, alpha0, evals, c1, c2, group)
        end
        alpha0 = alpha1
        f_prev = f_new
        alpha1 = min(2 * alpha1, alphamax)
        i += 1
    end
    return alpha1
end

function zoom(f, x, f0, g0, p, alpha_lo, alpha_hi, evals::Evaluaciones, c1, c2, group)
    max_iter = 20
    i = 1
    alpha = (alpha_lo + alpha_hi) / 2
    while i <= max_iter
        alpha = (alpha_lo + alpha_hi) / 2
        x_new = x + alpha * p
        f_new = f(x_new)
        evals.evals += 1
        g_new = gradient(f, x_new, evals, group)
        x_lo = x + alpha_lo * p
        f_lo = f(x_lo)
        evals.evals += 1
        if f_new > f0 + c1 * alpha * (g0' * p) || f_new >= f_lo
            alpha_hi = alpha
        else
            if abs(g_new' * p) <= -c2 * (g0' * p)
                return alpha
            end
            if g_new' * p * (alpha_hi - alpha_lo) >= 0
                alpha_hi = alpha_lo
            end
            alpha_lo = alpha
        end
        i += 1
    end
    return alpha
end


# Búsqueda de línea utilizando condiciones de Wolfe
function strong_wolfe_2(f, x0, f0, g0, p, evals::Evaluaciones, max_evals; c1=1e-4, c2=0.9)
    alpha_max = 2.5
    alpha = 1.0
    f_im1 = f0
    dphi0 = dot(g0, p)

    i = 1

    while evals.evals < max_evals && i <= 20
        x = x0 + alpha * p
        f_i = f(x)
        evals.evals += 1
        g_i = gradient(f, x, evals)
        dphi = dot(g_i, p)

        # Condición Wolfe 1
        if f_i > f0 + c1 * alpha * dphi0 || (i > 1 && f_i >= f_im1)
            return alpha * 0.5
        end

        # Condición Wolfe 2
        if abs(dphi) <= -c2 * dphi0
            return alpha
        end

        if dphi >= 0
            return alpha * 0.5
        end

        f_im1 = f_i
        alpha = min(alpha * 1.1, alpha_max)  # Limitar alpha_max
        i += 1
    end

    return alpha
end

# Función principal L-BFGS
function lbfgsb_n(f, x0, l, u, evals::Evaluaciones, group, max_evals=100; m=10, tol=1e-6, display=false)
    # Validar y proyectar x0 dentro de los límites
    x = validate_inputs(x0, l, u)
    n = length(x)
    f_val = f(x)
    evals.evals += 1
    g = gradient(f, x, evals, group)
    k = 0  # Contador de iteraciones
    S = zeros(n, 0)
    Y = zeros(n, 0)

    if display
        println("Iteración    f(x)       Norma del gradiente")
        println("-------------------------------------------")
    end

    while evals.evals < max_evals
        if display
            println("$k           $f_val     $(norm(g))")
        end

        # Calcular la dirección de búsqueda usando Two-loop Recursion
        p = two_loop_recursion(g, S, Y)

        # Búsqueda de línea
        alpha = strong_wolfe(f, x, f_val, g, p, evals, max_evals, group)
        x_new = clamp.(x + alpha * p, l, u)
        f_val_new = f(x_new)
        evals.evals += 1
        g_new = gradient(f, x_new, evals, group)

        # Actualizar la memoria limitada
        s = x_new - x
        y = g_new - g
        S, Y = update_lbfgs_memory(S, Y, s, y, m)

        # Actualizar variables para la siguiente iteración
        x = x_new
        f_val = f_val_new
        g = g_new
        k += 1

        # Verificar criterio de convergencia
        if norm(g, Inf) < tol
            if display
                println("Convergencia alcanzada en la iteración $k")
                println("Numero de evaluaciones $evals")
            end
            break
        end
    end

    return x, f_val, evals.evals
end

# Función principal del algoritmo L-BFGS-B (Intento 1)
function lbfgs_b(f, x0, l, u, evals::Evaluaciones, max_evals=100; tol=1e-6, m=10, display=false)
    # Validar las entradas
    x0 = validate_inputs(x0, l, u)

    # Inicializar las variables
    n = length(x0)
    x = copy(x0)
    g = gradient(f, x, evals)
    f_val = f(x)

    evals.evals +=1
    
    S = zeros(n, 0)  # Almacenar diferencias de x
    Y = zeros(n, 0)  # Almacenar diferencias de gradientes

    k = 0

    if display
        println("Iteración    f(x)       Optimalidad")
        println("----------------------------------")
    end
    
    while evals.evals < max_evals
        # Mostrar información de iteración
        optimality = get_optimality(x, g, l, u)
        if display
            println("$k           $f_val     $optimality")
        end

        # Condición de parada
        if optimality < tol
            println("Convergencia alcanzada en la iteración $k")
            println("Número de evaluaciones: ", evals)
            return x
        end

        # Calcular el punto de Cauchy y breakpoints
        cauchy_point, sorted_indices = get_cauchy_point_2(x, g, l, u)

        # Calcular la dirección de descenso usando L-BFGS
        p = lbfgs_direction(S, Y, g, k, m)

        # Proyectar dentro de los límites
        p = project_bounds(x + p, l, u) - x

        # Búsqueda de línea utilizando condiciones de Wolfe
        alpha = strong_wolfe_2(f, x, f_val, g, p, evals, max_evals)
        x_new = project_bounds(x + alpha * p, l, u)

        # Actualizar el gradiente y la función objetivo
        g_new = gradient(f, x_new, evals)
        f_val = f(x_new)
        evals.evals += 1
        # Actualizar las matrices S y Y
        s = x_new - x
        y = g_new - g
        S, Y = update_lbfgs_memory(S, Y, s, y, m)

        # Actualizar variables para la siguiente iteración
        x = x_new
        g = g_new

        k += 1
    end
    println("Número de iteraciones: ", evals)
    println("Número máximo de iteraciones alcanzado.")
    return x, f_val, evals
end

# Función principal del algoritmo L-BFGS-B (Intento 2)
function lbfgsb(f, x0, l, u, evals::Evaluaciones, max_evals=100; m=10, tol=1e-6, display=false)
    # Validar y proyectar x0 dentro de los límites
    x = validate_inputs(x0, l, u)
    n = length(x)
    f_val = f(x)
    evals.evals += 1
    g = gradient(f, x, evals)
    k = 0  # Contador de iteraciones
    S = zeros(n, 0)
    Y = zeros(n, 0)

    if display
        println("Iteración    f(x)       Norma del gradiente")
        println("-------------------------------------------")
    end

    while evals.evals < max_evals
        if display
            println("$k           $f_val     $(norm(g))")
        end

        # println("Iteration $k")
        # println("Size of x: ", size(x))
        # println("Size of g: ", size(g))
        # println("Size of S: ", size(S))
        # println("Size of Y: ", size(Y))

        # Calcular W y M para el subespacio
        if size(S, 2) > 0
            k = size(S, 2)
            theta = 1.0  # or compute based on your algorithm
            W = [Y   theta * S]  # W is (n, 2k)

            # Compute L, D, and M_inv
            L = tril(S' * Y, -1)
            D = diagm(map(i -> (S[:, i]' * Y[:, i]), 1:k))

            MM = vcat(
                hcat(D + theta * (S' * S), L'),
                hcat(L, -theta * (Y' * Y))
            )
            M = inv(MM)
        else
            W = zeros(n, 0)
            M = zeros(0, 0)
        end

        # Calcular el punto de Cauchy y el conjunto activo
        x_c, active_set = get_cauchy_point_2(x, g, l, u)

        # Minimización en el subespacio
        p = subspace_min_2(x_c, g, l, u, active_set, W, M)

        # Búsqueda de línea
        alpha = strong_wolfe_2(f, x_c, f_val, g, p, evals, max_evals)
        x_new = project_bounds(x_c + alpha * p, l, u)
        f_val_new = f(x_new)
        evals.evals += 1
        g_new = gradient(f, x_new, evals)

        # Actualizar la memoria limitada
        s = x_new - x
        y = g_new - g
        S, Y = update_lbfgs_memory(S, Y, s, y, m)

        # Actualizar variables para la siguiente iteración
        x = x_new
        f_val = f_val_new
        g = g_new
        k += 1

        # Verificar criterio de convergencia
        if norm(g, Inf) < tol
            if display
                println("Convergencia alcanzada en la iteración $k")
            end
            break
        end
    end

    return x, f_val
end

#=
# Función de Rosenbrock
function rosenbrock(x)
    sum = 0.0
    for i in 1:length(x)-1
        sum += 100.0 * (x[i+1] - x[i]^2)^2 + (1.0 - x[i])^2
    end
    return sum
end

# Función cuadrática simple
function quadratic_function(x)
    Q = [3.0 2.0; 2.0 6.0]
    c = [-2.0; -6.0]
    return 0.5 * x' * Q * x + c' * x
end

# --- Prueba con la función de Rosenbrock ---
println("Minimizando la función de Rosenbrock...")

# Dimensión del problema
n = 8

# Adivinanza inicial
x0 = [-1.2; 1.0; 1.2; 1.1; 1.3; 0.9; 0.7; -1.0]

# Límites inferiores y superiores
l = fill(-5.0, n)
u = fill(5.0, n)

# Inicializar el contador de evaluaciones

groups = [[1, 3], [2, 4], [5, 7], [6, 8]]

evals = Evaluaciones(0)

# Máximo número de evaluaciones
max_evals = 1000

# Llamar al algoritmo L-BFGS-B
for group in groups
    x_opt, _= lbfgsb_n(rosenbrock, x0, l, u, evals, group, max_evals, tol=1e-6, m=10, display=true)
    # Mostrar el resultado
    println("\nSolución óptima encontrada para la función de Rosenbrock:")
    println("x = ", x_opt)
    println("Valor de la función objetivo en x_opt: ", rosenbrock(x_opt))
    println("Número total de evaluaciones de la función: ", evals.evals)
    evals.evals = 0
    global x0 = x_opt
end
# --- Prueba con la función cuadrática ---
println("\nMinimizando la función cuadrática simple...")

# Adivinanza inicial
x0 = [0.0; 0.0]

# Límites inferiores y superiores
l = fill(-10.0, 2)
u = fill(10.0, 2)

# Reiniciar el contador de evaluaciones
evals.evals = 0

groups = [[1], [2]]

# Llamar al algoritmo L-BFGS-B
for group in groups
    x_opt, _= lbfgsb_n(quadratic_function, x0, l, u, evals, group, max_evals, tol=1e-6, m=10, display=true)

    # Mostrar el resultado
    println("\nSolución óptima encontrada para la función cuadrática:")
    println("x = ", x_opt)
    println("Valor de la función objetivo en x_opt: ", quadratic_function(x_opt))
    println("Número total de evaluaciones de la función: ", evals.evals)
    evals.evals = 0
end    
=#