export DG2

# Julia implementation of the DG2 algorithm

# Function to find connected components in a graph represented by an adjacency matrix
function findConnComp(C)
    L = size(C, 1)  # Number of vertices
    labels = zeros(Int, L)  # All vertices unexplored at the beginning
    ccc = 0  # Connected components counter

    while true
        ind = findall(labels .== 0)
        if !isempty(ind)
            fue = ind[1]  # First unexplored vertex
            list = [fue]
            ccc += 1
            labels[fue] = ccc
            while true
                list_new = Int[]
                for p in list
                    cp = findall(C[p, :] .!= 0)  # Points connected to p
                    cp1 = cp[labels[cp] .== 0]    # Get only unexplored vertices
                    labels[cp1] .= ccc
                    append!(list_new, cp1)
                end
                list = list_new
                if isempty(list)
                    break
                end
            end
        else
            break
        end
    end

    group_num = maximum(labels)
    allgroups = Vector{Vector{Int}}()
    for i in 1:group_num
        push!(allgroups, findall(labels .== i))
    end

    return allgroups
end

# Function to compute the interaction structure matrix (ISM)
function ism(fun, options)
    ub = options["ubound"]
    lb = options["lbound"]
    dim = options["dim"]

    FEs = 0
    temp = (ub + lb) / 2

    f_archive = fill(NaN, dim, dim)
    fhat_archive = fill(NaN, dim)
    delta1 = fill(NaN, dim, dim)
    delta2 = fill(NaN, dim, dim)
    lambda = fill(NaN, dim, dim)

    p1 = lb * ones(dim)
    fp1 = fun(p1)
    FEs += 1

    counter = 0
    prev = 0
    prog = 0

    for i in 1:(dim - 1)
        if !isnan(fhat_archive[i])
            fp2 = fhat_archive[i]
        else
            p2 = copy(p1)
            p2[i] = temp
            fp2 = fun(p2)
            FEs += 1
            fhat_archive[i] = fp2
        end

        for j in (i + 1):dim
            counter += 1
            prev = prog
            prog = floor(Int, counter / (dim * (dim - 1)) * 2 * 100)
            if mod(prog, 5) == 0 && prev != prog
                println("Progress = $prog%")
            end

            if !isnan(fhat_archive[j])
                fp3 = fhat_archive[j]
            else
                p3 = copy(p1)
                p3[j] = temp
                fp3 = fun(p3)
                FEs += 1
                fhat_archive[j] = fp3
            end

            p4 = copy(p1)
            p4[i] = temp
            p4[j] = temp
            fp4 = fun(p4)
            FEs += 1
            f_archive[i, j] = fp4
            f_archive[j, i] = fp4

            d1 = fp2 - fp1
            d2 = fp4 - fp3

            delta1[i, j] = d1
            delta2[i, j] = d2
            lambda[i, j] = abs(d1 - d2)
        end
    end

    evaluations = Dict("base" => fp1, "fhat" => fhat_archive, "F" => f_archive, "count" => FEs)
    delta = Dict("delta1" => delta1, "delta2" => delta2)
    return delta, lambda, evaluations
end

# Differential Grouping method to identify separable and non-separable variables
function dsm(evaluations, lambda, dim)
    fhat_archive = evaluations["fhat"]
    f_archive = evaluations["F"]
    fp1 = evaluations["base"]

    F1 = ones(dim, dim) * fp1
    F2 = repeat(reshape(fhat_archive, 1, dim), dim, 1)
    F3 = repeat(fhat_archive, 1, dim)
    F4 = f_archive

    FS = cat(F1, F2, F3, F4; dims=3)
    Fmax = dropdims(maximum(FS, dims=3), dims=3)
    Fmin = dropdims(minimum(FS, dims=3), dims=3)

    FS_inf = cat(F1 + F4, F2 + F3; dims=3)
    Fmax_inf = dropdims(maximum(FS_inf, dims=3), dims=3)

    theta = fill(NaN, dim, dim)
    muM = eps() / 2
    gamma = n -> (n * muM) / (1 - n * muM)
    errlb = gamma(2) * Fmax_inf
    errub = gamma(sqrt(dim)) * Fmax

    I1 = lambda .<= errlb
    I2 = lambda .>= errub
    theta[I1] .= 0
    theta[I2] .= 1

    si1 = sum(I1)
    si3 = sum(I2)

    I0 = lambda .== 0
    c0 = sum(I0)
    count_seps = sum((.!I0) .& I1)
    count_nonseps = sum(I2)
    reliable_calcs = count_seps + count_nonseps

    w1 = ((count_seps + c0) / (c0 + reliable_calcs))
    w2 = (count_nonseps / (c0 + reliable_calcs))
    epsilon = w1 * errlb + w2 * errub

    grayind = (lambda .< errub) .& (lambda .> errlb)
    grayindsum = sum(grayind)
    AdjTemp = lambda .> epsilon

    idx = isnan.(theta)
    theta[idx] .= AdjTemp[idx]
    theta = Bool.(theta) .| Bool.(theta')  # Asegurar la simetría

    # Corrección: Establecer la diagonal principal a true
    diag_indices = [i + (i - 1) * dim for i in 1:dim]  # Índices de la diagonal
    theta[diag_indices] .= true

    components = findConnComp(theta)

    h = x -> length(x) == 1
    sizeone = map(h, components)

    seps = components[sizeone]
    seps = isempty(seps) ? [] : reduce(vcat, seps)

    components = components[.!sizeone]
    nonseps = components

    return nonseps, seps, theta, epsilon
end

function DG2(fun, dim, lb, ub)
    # Configuración inicial
    options = Dict("lbound" => lb, "ubound" => ub, "dim" => dim)

    # Paso 1: Calcular matriz de interacción con ISM
    delta, lambda, evaluations = ism(fun, options)

    # Paso 2: Identificar separables y no separables con DSM
    nonseps, seps, theta, epsilon = dsm(evaluations, lambda, dim)

    # Paso 3: Formatear salida
    groups = copy(nonseps)
    # Agregar variables separables como grupos individuales
    for sep in seps
        push!(groups, [sep])
    end

    # Número total de evaluaciones de la función objetivo
    num_evals = evaluations["count"]

    return groups, num_evals
end

#=
# Example usage
# Define the function to be optimized
# For demonstration purposes, let's define a simple quadratic function
function test_function(x)
    return sum(x .^ 2) + (25 - x[1]*x[2])^2 + (x[1] + x[2])^2 + (25 - x[3]*x[4])^2 + (x[3] + x[4])^2
end

# Configuración del problema
dim = 100
lb = -5.0
ub = 5.0

# Llamar a DG2
groups, num_evals = DG2(test_function, dim, lb, ub)

# Mostrar resultados
println("Grupos identificados:")
println(groups)
println("Número de evaluaciones de la función objetivo:")
println(num_evals)
=#
