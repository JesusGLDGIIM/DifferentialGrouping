# Implements SHADE algorithm in Julia
# Tanabe, R.; Fukunaga, A., "Success-history based parameter adaptation
# for Differential Evolution," Evolutionary Computation (CEC), 2013 IEEE
#     Congress on , vol., no., pp.71,78, 20-23 June 2013
#     doi:10.1109/CEC.2013.655755510.1109/CEC.2013.6557555

# TODO: Implementación con archivo externo para peores soluciones (Error al intentar implementarlo en Julia)

module DifferentialEvolution

using ..AbstractAlgorithm
using ..Utils
using Random

export SHADE

mutable struct SHADE <: AbstractAlgorithm.GeneralAlgorithm
    population::Matrix{Float64}
    fitness::Vector{Float64}
    best_sol::Vector{Float64}
    best_fit::Float64
    lower_bounds::Float64
    upper_bounds::Float64
    MemF::Vector{Float64}
    MemCR::Vector{Float64}
    memory::Vector{Vector{Float64}}
    H::Int
    k::Int
    pmin::Float64
    currentEval::Int
    maxEval::Int
    best_fitness_history::Vector{Float64}
    num_eval_history::Vector{Int}
    #group::Vector{Int}
end

function SHADE(bounds::Vector{Float64}, dim::Int, pop_size::Int, H::Int, max_evals::Int, initial_eval::Int)
    lower = bounds[1]
    upper = bounds[2]
    population = random_population(lower, upper, dim, pop_size)
    fitness = fill(Inf, pop_size)
    best_sol = population[:, 1]
    best_fit = Inf
    MemF = fill(0.5, H)
    MemCR = fill(0.5, H)
    memory = [copy(col) for col in eachcol(population)]
    k = 1
    pmin = 2.0 / pop_size
    currentEval = initial_eval
    maxEval = max_evals
    best_fitness_history = Float64[]
    num_eval_history = Int[]
    return SHADE(population, fitness, best_sol, best_fit, lower, upper, MemF, MemCR, memory, H, k, pmin, currentEval, maxEval, best_fitness_history, num_eval_history)
end

function AbstractAlgorithm.total_evals(algo::SHADE)
    return algo.maxEval
end

function AbstractAlgorithm.current_evals(algo::SHADE)
    return algo.currentEval
end

function AbstractAlgorithm.population_size(algo::SHADE)
    return size(algo.population, 2)
end

function AbstractAlgorithm.num_dimensions(algo::SHADE)
    return size(algo.population, 1)
end

function AbstractAlgorithm.best_fitness(algo::SHADE)
    return algo.best_fit
end

function AbstractAlgorithm.best_solution(algo::SHADE)
    return algo.best_sol
end

function AbstractAlgorithm.all_fitness(algo::SHADE)
    return algo.fitness
end

function AbstractAlgorithm.all_population(algo::SHADE)
    return algo.population
end

function AbstractAlgorithm.bounds(algo::SHADE)
    return [algo.lower_bounds, algo.upper_bounds]
end

function AbstractAlgorithm.init(algo::SHADE, fun, H)
    algo.fitness = [fun(Vector{Float64}(ind)) for ind in eachcol(algo.population)]
    algo.best_fit = minimum(algo.fitness)
    algo.best_sol = algo.population[:,argmin(algo.fitness)]
    algo.currentEval += size(algo.population, 2)
    algo.MemF = fill(0.5, H)
    algo.MemCR = fill(0.5, H)
    return algo
end

function AbstractAlgorithm.update(algo::SHADE, fun, group, cicle_evals = algo.maxEval)
    run_evals = 0
    while algo.currentEval < algo.maxEval && run_evals < cicle_evals
        SCR = []
        SF = []
        F = zeros(size(algo.population, 2))
        CR = zeros(size(algo.population, 2))
        u = zeros(size(algo.population))
        weights = []

        numEvalFound = algo.currentEval

        for i in 1:size(algo.population, 2)
            index_H = rand(1:algo.H)
            meanF = algo.MemF[index_H]
            meanCR = algo.MemCR[index_H]
            Fi = clamp(randn() * 0.1 + meanF, 0, 1)
            CRi = clamp(randn() * 0.1 + meanCR, 0, 1)
            p = rand() * (0.2 - algo.pmin) + algo.pmin

             # Seleccionar r1 de la población excluyendo i
            r1 = random_indexes(1, size(algo.population, 2), [i])
            # Seleccionar r2 de la memoria excluyendo i y r1
            r2_idx = rand(1:length(algo.memory))  # Selecciona un índice aleatorio de la memoria
            r2 = r2_idx
            xr1 = algo.population[:, r1]
            xr2 = algo.memory[r2]

            #println("r1: ", r1, "r2: ", r2)

            num_best = max(1, round(Int, p * size(algo.population, 2)))
            best_indices = partialsortperm(algo.fitness, 1:num_best)
            best_index = rand(best_indices)
            xbest = algo.population[:, best_index]

            mask = falses(length(algo.best_sol))
            mask[group] .= true

            v = copy(algo.population[:, i])
            v[group] .= algo.population[:, i][group] .+ Fi .* (xbest[group] .- algo.population[:, i][group]) .+ Fi .* (xr1[group] .- xr2[group])
            v = shade_clip(algo.lower_bounds, algo.upper_bounds, v, algo.population[:, i])

            #v = algo.population[:, i] .+ Fi * (xbest .- algo.population[:, i]) .+ Fi .* (xr1 .- xr2)
            #v = shade_clip(algo.lower_bounds, algo.upper_bounds, v, algo.population[:, i])

            idxchange = (rand(length(v)) .< CRi) .& mask
            u[:, i] .= algo.population[:, i]
            u[idxchange, i] .= v[idxchange]

            #idxchange = rand(length(v)) .< CRi
            # println(rand(length(v)))
            #u[:, i] .= algo.population[:, i]
            #println("u antes del cambio: ", u[:, i])
            #u[idxchange, i] .= v[idxchange]
            #println("u despues del cambio: ", u[:, i])
            
            F[i] = Fi
            CR[i] = CRi
        end

        for i in 1:size(algo.population, 2)
            fitness_u = fun(Vector{Float64}(u[:,i]))

            if fitness_u < algo.fitness[i]
                #println("Dimension memoria: ", length(algo.memory), "Dimension elemento: ", length(copy(algo.population[i])))
                #algo.memory = hcat(algo.memory, copy(algo.population[i]))
                SF = vcat(SF, F[i])
                SCR = vcat(SCR, CR[i])
                weights = vcat(weights, algo.fitness[i] - fitness_u)
                if fitness_u < algo.best_fit
                    algo.best_fit = fitness_u
                    algo.best_sol = copy(u[:,i])
                    # println("Best: ", algo.best_sol, " Best fit: ", algo.best_fit)
                    numEvalFound = algo.currentEval
                    algo.best_fitness_history = vcat(algo.best_fitness_history, algo.best_fit)
                    algo.num_eval_history = vcat(algo.num_eval_history, algo.currentEval)
                    # println("Iteración: ", algo.currentEval, " Mejor fitness hasta ahora: ", algo.best_fit)
                end
                algo.population[:, i] = copy(u[:,i])
                algo.fitness[i] = fitness_u

                # Agregar a la memoria
                push!(algo.memory, copy(algo.population[:, i]))
            end
        end

        algo.currentEval += size(algo.population, 2)
        run_evals += size(algo.population, 2)
        algo.memory = limit_memory(algo.memory, size(algo.population, 2) * 2)

        if length(SCR) > 0 && length(SF) > 0
            Fnew, CRnew = update_FCR(SF, SCR, weights)
            algo.MemF[algo.k] = Fnew
            algo.MemCR[algo.k] = CRnew
            algo.k = mod1(algo.k + 1, algo.H)
        end
    end
    # return algo
end

function shade_clip(lower, upper, solution, original)
    # Crear una copia de la solución para compararla después del recorte
    clip_sol = clamp.(solution, lower, upper)
    
    # Si la solución ya está dentro de los límites, devolverla tal cual
    if all(solution .== clip_sol)
        return solution
    end

    # Ajustar los valores fuera de los límites
    for i in eachindex(solution)
        if solution[i] < lower
            solution[i] = (original[i] + lower) / 2.0
        elseif solution[i] > upper
            solution[i] = (original[i] + upper) / 2.0
        end
    end
    return solution
end


function limit_memory(memory::Vector{Vector{Float64}}, memorySize::Int)
    if length(memory) > memorySize
        indexes = randperm(length(memory))[1:memorySize]
        return memory[indexes]
    else
        return memory
    end
end


function update_FCR(SF, SCR, improvements)
    total = sum(improvements)
    @assert total > 0 "Total improvements must be greater than 0"
    weights = improvements ./ total
    Fnew = sum(weights .* SF .* SF) / sum(weights .* SF)
    Fnew = clamp(Fnew, 0, 1)
    CRnew = sum(weights .* SCR)
    CRnew = clamp(CRnew, 0, 1)
    return Fnew, CRnew
end

end

