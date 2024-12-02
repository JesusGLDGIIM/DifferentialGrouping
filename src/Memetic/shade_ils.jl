using Random
using Statistics
using Printf
# using DelimitedFiles
using GeneticAlgorithms.DifferentialEvolution
using GeneticAlgorithms.MyLocalSearch
using GeneticAlgorithms.AbstractAlgorithm
using GeneticAlgorithms.Utils

function checkBounds(x::Vector{Float64}, lb::Float64, ub::Float64)::Bool
    for (i, xi) in enumerate(x)
        if xi < lb || xi > ub
            println("Out of bounds: x[$i] = $xi, lb = $lb, ub = $ub")
            return false
        end
    end
    return true
end

    # Definir la estructura EAresult
    struct EAresult
        solution::Vector{Float64}
        fitness::Float64
        evaluations::Int
    end

# Implementar PoolLast en Julia
mutable struct PoolLast
    options::Vector{String}
    improvements::Dict{String, Float64}
    count_calls::Int
    first::Vector{String}
    new::Union{Nothing, String}

    function PoolLast(options::Vector{String})
        size = length(options)
        @assert size > 0
        return new(copy(options), Dict(options .=> 0.0), 0, shuffle(copy(options)), nothing)
    end
end

function reset!(pool::PoolLast)
    pool.first = shuffle(copy(pool.options))
    pool.new = nothing
    pool.improvements = Dict(pool.options .=> 0.0)
end

function has_no_improvement(pool::PoolLast)::Bool
    return all(v -> v == 0, values(pool.improvements))
end

function get_new(pool::PoolLast)::String
    if !isempty(pool.first)
        return popfirst!(pool.first)
    end

    if pool.new === nothing
        pool.new = update_prob(pool)
    end

    return pool.new
end

function improvement!(pool::PoolLast, obj::String, account::Float64)
    if account < 0
        return
    end

    if !haskey(pool.improvements, obj)
        error("Error, object not found in PoolLast")
    end

    previous = pool.improvements[obj]
    pool.improvements[obj] = account
    pool.count_calls += 1

    if isempty(pool.first)
        if pool.new === nothing || account == 0 || account < previous
            pool.new = update_prob(pool)
        end
    end
end

function update_prob(pool::PoolLast)::String
    if has_no_improvement(pool)
        return pool.options[rand(1:length(pool.options))]
    end

    improvements_values = collect(values(pool.improvements))
    indexes = sortperm(improvements_values)
    posbest = last(indexes)
    best = collect(keys(pool.improvements))[posbest]
    return best
end


# Función para calcular la mejora
function get_improvement(alg_name, before, after)
    ratio = before == 0 ? 0 : (before - after) / before
    return @sprintf("%s: %.3e -> %.3e [%.2e, %.2f]\n", alg_name, before, after, before - after, ratio)
end

# Función para aplicar búsquedas locales
function apply_localsearch(name, method, fitness_fun, bounds, current_best, current_best_fitness, maxevals, SR, group)
    lower = bounds[1]
    upper = bounds[2]
    if method == "grad"
        evals = Evaluaciones(0)
        sol, fit, funcalls = lbfgsb_n(fitness_fun, current_best, lower, upper, evals, group, maxevals)
        # clip_sol = clamp.(sol, lower, upper)
        # if !all(sol .== clip_sol)
        #     fit = fitness_fun(sol)
        # end
    elseif method == "mts"
        res, SR = mtsls(fitness_fun, current_best, current_best_fitness, lower, upper, maxevals, SR, group)
        sol = res.solution
        fit = res.fitness
        funcalls = res.evaluations
    else
        error("Method not implemented")
    end

    if fit <= current_best_fitness
        # println(get_improvement("$method $name", current_best_fitness, fit))
        return EAresult(sol, fit, funcalls)
    else
        return EAresult(current_best, current_best_fitness, funcalls)
    end
end

# Define a function to calculate the ratio of improvement
function get_ratio_improvement(previous_fitness, new_fitness)
    if previous_fitness == 0
        improvement = 0.0
    else
        improvement = (previous_fitness - new_fitness) / previous_fitness
    end
    return improvement
end

# Variables globales para las búsquedas locales
# global SR_global_MTS
# global SR_MTS

global SR_MTS

function shadeils(fitness_fun, funinfo, dim, evals, initial_eval, groups, popsize = 100, threshold = 0.05, p = 0.1, H = 100)    
    lower = funinfo["lower"]
    upper = funinfo["upper"]
    evals = copy(evals)
    bound = [lower, upper]
    # bounds = (fill(lower, dim), fill(upper, dim))
    bounds_partial = (fill(lower, dim), fill(upper, dim))
    popsize = min(popsize, 100)
    maxevals = Int(last(evals))
    totalevals = initial_eval
    num_groups = length(groups)
    # println("Num groups: ", num_groups)
    group_index = 1
    group = groups[group_index]
    #global aux

    shade = SHADE(bound, dim, popsize, H, maxevals, initial_eval)
    DifferentialEvolution.init(shade, fitness_fun, H)

    current_best = EAresult(AbstractAlgorithm.best_solution(shade), AbstractAlgorithm.best_fitness(shade), AbstractAlgorithm.current_evals(shade))
    best_global_solution = current_best.solution
    best_global_fitness = current_best.fitness
    current_best_solution = best_global_solution
    current_best_fitness = best_global_fitness
    
    apply_de = true
    apply_ls = true
    SR = reset_ls(dim, lower, upper)
    methods = ["mts", "grad"]

    # pool_global = PoolLast(methods)
    pool = PoolLast(methods)

    num_worse = 0
    evals_ls = min(50 * dim, 25000)
    evals_de = min(50 * dim, 25000)

    num_restarts = 0

    totalevals = AbstractAlgorithm.current_evals(shade)

    improvement = 0

    while totalevals < maxevals
        current_best = EAresult(current_best_solution, current_best_fitness, 0)
        previous_fitness = current_best.fitness
        method = get_new(pool)
        for g in 1:num_groups
            group = groups[g]

            factor = length(group)/dim*10
            if num_groups == 1
                factor = 1
            end
            # println("Factor: ", factor)

            if apply_de
                AbstractAlgorithm.update(shade, fitness_fun, group, evals_de * factor)
                #improvement = current_best.fitness - best_fitness(shade)
                totalevals += max(popsize, evals_de * factor)
                evals = check_evals(totalevals, evals, best_fitness(shade), best_global_fitness)
                current_best = EAresult(best_solution(shade), best_fitness(shade), totalevals)
                # println("Lower: ", lower, " Upper: ", upper)
                # checkBounds(current_best.solution, lower, upper)
            end

            if apply_ls
                # println("before ls $method")
                # eval_count[] = 0
                # checkBounds(current_best.solution, lower, upper)
                result = apply_localsearch("Local", method, fitness_fun, bounds_partial, current_best.solution, current_best.fitness, evals_ls*factor, SR, group)
                # println("after ls $method")
                checkBounds(result.solution, lower, upper)

                improvement += get_ratio_improvement(current_best_fitness, result.fitness)
                totalevals += result.evaluations
                evals = check_evals(totalevals, evals, result.fitness, best_global_fitness)
                current_best = result

                best_ind = argmin(all_fitness(shade))

                shade.population[:,best_ind] = copy(current_best.solution)
                shade.fitness[best_ind] = current_best.fitness
                
                #improvement!(pool, method, improvement)
            end

            current_best_solution = copy(current_best.solution)
            current_best_fitness = current_best.fitness

            if current_best_fitness < best_global_fitness
                best_global_fitness = current_best_fitness
                best_global_solution = copy(current_best_solution)
            end

            if current_best_fitness < best_fitness(shade)
                shade.best_sol = copy(current_best_solution)
                shade.best_fit = current_best_fitness
            end

            if totalevals >= maxevals
                break
            end
        end

        improvement!(pool, method, improvement)
        improvement = 0
        ratio_improvement = previous_fitness == 0 ? 1 : get_ratio_improvement(previous_fitness, current_best.fitness)
        # @printf("TotalImprovement[%d%%] %.3e => %.3e (%d)\tRestart: %d\n", (round(100 * ratio_improvement)), previous_fitness, current_best.fitness, num_worse, num_restarts)

        if ratio_improvement >= threshold
            num_worse = 0
        else
            num_worse += 1
            # println("Pools Improvements: $(Dict(pool.improvements))")

            SR = reset_ls(dim, lower, upper, method)
        end
        
        # Arreglar el restart
        if num_worse >= 3
            num_worse = 0
            # @printf("Restart: %.2e for %.2f with %d evaluations\n", current_best.fitness, ratio_improvement, totalevals)
            posi = rand(1:popsize)
            new_solution = clamp.(rand(dim) * 0.02 .- 0.01 .+ shade.population[:,posi], lower, upper)
            current_best = EAresult(new_solution, fitness_fun(new_solution), 0)
            current_best_solution = current_best.solution
            current_best_fitness = current_best.fitness

            reset_de(shade, fitness_fun, H, new_solution)
            totalevals += popsize
            # reset!(pool_global)
            reset!(pool)
            SR = reset_ls(dim, lower, upper)
            num_restarts += 1
        end

        # @printf("%.2e(%.2e): with %d evaluations\n", current_best.fitness, best_global_fitness, totalevals)

        if totalevals >= maxevals
            break
        end 
        
    end

    # @printf("%.2e,%s,%d\n", abs(best_global_fitness), join(best_global_solution, " "), totalevals)
    global_res = EAresult(best_global_solution,best_global_fitness,totalevals)
    return global_res, shade
end

function reset_ls(dim, lower, upper, method="all")
    SR_MTS = fill((upper - lower) * 0.2, dim)
    return SR_MTS
end

function reset_de(algo::SHADE, fun, H, p)
    algo.population = Utils.random_population(algo.lower_bounds, algo.upper_bounds,num_dimensions(algo), population_size(algo))
    init(algo, fun, H)
    posi = rand(1:population_size(algo))
    algo.population[: ,posi] = p
end

function check_evals(totalevals, evals, bestFitness, globalBestFitness)
    if !isempty(evals) && totalevals >= evals[1]
        best = min(bestFitness, globalBestFitness)
        # @printf("[%.1e]: %e,%d\n", evals[1], best, totalevals)
        evals = evals[2:end]
    end
    return evals
end
#=
# Ejemplo de uso
dim = 10
funinfo = Dict("lower" => -5.0, "upper" => 5.0)
evals = [100, 500, 1000, 50000]
fitness_fun = x -> sum(x .^ 2)
result , shade = shadeils(fitness_fun, funinfo, dim, evals)
println("Best result: ", result)
=#