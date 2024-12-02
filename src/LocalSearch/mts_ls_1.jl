using Random
using Statistics

struct EAresult
    solution::Vector{Float64}
    fitness::Float64
    evaluations::Int
end

function _mtsls_improve_dim(fun, sol, best_fitness, i, check, SR)
    newsol = copy(sol)
    newsol[i] -= SR[i]
    newsol = check(newsol)
    fitness_newsol = fun(newsol)
    evals = 1

    if fitness_newsol < best_fitness
        best_fitness = fitness_newsol
        sol = newsol
    elseif fitness_newsol > best_fitness
        newsol = copy(sol)
        newsol[i] += 0.5 * SR[i]
        newsol = check(newsol)
        fitness_newsol = fun(newsol)
        evals += 1

        if fitness_newsol < best_fitness
            best_fitness = fitness_newsol
            sol = newsol
        end
    end

    return EAresult(sol, best_fitness, evals)
end

function mtsls(fun, sol, fitness, lower, upper, maxevals, SR, group)
    #dim = length(sol)
    dim = length(group)
    improved_dim = falses(dim)
    check = x -> clamp.(x, lower, upper)
    current_best = EAresult(sol, fitness, 0)
    totalevals = 0
    improvement = zeros(dim)

    if totalevals < maxevals
        dim_sorted = randperm(dim)

        for i in dim_sorted
            index = group[i]
            # println("Index: ", index)
            result = _mtsls_improve_dim(fun, current_best.solution, current_best.fitness, index, check, SR)
            totalevals += result.evaluations
            improve = max(current_best.fitness - result.fitness, 0)
            improvement[i] = improve

            if improve > 0
                improved_dim[i] = true
                current_best = result
            else
                SR[index] /= 2
            end
        end

        dim_sorted = sortperm(improvement, rev=true)
        d = 1
    end

    while totalevals < maxevals
        i = dim_sorted[d]
        index = group[dim_sorted[d]]
        # println("Index: ", index)
        result = _mtsls_improve_dim(fun, current_best.solution, current_best.fitness, index, check, SR)
        totalevals += result.evaluations
        improve = max(current_best.fitness - result.fitness, 0)
        improvement[i] = improve
        next_d = mod1(d + 1, dim)
        next_i = dim_sorted[next_d]

        if improve > 0
            improved_dim[i] = true
            current_best = result

            if improvement[i] < improvement[next_i]
                dim_sorted = sortperm(improvement, rev=true)
            end
        else
            SR[i] /= 2
            d = next_d
        end
    end

    # Check lower value
    initial_SR = 0.2 * (upper .- lower)
    SR = map(x -> ifelse(x < 1e-15, initial_SR, x), SR)  # Usar `map` para asegurar que `SR` se actualiza correctamente

    final_result = EAresult(current_best.solution, current_best.fitness, totalevals)
    return final_result, SR
end

#=
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
SR = fill(0.5, dim)

group = [1, 5, 6, 10]

println("Inicial solution: ", sol)
result, final_SR = mtsls(objective_function, sol, fitness, lower, upper, maxevals, SR, group)
println("Best solution: ", result.solution)
println("Best fitness: ", result.fitness)
println("Evaluations: ", result.evaluations)
println("Final SR: ", final_SR)
=#