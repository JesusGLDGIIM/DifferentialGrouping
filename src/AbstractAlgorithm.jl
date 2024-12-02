# src/abstract_algorithm.jl
module AbstractAlgorithm

export VariableGroupingAlgorithm, GeneralAlgorithm, get_groups, num_groups, group_sizes,
       population_size, num_dimensions, best_fitness, best_solution, all_fitness, all_population, bounds,
       init, update

# A grouping algorithm should implement this methods
abstract type VariableGroupingAlgorithm end

function get_groups(algo::VariableGroupingAlgorithm)
    error("get_groups not implemented")
end

function num_groups(algo::VariableGroupingAlgorithm)
    error("num_groups not implemented")
end

function group_sizes(algo::VariableGroupingAlgorithm)
    error("group_sizes not implemented")
end

# A optimizing algorithm should implement this methods
abstract type GeneralAlgorithm end

function total_evals(algo::GeneralAlgorithm)
    error("total_evals not implemented")
end

function current_evals(algo::GeneralAlgorithm)
    error("current_evals not implemented")
end

function population_size(algo::GeneralAlgorithm)
    error("population_size not implemented")
end

function num_dimensions(algo::GeneralAlgorithm)
    error("num_dimensions not implemented")
end

function best_fitness(algo::GeneralAlgorithm)
    error("best_fitness not implemented")
end

function best_solution(algo::GeneralAlgorithm)
    error("best_solution not implemented")
end

function all_fitness(algo::GeneralAlgorithm)
    error("all_fitness not implemented")
end

function all_population(algo::GeneralAlgorithm)
    error("all_population not implemented")
end

function bounds(algo::GeneralAlgorithm)
    error("bounds not implemented")
end

function init(algo::VariableGroupingAlgorithm)
    error("init not implemented")
end

function update(algo::VariableGroupingAlgorithm)
    error("update not implemented")
end

function init(algo::GeneralAlgorithm)
    error("init not implemented")
end

function update(algo::GeneralAlgorithm)
    error("update not implemented")
end

end
