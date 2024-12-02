module VariableGrouping

# Exporta las funciones que estarán disponibles desde el módulo
export DG2, ERDG

# Incluye las implementaciones de DG2 y ERDG desde archivos separados
include("DG2.jl")
include("ERDG.jl")

end
