# Penalty functions for Flux model training
# Here define a general default penalty (which is 0) for every possible model or layer
# Then, a specific penalty is added to each custom layer or model if needed, making use of multiple dispatch
module Penalties

using Flux

# default to 0 for all layers and models
penalty(l) = 0

# penalty for a whole Flux.Chain model
function penalty(model::Flux.Chain)
    sum(penalty(layer) for layer in model)
end

export penalty

end # module
