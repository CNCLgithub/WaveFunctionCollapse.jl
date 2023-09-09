"""
The [`WaveFunctionCollapse`](@ref) module.

Exports:
$(EXPORTS)

Imports:
$(IMPORTS)

---

Wave function collapse in Julia

> This is a work in progress.

---

The [`LICENSE`](@ref) abbreviation can be used in the same way for the `LICENSE.md` file.
"""
module WaveFunctionCollapse

#######################################################################
# Exports
#######################################################################

export HT2D_vec,
    WaveState,
    collapse!,
    gen_prop_rules,
    expand

#######################################################################
# Dependencies
#######################################################################

using Random
using LinearAlgebra
using Distributions
using StaticArrays
using SparseArrays
using DocStringExtensions

#######################################################################
# Types
#######################################################################

# TODO: add ability to extend hypertile spaces
# "A space of hyper tile values"
# abstract type HyperTileSpace end

# "A cartesian hyper tile of NxN dimensions"
# struct HyperTile{N} <: HyperTileSpace end

# const HT2 = HyperTile{2}

#######################################################################
# Constants
#######################################################################

const HT2D = SMatrix{2, 2, Bool}

const _one_ht2d = HT2D([1 0;
                        0 0])
const _two_ht2d = HT2D([1 1;
                        0 0])
const _three_ht2d = HT2D([1 1;
                          1 0])
const HT2D_vec = Vector{HT2D}([
    # zero
    HT2D(falses(2,2)),
    # one
    _one_ht2d,
    rotr90(_one_ht2d, 1),
    rotr90(_one_ht2d, 2),
    rotr90(_one_ht2d, 3),
    # two
    _two_ht2d,
    rotr90(_two_ht2d, 1),
    rotr90(_two_ht2d, 2),
    rotr90(_two_ht2d, 3),
    # three
    _three_ht2d,
    rotr90(_three_ht2d, 1),
    rotr90(_three_ht2d, 2),
    rotr90(_three_ht2d, 3),
    # four
    HT2D(trues(2,2))
])

const _ht2_hash_map = Dict(zip(HT2D_vec, collect(1:length(HT2D_vec))))

#######################################################################
# Methods
#######################################################################

# TODO: generalize
"""
    $(SIGNATURES)

Generates a propagation weight matrix for the 2D hypertile space.
"""
function gen_prop_rules()
    n = length(HT2D_vec)
    ws = zeros(n, n)
    for (i, hti) = enumerate(HT2D_vec)
        ni = count(hti)
        for (j,  htj) = enumerate(HT2D_vec)
            nj = count(htj)
            ws[i, j] = ((hti[3] == htj[1]) + (hti[4] == htj[2])) /
                    ((1+nj)^2)
        end
    end
    return ws
end

"""
The state describing a step in the collapse process

$(TYPEDEF)

---

$(TYPEDFIELDS)
"""
mutable struct WaveState
    """
    A KxN matrix where `K` denotes the number of hypertiles and
    `N` denotes the number of indices in the wave
    """
    weights::Matrix{Float64}
    "The entropy of each index in the wave"
    entropies::Vector{Float64}
    "A matrix of collapsed hyper tiles. `0` denotes a non-collapsed cell"
    wave::Matrix{Int64} # 0 - uninitialized
    "The number of collapsed cells"
    collapsed::Int64
end

"""
    $(TYPEDSIGNATURES)

Initializes a `WaveState` from a template wave matrix.
"""
function WaveState(template::AbstractMatrix{Int64},
                   prop_rules::AbstractMatrix{Float64})
    ni,nj,nz = findnz(template)
    collapsed = length(nz)
    r,c = size(template)
    n_htiles = size(prop_rules, 1)
    weights = fill(1.0 / n_htiles, (n_htiles, r * c))
    weights = ones(n_htiles, r * c)
    entropies = fill(entropy(weights[:, 1]), r * c)
    for (i,j) = zip(ni, nj)
        htile_id = template[i, j]
        lin_idx = i + (r * (j - 1))
        hweights = @view weights[:, lin_idx]
        # update weights and entropy
        fill!(hweights, 0.0)
        hweights[htile_id] = 1.0
        entropies[lin_idx] = 0.0
        propagate!(weights, entropies, template, lin_idx, prop_rules)
    end

    WaveState(weights, entropies, template, collapsed)
end


"""
    $(TYPEDSIGNATURES)

Propagates effects driven by collapsing a given cell.
"""
function propagate!(state::WaveState, cell_id::Int64,
                    prop_rules::AbstractMatrix{Float64})
    propagate!(state.weights, state.entropies, state.wave,
               cell_id, prop_rules)
end

function propagate!(weights, entropies, wave, cell_id, prop_rules)
    # @show cell_id
    selection = wave[cell_id]
    r, c = size(wave)
    for (ncell, d) in neighbors(r, c, cell_id)
        # @show ncell
        wave[ncell] == 0 || continue # already collapsed
        hweights = @view weights[:, ncell]
        # @show hweights
        # convert selection to standard form
        standardized = rotate(selection, d)
        # lookup rules
        rules = prop_rules[:, standardized]
        # @show rules
        # transform to current orientation
        rules = rotate(rules, -d)
        # @show selection
        # @show standardized
        # @show d
        # @show rules
        hweights .*= rules
        # rmul!(hweights, 0.5)
        # re-normalize weights
        rmul!(hweights, 1.0 / sum(hweights))
        # softmax!(hweights, hweights; t = 0.1) # TODO
        # @show hweights
        entropies[ncell] = entropy(hweights)
    end
    return nothing
end

function collapse_cell!(state::WaveState, cell_id::Int64)
    hweights = state.weights[:, cell_id]
    # sample hyper tile
    htile_id = rand(Categorical(hweights))
    state.wave[cell_id] = htile_id
    # update weights and entropy
    fill!(hweights, 0.0)
    hweights[htile_id] = 1.0
    state.entropies[cell_id] = 0.0
    return nothing
end

function collapse_step!(state::WaveState,
                        prop_rules::AbstractMatrix{Float64})
    selected_cell = argmax(state.entropies)
    # println("collapsed $(selected_cell)")
    collapse_cell!(state, selected_cell)
    propagate!(state, selected_cell, prop_rules)
    state.collapsed += 1
    return nothing
end

"""
    $(TYPEDSIGNATURES)

Iteralively collapses `state`. The resulting wave can be instantiated via [`expand`](@ref).
"""
function collapse!(state::WaveState,
                   prop_rules::AbstractMatrix{Float64})
    n = length(state.wave)
    while state.collapsed < n
        collapse_step!(state, prop_rules)
    end
    return nothing
end

#######################################################################
# Misc.
#######################################################################

"""
    $(TYPEDSIGNATURES)

Instantiates the binary matrix defined in `state.wave`.
"""
function expand(state::WaveState)
    r, c = size(state.wave)
    result = Matrix{Bool}(undef, r * 2, c * 2)
    for ir = 1:r, ic = 1:c
        row_start = (ir - 1) * 2 + 1
        row_end = ir * 2
        col_start = (ic - 1) * 2 + 1
        col_end = ic * 2
        result[row_start:row_end, col_start:col_end] = HT2D_vec[state.wave[ir, ic]]
    end
    return result
end


const _neighbhors = [((0, -1), 0),
                     ((-1, 0), 3),
                     ((1, 0),  1),
                     ((0, 1),  2)]
function neighbors(r::Int64, c::Int64, cell_id::Int64)
    cell_row = ((cell_id - 1) % r) + 1
    cell_col = ceil(Int64, cell_id / r)
    ch = Channel{NTuple{2, Int64}}() do ch
        for ((shift_r, shift_c), d) in _neighbhors
            n_row = cell_row + shift_r
            n_col = cell_col + shift_c
            ((n_row < 1 || n_row > r) ||
                (n_col < 1 || n_col > c)) &&
                continue
            idx = (n_col - 1) * r + n_row
            put!(ch, (idx, d))
        end
    end
end

# TODO: remove intermediate step
function rotate(i::Int64, r::Int64)
    htile = HT2D_vec[i]
    rotated = rotl90(htile, r)
    _ht2_hash_map[rotated]
end

function rotate(v::AbstractVector{Float64}, r::Int64)
    result = Vector{Float64}(undef, length(v))
    for i = eachindex(v)
        result[rotate(i, r)] = v[i]
    end
    return result
end

#######################################################################
# Math
#######################################################################

function entropy(a::AbstractArray{Float64})
    result = 0.0
    for w in a
        result += w * log(w)
    end
    -1.0 * result
end

function softmax(x::AbstractArray{Float64}; t::Float64 = 1.0)
    out = similar(x)
    softmax!(out, x; t = t)
    return out
end

function softmax!(out::AbstractArray{Float64},
                  x::AbstractArray{Float64}; t::Float64 = 1.0)
    nx = length(x)
    maxx = maximum(x)
    sxs = 0.0

    if maxx == -Inf
        out .= 1.0 / nx
        return nothing
    end

    @inbounds for i = 1:nx
        out[i] = @fastmath exp((x[i] - maxx) / t)
        sxs += out[i]
    end
    rmul!(out, 1.0 / sxs)
    return nothing
end

end # module WaveFunctionCollapse
