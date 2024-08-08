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
    HT2D,
    WaveState,
    collapse!,
    expand,
    TileSet,
    weight,
    tile_count,
    update!,
    Space,
    relations,
    GridSpace,
    GridRelation,
    neighbors

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

abstract type Space end

function neighbors end

struct GridSpace <: Space
    length::Int64
    size::Tuple{Int64, Int64}
end

@enum GridRelation begin
    Above = 1
    Below = 2
    Left  = 3
    Right = 4
end

relations(::GridSpace) = GridRelation

move(::Val{Above}, gs::GridSpace, i::Int) = i - 1
move(::Val{Below}, gs::GridSpace, i::Int) = i + 1
move(::Val{Left}, gs::GridSpace, i::Int) = i - gs.size[1]
move(::Val{Right}, gs::GridSpace, i::Int) = i + gs.size[1]

function neighbors(gs::GridSpace, i::Int)
    result = Tuple{Int, GridRelation}[]
    n = gs.length
    for r = instances(GridRelation)
        idx = move(Val(r), gs, i)
        if checkindex(Bool, 1:n, idx)
            push!(result, (idx, r))
        end
    end
    return result
end

function sealed(::Val{Above}, x::T, o::T) where {T<:AbstractMatrix}
    @assert size(x) == size(o) "Hypertiles must match"
    col = size(x, 2)
    sealed = true
    @inbounds for i = 1:col
        if x[1, i] != o[end, i]
            sealed = false
            break
        end
    end
    return sealed
end


function sealed(::Val{Below}, x::T, o::T) where {T<:AbstractMatrix}
    @assert size(x) == size(o) "Hypertiles must match"
    col = size(x, 2)
    sealed = true
    @inbounds for i = 1:col
        if x[end, i] != o[1, i]
            sealed = false
            break
        end
    end
    return sealed
end

function sealed(::Val{Left}, x::T, o::T) where {T<:AbstractMatrix}
    @assert size(x) == size(o) "Hypertiles must match"
    row = size(x, 1)
    sealed = true
    @inbounds for i = 1:row
        if x[i, 1] != o[i, end]
            sealed = false
            break
        end
    end
    return sealed
end


function sealed(::Val{Right}, x::T, o::T) where {T<:AbstractMatrix}
    @assert size(x) == size(o) "Hypertiles must match"
    row = size(x, 1)
    sealed = true
    @inbounds for i = 1:row
        if x[i, end] != o[i, 1]
            sealed = false
            break
        end
    end
    return sealed
end


struct TileSet{T}
    tiles::Array{T}
    tile_map::Dict{T, Int}
    weights::Array{Float64, 3}
end

function TileSet(tiles::Array{T}, sp::GridSpace) where {T<:AbstractMatrix}
    nt = length(tiles)
    rs = instances(GridRelation)
    nr = length(rs)
    weights = Array{Float64, 3}(undef, nt, nt, nr)
    for i = 1:nt, j = 1:nt, r = 1:nr
        # ex: count when `j` is above `i`
        weights[j, i, r] = sealed(Val(rs[r]), tiles[i], tiles[j])
    end
    TileSet(tiles, weights)
end

function TileSet(tiles::Array{T}, ws::Array{Float64, 3}) where {T}
    TileSet{T}(tiles, Dict(zip(tiles, 1:length(tiles))), ws)
end

function weight(ts::TileSet{T}, x::T, y::T) where {T}
    xid = ts.tile_map[x]
    yid = ts.tile_map[y]
    weight(ts, xid, yid)
end

function weight(ts::TileSet, x::Int, y::Int)
    ts.weights[x, y]
end

tile_count(ts::TileSet) = length(ts.tiles)

# update!(nweights, ts, selection, rel) # TODO
function update!(weights, ts::TileSet, sel::Int, rel::Enum)
    relidx = Int(rel)
    mass = 0.0
    for i = eachindex(weights)
        v = min(weights[i], ts.weights[i, sel, relidx])
        weights[i] = v
        mass += v
    end
    if mass < 1E-4
        fill!(weights, 0.0)
    else
        rmul!(weights, 1.0 / mass)
    end
    return nothing
end

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

#######################################################################

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
    "A vector of collapsed hyper tiles. `0` denotes a non-collapsed cell"
    wave::AbstractArray{Int64} # 0 - uninitialized
    "The number of collapsed cells"
    collapsed::Int64
end

"""
    $(TYPEDSIGNATURES)

Initializes a `WaveState` from a template wave matrix.
"""
function WaveState(template::AbstractMatrix{Int64},
                   sp::GridSpace,
                   ts::TileSet)
    ni,nj,nz = findnz(sparse(template))
    collapsed = length(nz)
    r,c = size(template)
    tc = tile_count(ts)
    weights = ones(tc, r * c)
    entropies = fill(entropy(weights[:, 1]), r * c)
    for (i,j) = zip(ni, nj)
        htile_id = template[i, j]
        lin_idx = i + (r * (j - 1))
        hweights = @view weights[:, lin_idx]
        # update weights and entropy
        fill!(hweights, 0.0)
        hweights[htile_id] = 1.0
        entropies[lin_idx] = 0.0
        propagate!(weights, entropies, template, lin_idx, sp, ts)
    end

    WaveState(weights, entropies, template, collapsed)
end


"""
    $(TYPEDSIGNATURES)

Propagates effects driven by collapsing a given cell.
"""
function propagate!(state::WaveState, cell_id::Int64,
                    sp::Space, ts::TileSet)
    propagate!(state.weights, state.entropies, state.wave,
               cell_id, sp, ts)
end

function propagate!(weights, entropies, wave, cell_id, sp, ts)
    selection = wave[cell_id]
    r, c = size(wave)
    for (ncell, rel) in neighbors(sp, cell_id)
        wave[ncell] == 0 || continue # already collapsed
        # weigths to update
        nweights = @view weights[:, ncell]
        # integrate prop rules
        update!(nweights, ts, selection, rel)
        # update entropy
        entropies[ncell] = entropy(nweights)
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

function collapse_step!(state::WaveState, sp::Space, ts::TileSet)
    selected_cell = argmax(state.entropies)
    collapse_cell!(state, selected_cell)
    propagate!(state, selected_cell, sp, ts)
    state.collapsed += 1
    return nothing
end

"""
    $(TYPEDSIGNATURES)

Iteralively collapses `state`. The resulting wave can be instantiated via [`expand`](@ref).
"""
function collapse!(state::WaveState, sp::Space, ts::TileSet)
    n = length(state.wave)
    while state.collapsed < n
        collapse_step!(state, sp, ts)
    end
    return nothing
end

#######################################################################
# Misc.
#######################################################################

"""
    $(TYPEDSIGNATURES)

Converts the wave into dense collection
"""
function expand(state::WaveState, sp::GridSpace,
                ts::TileSet{<:AbstractMatrix{T}}) where {T}
    dims = (r,c) = sp.size
    imat = reshape(state.wave, dims)
    er, ec = size(ts.tiles[1])
    result = Matrix{T}(undef, r * er, c * ec)
    for ir = 1:r, ic = 1:c
        row_start = (ir - 1) * er + 1
        row_end = ir * er
        col_start = (ic - 1) * ec + 1
        col_end = ic * ec
        lidx = ir + (ic - 1) * c
        result[row_start:row_end, col_start:col_end] =
            ts.tiles[state.wave[lidx]]
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

function softmax(x::AbstractArray{Float64}, t::Float64 = 1.0)
    out = similar(x)
    softmax!(out, x, t)
    return out
end

function softmax!(out::AbstractArray{Float64},
                  x::AbstractArray{Float64}, t::Float64 = 1.0)
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
