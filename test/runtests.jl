using SparseArrays
using StaticArrays
using Distributions
using WaveFunctionCollapse
import WaveFunctionCollapse as WFC


const HT3D = SMatrix{3, 3, Bool}

ht3d_elems = HT3D[
    HT3D([0 0 0;
          0 0 0;
          0 0 0]),
    HT3D([0 1 0;
          0 1 0;
          0 1 0]),
    HT3D([0 0 0;
          1 1 1;
          0 0 0]),
    HT3D([0 1 0;
          1 1 1;
          0 0 0]),
    HT3D([0 0 0;
          1 1 1;
          0 1 0]),
    HT3D([0 1 0;
          1 1 0;
          0 1 0]),
    HT3D([0 1 0;
          0 1 1;
          0 1 0]),
    HT3D([0 1 0;
          1 1 0;
          0 0 0]), # 8
    HT3D([0 1 0;
          0 1 1;
          0 0 0]), # 9
    HT3D([0 0 0;
          0 1 1;
          0 1 0]), # 10
    HT3D([0 0 0;
          1 1 0;
          0 1 0]), # 11
    HT3D([0 1 0;
          1 1 1;
          0 1 0]), # 12
]

function test_grid_relations()
    @show WFC.sealed(Val(WFC.Above), ht3d_elems[2], ht3d_elems[1])
    @show WFC.sealed(Val(WFC.Above), ht3d_elems[2], ht3d_elems[5])

    @show WFC.sealed(Val(WFC.Below), ht3d_elems[2], ht3d_elems[3])
    @show WFC.sealed(Val(WFC.Below), ht3d_elems[2], ht3d_elems[4])

    @show WFC.sealed(Val(WFC.Left), ht3d_elems[2], ht3d_elems[3])
    @show WFC.sealed(Val(WFC.Left), ht3d_elems[2], ht3d_elems[1])


    @show WFC.sealed(Val(WFC.Right), ht3d_elems[2], ht3d_elems[3])
    @show WFC.sealed(Val(WFC.Right), ht3d_elems[3], ht3d_elems[12])
    return nothing
end

test_grid_relations()

function test_tileset()
    ts = TileSet(ht3d_elems, GridSpace(64, (8,8)))
    display(ts.weights)
    return nothing
end

test_tileset();



function gen_template(n::Int64)
    template = zeros(Int64, (n,n))
    template[1:end, 1] .= 1
    template[1:end, end] .= 1
    template[1, 1:end] .= 1
    template[end, 1:end] .= 1
    # entrance
    template[end, div(n, 2) + 1] = 2
    # exit
    template[1, 3] = 2
    template[1, n - 2] = 2
    return template
end

function dist(x::CartesianIndex{2}, y::CartesianIndex{2})
    sqrt((x[1] - y[1])^2 + (x[2] - y[2])^2)
end

function add_segment!(m::Matrix{Int64}, pidx::Int, cdir, pdir)
    if cdir == WFC.Above # going up
        if pdir == WFC.Above
            m[pidx] = 2
        elseif pdir == WFC.Left
            m[pidx] = 9
        elseif pdir == WFC.Right
            m[pidx] = 8
        end
    elseif cdir == WFC.Below
        if pdir == WFC.Below
            m[pidx] = 2
        elseif pdir == WFC.Left
            m[pidx] = 10
        elseif pdir == WFC.Right
            m[pidx] = 11
        end
    elseif cdir == WFC.Left
        if pdir == WFC.Below
            m[pidx] = 8
        elseif pdir == WFC.Above
            m[pidx] = 9
        elseif pdir == WFC.Left
            m[pidx] = 3
        end
    else
        if pdir == WFC.Below
            m[pidx] = 9
        elseif pdir == WFC.Above
            m[pidx] = 10
        elseif pdir == WFC.Right
            m[pidx] = 3
        end
    end
    return nothing
end

function sample_path(n::Int, start::Int, dest::Int)
    m = zeros(Int64, (n, n))
    m[:, 1] .= 1
    # m[:, end] .= 1
    cis = CartesianIndices(m)
    dest_ci = cis[dest]
    gs = GridSpace(n * n, (n, n))
    current = prev = start
    dir = dir_prev = WFC.Right
    temp = 10.0
    while current != dest
        ns = neighbors(gs, current)
        nns = length(ns)
        ws = fill(-Inf, nns)
        for i = 1:nns
            n, _ = ns[i]
            m[n] == 0 || continue
            ws[i] = -dist(cis[n], dest_ci)
        end
        ws = WFC.softmax(ws, temp)
        selected = rand(Categorical(ws))
        prev = current
        dir_prev = dir
        current, dir = ns[selected]
        add_segment!(m, prev, dir, dir_prev)
        temp *= 0.5
    end

    add_segment!(m, current, WFC.Right, dir)


    return m
end

function test_collapse()
    template = sample_path(9, 5, 80)
    # template = sample_path(5, 3, 24)
    display(template)
    # template = gen_template(9)
    sp = GridSpace(length(template), size(template))
    ts = TileSet(ht3d_elems, sp)
    # template = [1 1; 0 2]
    ws = WaveState(sparse(template), sp, ts)
    # display(template)
    # display(ws.weights)
    @time collapse!(ws, sp, ts)
    display(ws.wave)
    display(expand(ws, sp, ts))
    return nothing
end


test_collapse();
