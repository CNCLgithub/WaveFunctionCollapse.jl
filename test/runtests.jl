using WaveFunctionCollapse
using SparseArrays

function gen_template(n::Int64)
    template = zeros(Int64, (n,n))
    template[1, 1] = 10
    template[2:end-1, 1] .= 9
    template[end, 1] = 13
    template[1, end] = 11
    template[2:end-1, end] .= 7
    template[1, 2:end-1] .= 6
    template[end, end] = 12
    template[end, 2:end-1] .= 8
    return template
end

function test_collapse()
    prop_rules = gen_prop_rules()
    display(prop_rules)
    template = gen_template(8)
    # template = [1 1; 0 2]
    ws = WaveState(sparse(template), prop_rules)
    display(ws.weights)
    collapse!(ws, prop_rules)
    display(ws.wave)
    display(BitMatrix(expand(ws)))
end


test_collapse();
