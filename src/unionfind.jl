module UnionFindAlgo 

export UnionFind, union!, find, isconnected

"""
Union find with the *Weighted quick union* (WQU)
algorithm. 

- `id`: the parent node's id
- `sz`: size of the tree
"""
mutable struct UnionFind
    id::Vector{Int}
    sz::Vector{Int}
    ncomp::Int
    UnionFind(n::Int) = new(collect(1:n), fill(1, n), n) 
end

function UnionFind(data::NamedTuple)
    uf = UnionFind(data.nv)
    edges = data.edges
    for (p, q) in edges
        union!(uf, p, q)
    end
    uf
end 

function union!(uf, p, q) 
    i = find(uf, p)
    j = find(uf, q)
    i == j && return
    # weighted quick-union: always connect the root 
    # of smaller tree to the root of the larger tree 
    if (uf.sz[i] < uf.sz[j])
        uf.id[i] = j
        uf.sz[j] += uf.sz[i]
    else
        uf.id[j] = i
        uf.sz[i] += uf.sz[j]
    end
    uf.ncomp -= 1
    return nothing
end

"""
    find(uf, p)

Find the root of p. The component in WQUPC is defined by
the root of each node. 

In theory path compression can help further flatten
the tree. There are two ways to conduct path compression: 

1. find the root for `p` and then assign all nodes on the path 
   from `p` to root to the newly found root. This requires running 
   the `find` twice.
2. Directly change parent to grandparent, this will halve the 
   height in the subtree. In code: `uf.id[i] = uf.id[uf.id[i]]`

In practice, the performance gain from path compression is not 
significant. On `largeUF.txt`(V=1e6, E=2e6) approach (2) gives
about 1% improvement. The difference might be larger for larger
dataset.
"""
function find(uf, p) 
    i = uf.id[p] 
    while uf.id[i] != i
        # path compression: set parent to grandparent
        # this trick halves the subtree each time
        uf.id[i] = uf.id[uf.id[i]]
        # go to the parent node
        i = uf.id[i]
    end
    i
end

isconnected(uf, p, q) = find(uf, p) == find(uf, q)

end