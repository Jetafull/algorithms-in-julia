module Searching 

export binarysearch, BinarySearchTree, BinaryNode, delete!
export size, get, maximum, minimum

# Binary search 
struct UnsortedException <: Exception end
Base.showerror(io::IO, e::UnsortedException) = print(io, "input vecotr v is not sorted")

midpoint(lo::Integer, hi::Integer) = lo + div(hi - lo, 2)

function binarysearch(key, v::AbstractVector; algo=:recursive)
    issorted(v) || throw(UnsortedException())
    lo, hi = 1, length(v)
    if algo == :recursive
        binarysearch_rec(key, v, lo, hi)
    elseif algo == :iterative
        binarysearch_iter(key, v, lo, hi)
    end
end

function binarysearch_rec(key, v::AbstractVector, lo::Integer, hi::Integer)
    i, j = lo, hi
    if i <= j
        mid = midpoint(i, j)
        if key < v[mid] 
            binarysearch_rec(key, v, i, mid-1)
        elseif key > v[mid]
            binarysearch_rec(key, v, mid+1, j)
        else
            return mid
        end
    else
        return -1
    end
end

function binarysearch_iter(key, v::AbstractVector, lo::Integer, hi::Integer)
    i, j = lo, hi
    while i <= j
        mid = midpoint(i, j) 
        if key < v[mid]
            j = mid - 1
        elseif key > v[mid]
            i = mid + 1
        else
            return mid
        end
    end
    return -1
end

# Binary search tree

mutable struct BinaryNode{T} 
    val::T
    numnodes::Int
    left::Union{BinaryNode{T}, Nothing}
    right::Union{BinaryNode{T}, Nothing}

    # root constructor
    BinaryNode(val::T) where T = new{T}(val, 1, nothing, nothing)
end

abstract type SearchTree end

"""
Binary search tree.
"""
struct BinarySearchTree{T} <: SearchTree 
    root::BinaryNode{T}
end

function BinarySearchTree(v::AbstractVector)
    !isempty(v) || error("empty vector")
    
    bst = BinarySearchTree(BinaryNode(v[1]))
    for val in v[2:end]
        put!(bst, val)
    end

    return bst
end

Searching.size(::Nothing) = 0
Searching.size(node::BinaryNode{T}) where T = node.numnodes
Searching.size(tree::BinarySearchTree{T}) where T = Searching.size(tree.root)

function updatesize!(node)
    node.numnodes = Searching.size(node.left) + Searching.size(node.right) + 1
    nothing
end

"""
    put!(tree, val)

Insert a value into the binary search tree.

    put!(node, val)

Insert a value following the node.
"""
function put!(node::BinaryNode{T}, val::T) where T
    if val < node.val
        node.left = put!(node.left, val)
    elseif val > node.val 
        node.right = put!(node.right, val)
    else
        node.val = val
    end
    updatesize!(node)
    return node
end

put!(tree::BinarySearchTree{T}, val::T) where T = BinarySearchTree(put!(tree.root, val))
put!(::Nothing, val) = BinaryNode(val)

"""
    get(node, val)

Search for the node by value. Return `nothing` if not found.
"""
function get(node::BinaryNode, val)
    if val < node.val
        return get(node.left, val)
    elseif val > node.val
        return get(node.right, val)
    else
        return node
    end
end

get(tree::BinarySearchTree, val) = get(tree.root, val)
get(::Nothing, val)= nothing

"""
    maximum(BinarySearchTree)

Return node with maximum value.
"""
function maximum(node) 
    isnothing(node) && return nothing

    while !isnothing(node.right)
        node = node.right
    end

    return node
end

maximum(tree::BinarySearchTree) = maximum(tree.root)

function minimum(node)
    isnothing(node) && return nothing

    while !isnothing(node.left)
        node = node.left
    end

    return node
end

minimum(tree::BinarySearchTree) = minimum(tree.root)

"""
    rank(node, val)

Find the rank for value `val`: number of nodes with values less 
than `val` + 1.
"""
function rank(node::BinaryNode, val)
    if val < node.val
        valrank = rank(node.left, val)
    elseif val > node.val
        valrank = rank(node.right, val) + Searching.size(node.left) + 1
    else
        valrank = Searching.size(node.left)
    end

    return valrank
end

rank(::Nothing, val) = 0
rank(tree::BinarySearchTree, val) = rank(tree.root, val) + 1

"""
    select(node, rankval)

Select the node with ranking `rankval`. 
We assume rankval is in range between 1 to the size of the tree.
"""
function select(node::BinaryNode, valrank)
    noderank = Searching.size(node.left) + 1
    if valrank < noderank
       node = select(node.left, valrank)
    elseif valrank > noderank
       node = select(node.right, valrank - noderank)
    else
        # found position with `valrank`
        # return node
    end

    return  node
end

select(::Nothing, valrank) = nothing
select(tree::BinarySearchTree, valrank) = select(tree.root, valrank)

"""
    deletemin!(node)
Find the minimum node and delete it from the tree.
We can find the minimum node by move left recursively, until 
we find the node with no left child.

"""
function deletemin!(node)
    if isnothing(node.left)
        node = node.right
    else
        node.left = deletemin!(node.left)
    end
    updatesize!(node)
    return node
end

deletemin!(tree::BinarySearchTree) = BinarySearchTree(deletemin!(tree.root))

"""
    copyval!(tonode, fromnode)

Copy values from `fromnode` to `tonode`.
"""
function copyval!(tonode::BinaryNode, fromnode::BinaryNode)
    tonode.val = fromnode.val 
    tonode.numnodes = fromnode.numnodes
    nothing
end

"""
    delete!(tree, val)

Delete the node from the binary search tree.
It returns the successor of the deleted node.

    delete!(node, val)

Delete the node with value `val` in the substree 
starts with `node`.
"""
function delete!(node, val)
    if isnothing(node)
        return nothing
    elseif val < node.val
        node.left = delete!(node.left, val)
    elseif val > node.val
        node.right = delete!(node.right, val)
    # found the value and node has only one child node
    elseif isnothing(node.right)
        node = node.left
    elseif isnothing(node.left)
        node = node.right
    # node has both left and right child nodes
    else 
        # replace current node with its successor
        # here I cheated using copy value (without links), 
        # instead of replacing the object
        # This way we keep the original reference to make the
        # side effect consistent: the returned node and original "node"
        # refers to the same object (a substree starting from that node)
        copyval!(node, minimum(node.right)) 

        # left child remain the same 
        # delete the the successor from the right subtree
        # link successor the right child to the pruned subtree
        node.right = deletemin!(node.right)
    end

    updatesize!(node)
    return node
end

delete!(tree::BinarySearchTree, val) = BinarySearchTree(delete!(tree.root, val))

end # module