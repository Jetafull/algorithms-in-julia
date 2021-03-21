module HeapQueueModule

import Base.popfirst!

export Max, 
       Min, 
       MinHeapQueue,
       MaxHeapQueue,
       IndexedMinHQ,
       IndexedMaxHQ,
       push!,
       decreasekey!,
       heapify, 
       heapify!,
       heapsort!
       popfirst!

    
abstract type HeapType end
struct MinHeap <: HeapType end
struct MaxHeap <: HeapType end
const Min = MinHeap()
const Max = MaxHeap()

abstract type AbstractHeapQueue end

mutable struct HeapQueue{H<:HeapType, T} <: AbstractHeapQueue
    arr::Vector{T}
    n::Int

    function HeapQueue{H, T}() where {T,H<:HeapType}
        new{H, T}(Vector{T}[], 0)
    end
end

MinHeapQueue{T} = HeapQueue{MinHeap, T}
MaxHeapQueue{T} = HeapQueue{MaxHeap, T}

Base.getindex(hq::HeapQueue, i::Int) = hq.arr[i]
Base.setindex!(hq::HeapQueue, v, i::Int) = hq.arr[i] = v
Base.length(hq::AbstractHeapQueue) = hq.n
Base.isempty(hq::AbstractHeapQueue) = hq.n == 0

"""
Test the heap dominance between i and j.
"""
testheapdom(hq::MinHeapQueue, i, j) = hq[i] < hq[j]
testheapdom(hq::MaxHeapQueue, i, j) = hq[i] > hq[j]

function Base.push!(hq::HeapQueue, transaction)
    push!(hq.arr, transaction)
    hq.n += 1
    swim!(hq, hq.n)
end

function Base.push!(hq::HeapQueue, transactions...)
    for transaction in transactions
        push!(hq, transaction)
    end
end

function popfirst!(hq::HeapQueue)
    last_transaction = hq.arr[1]
    swap!(hq, 1, hq.n)
    hq.n -= 1
    sink!(hq, 1)
    Base.pop!(hq.arr)
    last_transaction
end

function swap!(hq::HeapQueue, i, j) 
    hq[i], hq[j] = hq[j], hq[i]
    nothing
end

"""
Move node from bottom to top of the tree.
"""
function swim!(hq::AbstractHeapQueue, k::Int)
    # if k-th node dominates its parent, 
    # swap with its the parent node
    while k > 1 && testheapdom(hq, k, div(k,2))
        swap!(hq, div(k,2), k)
        k = div(k,2) 
    end
end

function sink!(hq::AbstractHeapQueue, k::Int, n::Int)
    while 2*k <= n
        # child node to swap with, start with the left child
        j = 2*k
        # test whether the right child dominates he left child  
        # if so, swap with the right child node
        if j < n && testheapdom(hq, j+1, j)
            j += 1
        end
        # now j-th node is the most dominant node in child-level
        # if parent node dominates j-th node, break
        # if not, then swap with j-th node, in this way we still 
        # maintain the heap invariance
        if testheapdom(hq, k, j)
            break
        else
            swap!(hq, k, j)
            # move k to next child level
            k = j
        end
    end
end

sink!(hq::AbstractHeapQueue, k::Int) = sink!(hq, k, hq.n)

function heapify(v::Vector{T}, ::H=Min) where {T,H<:HeapType} 
    hq = HeapQueue{H, T}()
    for elem in v
        push!(hq, elem)
    end
    hq
end

function heapify!(v::Vector{T}, ::H=Min) where{T,H<:HeapType}
    n = length(v)
    hq = HeapQueue{H, T}()
    hq.arr = v
    hq.n = n 
    # we start from middle because the right half index will be 
    # the last level and there will be no where to sink
    for i in div(n, 2):-1:1
        sink!(hq, i, n)
    end
    hq
end

"""
In-place heapsort. It is similar to `popfirst!` and the only difference is
we don't pop the array but manipulate the index. 
"""
function heapsort!(v::Vector{T}) where {T}
    hq = heapify!(v, Max)
    n = hq.n
    while n > 1
        swap!(hq, 1, n)
        n -= 1
        sink!(hq, 1, n)
    end
    v
    nothing
end

# function naive_heapsort!(v::Vector{T}) where {T}
#     hq = heapify!(v)
#     result = T[]
#     while !isempty(hq)
#         push!(result, pop!(hq))
#     end
#     result
# end

# Indexed heap queue
mutable struct IndexedHeapQueue{H<:HeapType, T} <: AbstractHeapQueue
    keys::Vector{Union{T, Nothing}}
    pq::Vector{Union{Int,Nothing}}
    qp::Vector{Union{Int,Nothing}}
    n::Int
    capacity::Int

    IndexedHeapQueue{H, T}(capacity::Int) where {H<:HeapType,T} = new(
        Vector{Union{T, Nothing}}(nothing, capacity),
        Vector{Union{Int, Nothing}}(nothing, capacity),
        Vector{Union{Int, Nothing}}(nothing, capacity),
        0, capacity
    ) 
end

IndexedMinHQ{T} = IndexedHeapQueue{MinHeap, T}
IndexedMaxHQ{T} = IndexedHeapQueue{MaxHeap, T}

Base.getindex(hq::IndexedHeapQueue, keyindex::Int) = hq.keys[keyindex]
pqtokey(hq::IndexedHeapQueue, i) = hq.keys[hq.pq[i]]
keytopq(hq::IndexedHeapQueue, keyindex) = hq.qp[keyindex]

testheapdom(hq::IndexedMinHQ, i, j) = pqtokey(hq, i) < pqtokey(hq, j) 
testheapdom(hq::IndexedMaxHQ, i, j) = pqtokey(hq, i) > pqtokey(hq, j) 

function swap!(hq::IndexedHeapQueue, i, j) 
    hq.pq[i], hq.pq[j] = hq.pq[j], hq.pq[i]
    # key indices in i and j has been swapped in pq
    # Since pq[i] ≡ pq[j] ≡ keyindex_j,
    # so we need to assign keyindex_j to i in `qp`
    hq.qp[hq.pq[i]], hq.qp[hq.pq[j]] = i, j 
    nothing
end

contains(hq::IndexedHeapQueue, keyindex) = !isnothing(hq.qp[keyindex])

struct NoSuchElementException <: Exception 
    msg::String
end

"""
    popfirst!(hq)

Remove the most dominant item in the queue and returns the associated
index in `hq.keys`.
"""
function popfirst!(hq::IndexedHeapQueue)
    hq.n == 0 && throw(NoSuchElementException("priority queue underflow"))

    # pop the minimum
    firstkeyindex = hq.pq[1]
    swap!(hq, 1, hq.n)
    hq.n -= 1
    sink!(hq, 1)
    @assert firstkeyindex == hq.pq[hq.n+1]

    # clean up
    hq.pq[hq.n+1] = nothing
    hq.qp[firstkeyindex] = nothing
    firstkey = hq.keys[firstkeyindex]
    hq.keys[firstkeyindex] = nothing

    firstkeyindex, firstkey
end

"""
    insert!(hq, i, key)

Associate key with index `keyindex` in the heap queue.
"""
function insert!(hq::IndexedHeapQueue, keyindex::Int, key)
    contains(hq, keyindex) && throw(ArgumentError("index $keyindex is already in the heap queue"))
    hq.n += 1
    hq.qp[keyindex] = hq.n 
    hq.keys[keyindex] = key
    hq.pq[hq.n] = keyindex 
    swim!(hq, hq.n)

    nothing
end

"""
Change the key value at `keyindex`. If it is a MinHeapQueue, then 
make sure the updated key is smaller then existing one. 
"""
function changekey!(hq::IndexedHeapQueue, keyindex::Int, key)
    contains(hq, keyindex) || throw(NoSuchElementException("index $keyindex is not in the heap queue")) 
    checkkey(hq, keyindex, key) 
    hq.keys[keyindex] = key
    # a more dominant value updated, should swim! to maintain 
    # heap invariance
    swim!(hq, keytopq(hq, keyindex))
    nothing
end

decreasekey!(hq::IndexedMinHQ, keyindex::Int, key) = changekey!(hq, keyindex, key)

struct InvalidKeyChangeError <: Exception 
    msg::String
end
function checkkey(hq::IndexedMinHQ, keyindex, key)  
    hq.keys[keyindex] <= key && throwerror(hq)
end
throwerror(hq::IndexedMinHQ) = throw(InvalidKeyChangeError("new key should be smaller than current key")) 
throwerror(hq::IndexedMaxHQ) = throw(InvalidKeyChangeError("new key should be greater than current key")) 

end # module
