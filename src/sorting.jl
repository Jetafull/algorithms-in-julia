# Insertion sort
function insertionsort!(arr, low, high)
    @inbounds for i = (low+1):high 
        j = i
        x = arr[i]
        while j > low && arr[j] < arr[j-1] 
            exch!(arr, j, j-1)
            j -= 1
        end
    end
end

insertsort!(arr) = insertionsort!(arr, 1, length(arr))
const SMALL_THRESHOLD = 20

# Merge sort
function mergesort!(arr)
    mergesort!(arr, 1, length(arr))
end

function mergesort!(arr, low, high)
    if low < high
        middle = Int(floor((low + high) / 2))
        mergesort!(arr, low, middle)
        mergesort!(arr, middle+1, high)
        merge!(arr, low, middle, high)
    end
end

function merge!(arr, low, middle, high)
    leftarr = copy(arr[low:middle])
    rightarr = copy(arr[(middle+1):high])

    i = low
    # compare the first elements between two arrays
    while (!(isempty(leftarr) || isempty(rightarr)))
        if leftarr[1] <= rightarr[1]
            arr[i] = popfirst!(leftarr) 
        else
            arr[i] = popfirst!(rightarr)
        end
        i += 1
    end

    # add remaining elements 
    while !isempty(leftarr)
        arr[i] = popfirst!(leftarr)
        i += 1
    end
    while !isempty(rightarr)
        arr[i] = popfirst!(rightarr)
        i += 1
    end
end

"""
    quicksort!(arr)

Quicksort implemntations based on Algorithm Design Manual and Julia's
Official implementation. 

This implementation is about 10% slow than Julia's `sort!` 
(Quicksort + Insertionsort). The pivot selecting mechanism follows Julia's 
implementaiton that select three points and pick up the middle one, instead of 
shuffling the entire list. One bottleneck is in Julia they have implemented 
a faster float comparison: split the list into negative and positve, and then 
sort with `slt_int`. 

See [my gist](https://gist.github.com/Jetafull/977eaf8a501891b2527a950d777efb5a)
for more details.
"""
function quicksort!(arr::AbstractVector)
    quicksort!(arr, 1, length(arr))
end

# standard implementation: 
# pivot is the final position, recursively sort on both sides of the pivot.
function quicksort!(arr, lo, hi)
    lenarr = hi - lo + 1
    if lenarr <= SMALL_THRESHOLD
        insertionsort!(arr, lo, hi)
    else
        if lo < hi
            firsthigh = partition!(arr, lo, hi)
            quicksort!(arr, lo, firsthigh-1)
            quicksort!(arr, firsthigh+1, hi)
        end
    end
end

lt(x, y) = x < y 

@inline function exch!(arr, i, j)
    @inbounds begin 
        arr[i], arr[j] = arr[j], arr[i]
    end
end

# Baseline version
# Algorithm Design Manual: 4.6
# Need to shuffle the entire list before sorting (uncomment shuffle!(arr))
# function partition!(arr, low, high)
#     pivot = high 
#     firsthigh = low
#     i = low
#     @inbounds while i < high
#         if arr[i] < arr[pivot]
#             exch!(arr, i, firsthigh)
#             firsthigh += 1
#         end
#         i += 1
#     end
#     exch!(arr, pivot, firsthigh)
#     return firsthigh
# end

# Julia offical implementation:
# https://github.com/JuliaLang/julia/blob/master/base/sort.jl
function partition!(v, lo, hi)
    @inbounds begin
        pivot = v[selectpivot!(v, lo, hi)]
        i, j = lo, hi
        while true
            i += 1; j -= 1
            while lt(v[i], pivot); i += 1; end;
            while lt(pivot, v[j]); j -= 1; end;
            i >= j && break
            v[i], v[j] = v[j], v[i]
        end
        v[j], v[lo] = pivot, v[j]
        return j
    end
end

midpoint(lo, hi) = lo + ((hi - lo) >>> 0x01)
# Given 3 locations in an array (lo, mi, and hi), sort v[lo], v[mi], v[hi]) and
# choose the middle value as a pivot
#
# Upon return, the pivot is in v[lo], and v[hi] is guaranteed to be
# greater than the pivot
@inline function selectpivot!(v, lo, hi)
    @inbounds begin
        mi = midpoint(lo, hi)
        # sort v[mi] <= v[lo] <= v[hi] such that the pivot is immediately in place
        if lt(v[lo], v[mi])
            v[mi], v[lo] = v[lo], v[mi]
        end

        if lt(v[hi], v[lo]) # v[lo] > v[hi] and v[lo] > v[mi]
            if lt(v[hi], v[mi])
                v[hi], v[lo], v[mi] = v[lo], v[mi], v[hi]
            else
                v[hi], v[lo] = v[lo], v[hi]
            end
        end

        return lo 
    end
end