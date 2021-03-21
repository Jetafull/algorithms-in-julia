using StatsBase: mode, mean

abstract type TreeNode end

mutable struct ClassificationNode <: TreeNode
    depth::Int
    isleaf::Bool
    regionid::Union{Nothing, Int}
    prediction::Union{Nothing, Int}
    featureindex::Union{Nothing, Int}
    thresh::Union{Nothing, Float64}
    leftnode::Union{Nothing, TreeNode}
    rightnode::Union{Nothing, TreeNode}

    function ClassificationNode(depth)
        new(depth, false, nothing, nothing, nothing, nothing, nothing, nothing)
    end

end

struct TreeStoppingRule
    maxdepth::Int
    minsamplesplit::Int
end

"""
Tree node for regression task. 
"""
mutable struct RegressionNode <: TreeNode
    depth::Int
    isleaf::Bool
    regionid::Union{Nothing, Int}
    prediction::Union{Nothing, Float64}
    featureindex::Union{Nothing, Int}
    thresh::Union{Nothing, Float64}
    leftnode::Union{Nothing, TreeNode}
    rightnode::Union{Nothing, TreeNode}

    function RegressionNode(depth)
        new(depth, false, nothing, nothing, nothing, nothing, nothing, nothing)
    end

end

abstract type DecisionTree end

struct ClassificationTree <: DecisionTree
    rootnode::TreeNode
    lossfunc::Function
    stoppingrule::TreeStoppingRule
    numregions::Int

    function ClassificationTree(X, y, lossfunc, stoppingrule)
        rootnode, lastregionid = fittree!(ClassificationNode(1), X, y, lossfunc, stoppingrule)
        new(rootnode, lossfunc, stoppingrule, lastregionid)
    end
end

struct RegressionTree <: DecisionTree
    rootnode::TreeNode
    lossfunc::Function
    stoppingrule::TreeStoppingRule
    numregions::Int

    function RegressionTree(X, y, lossfunc, stoppingrule)
        rootnode, lastregionid = fittree!(RegressionNode(1), X, y, lossfunc, stoppingrule)
        new(rootnode, lossfunc, stoppingrule, lastregionid)
    end
end

function RegressionTree(X, y, stoppingrule)
    return RegressionTree(X, y, squaredloss, stoppingrule)
end

function worthsplitting(y, leftindices, currentdepth, stoppingrule)
    rightindices = .!leftindices
    if (currentdepth > stoppingrule.maxdepth || 
        !checkminsamplesplit(sum(leftindices), sum(rightindices), 
                             stoppingrule.minsamplesplit))
        return false
    else
        return true
    end
end

function checkminsamplesplit(leftnum, rightnum, minsamplesplit)
    return leftnum >= minsamplesplit && rightnum >= minsamplesplit
end

function findbestsplit(X, y, lossfunc, minsamplesplit)
    # no split case, loss value is evluated on all observations
    bestfeature = 1
    bestthresh = -Inf
    lossval = lossfunc(y) 

    for (featureindex, featurearray) in enumerate(eachcol(X))
        sortedindices = sortperm(featurearray)
        sortedX = X[sortedindices, :]
        sortedy = y[sortedindices]
        for i in 1:(length(sortedy) - 1)
            templossval = lossfunc(sortedy[1:i], sortedy[i+1:end])
            if (templossval < lossval && checkminsamplesplit(i, length(y)-i, minsamplesplit))
                lossval = templossval
                bestfeature = featureindex
                bestthresh = (sortedX[i, featureindex] + sortedX[i+1, featureindex])/2 
            end
        end
    end

    return bestfeature, bestthresh
end


function split(X, y, lossfunc, minsamplesplit)
    featureindex, thresh = findbestsplit(X, y, lossfunc, minsamplesplit)
    leftindices = X[:, featureindex] .< thresh
    return featureindex, thresh, leftindices
end

function _buildbranch(node::N, X, y, indices, depth) where N<:TreeNode
    selectedX = X[indices, :]
    selectedy = y[indices]
    node = N(depth)
    return selectedX, selectedy, node
end

function predict!(node::ClassificationNode, y)
    node.prediction = mode(y) 
    nothing
end

function predict!(node::RegressionNode, y)
    node.prediction = mean(y)
    nothing
end

function _createyhat(model::RegressionTree, X)
    zeros(Float64, size(X)[1])
end

function _createyhat(model::ClassificationTree, X)
    zeros(Int, size(X)[1])
end

function predict(model::DecisionTree, X)
    yhat = _createyhat(model, X)
    for (i, x) in enumerate(eachrow(X))
        node = model.rootnode
        while !node.isleaf 
            if x[node.featureindex] < node.thresh
                node = node.leftnode
            else
                node = node.rightnode
            end
        end
        yhat[i] = node.prediction
    end
    return yhat
end      


function fittree!(node::TreeNode, X, y, lossfunc, stoppingrule)
    function fit!(node::TreeNode, X, y)
        featureindex, thresh, leftindices = split(X, y, lossfunc, stoppingrule.minsamplesplit)
        nextdepth = node.depth + 1

        if !worthsplitting(y, leftindices, nextdepth, stoppingrule)
            predict!(node, y)
            regionid += 1
            node.regionid = regionid 
            node.isleaf = true
            return node
        else
            node.featureindex = featureindex
            node.thresh = thresh
            rightindices = .!leftindices

            leftX, lefty, leftnode = _buildbranch(node, X, y, leftindices, nextdepth)
            node.leftnode = fit!(leftnode, leftX, lefty)

            rightX, righty, rightnode = _buildbranch(node, X, y, rightindices, nextdepth)
            node.rightnode = fit!(rightnode, rightX, righty)
            return node
        end
    end

    regionid = 0
    fittednode = fit!(node::TreeNode, X, y)
    return fittednode, regionid
end

# Loss functions 
function squaredloss(ys...)
    totalloss = 0.0
    for y in ys
        yhat = mean(y)
        totalloss += sum((y.-yhat).^2)
    end
    return totalloss
end

function _giniindex(y)
    num_ones = float(sum(y.==1))
    total_num = length(y)
    p = num_ones / total_num
    return 2*p*(1-p)
end

function giniindex(ys...)
    totalval = 0.0
    totalnum = sum([length(y) for y in ys])
    for y in ys
        n = float(length(y))
        totalval += n / totalnum * _giniindex(y) 
    end
    return totalval
end
