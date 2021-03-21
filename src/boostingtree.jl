using StatsBase: mean

include("decisiontree.jl")

"""Gradient tree boosting algorithm.

The implmentation follows **Algorithm 10.3** in the *Elements of 
Statistical Learning*. The original implementation of this algorithm called 
MART for "multiple additive regression tree.
"""

# Loss functions
abstract type LossFunction end

struct MeanSquaredError <: LossFunction end 

function (mse::MeanSquaredError)(y, pred)
    mean((y - pred).^2)
end

negativegradient(mse::MeanSquaredError, y, pred) = y - pred

"Initial prediction in case of Meansquared error is global mean."
predict(lossfunc::MeanSquaredError, y) = mean(y)

struct LogLoss <: LossFunction end

"""
LogLoss for binary classification problem.
The implmentation follows the article: 
http://zpz.github.io/blog/gradient-boosting-tree-for-binary-classification/.
Here `pred` is the predicted log-odds and logistic/sigmmoid function 
converted the log-odds to the predicted probability.
"""
function (logloss::LogLoss)(y, pred)
    sum(log.(1+exp.(pred)) - y.*pred)
end

sigmoid(pred) = 1 / (1 + exp(-pred))

gradient(logloss::LogLoss, y, pred) =  sigmoid.(pred) .- y

function hessian(logloss::LogLoss, y, pred)
    p = sigmoid.(pred)
    return p.*(1 .- p)
end
negativegradient(logloss::LogLoss, y, pred) =  -gradient(logloss, y, pred)

function predict(logloss::LogLoss, y)
    numobs = length(y)
    numpos = sum(y)
    return log(numpos/(numobs-numpos))
end

"""
Stopping rule for boosting model.
TODO: 
    - Check the minimal increase between two consecutive model updates.
"""
struct BoostingStoppingRule 
    numtrees::Int
    treestoppingrule::TreeStoppingRule
end

# Gradient boosting model
mutable struct GradientBoostingTrees
    lossfunc::LossFunction
    boostingstoppingrule::BoostingStoppingRule
    trees::Vector{DecisionTree}
    learningrate::Float64
    initialpred::Float64

    function GradientBoostingTrees(X, y, lossfunc, boostingstoppingrule; learningrate=0.1)
        numtrees = boostingstoppingrule.numtrees
        treestoppingrule = boostingstoppingrule.treestoppingrule
        trees = RegressionTree[]

        # Initialize boosting model: f0(x)
        initialpred = predict(lossfunc, y)
        preds = fill(initialpred, length(y))

        for m = 1:numtrees
            # generalized residual (negative gradient) 
            residuals = negativegradient(lossfunc, y, preds) 
            tree = RegressionTree(X, residuals, treestoppingrule) 

            # update predictions on the terminal regions (leaves) in the newly fitted tree
            updateleaves!(lossfunc, tree, X, y, preds)
            newtreepreds = predict(tree, X)
            preds += learningrate * newtreepreds
            push!(trees, tree)
        end

        new(lossfunc, boostingstoppingrule, trees, learningrate, initialpred)
    end

end

"""
    updateleaves!(lossfunc, tree, X)

Update tree leaves with Meansquared error as the loss function.
For Meansquared error we have a closed form solution to minimize the loss
w.r.t the current model: the optimal point the mean of the residual in each 
leave. See https://explained.ai/gradient-boosting/descent.html for details.

Because the regression tree has already fitted the residual, we don't need to
change the leaf value in this case.
"""
function updateleaves!(lossfunc, tree, X, y, preds) end

"""
    updateleaves!(lossfunc::LogLoss, tree, X)

Update tree leaves for log-loss. In general, we don't have a closed form solution. 
We will conduct one step update using Newton's method. To use Newton's method, 
we need gradient, Hessian and regularization terms.
"""
function updateleaves!(lossfunc::LogLoss, tree, X, y, preds) 
    numregions = tree.numregions
    residuals = negativegradient(lossfunc, y, preds)
    hessians = hessian(lossfunc, y, preds)

    regionresidualsums = zeros(numregions)
    regionhessiansums = zeros(numregions)

    for i in 1:size(X)[1]
        regiondid = findregionid(lossfunc, tree, X[i, :])
        regionresidualsums[regiondid] += residuals[i] 
        regionhessiansums[regiondid] += hessians[i]
    end

    regionupdates = regionresidualsums ./ regionhessiansums

    replacepreds!(tree, regionupdates)
end

function findregionid(lossfunc::LogLoss, tree, x) 
    node = tree.rootnode
    while !node.isleaf
        featureindex = node.featureindex
        thresh = node.thresh
        if x[featureindex] < thresh
            node = node.leftnode
        else
            node = node.rightnode
        end
    end

    return node.regionid
end

function replacepreds!(tree::DecisionTree, regionupdates)
    stack = [tree.rootnode]

    while !isempty(stack)
        node = pop!(stack)
        if node.isleaf
            node.prediction = regionupdates[node.regionid]
        else
            @assert !isnothing(node.leftnode)
            @assert !isnothing(node.rightnode)
            push!(stack, node.leftnode)
            push!(stack, node.rightnode)
        end
    end

end

function predict(boostingmodel::GradientBoostingTrees, X)
    lossfunc = boostingmodel.lossfunc
    learningrate = boostingmodel.learningrate
    preds = fill(boostingmodel.initialpred, size(X)[1])

    for tree in boostingmodel.trees 
        preds += learningrate*predict(tree, X)
    end

    preds
end
