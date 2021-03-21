mutable struct Node
    depth
    is_leaf
    prediction
    feature_index
    thresh
    left
    right
end 

function Node(depth, is_leaf, prediction)
    Node(depth, is_leaf, prediction, nothing, nothing, nothing, nothing)
end
struct WeightedDecisionStump
    root_node
    
    function WeightedDecisionStump(X, y, weights)
        node = Node(1, false, nothing)
        feature_index, thresh, left_prediction = split(X, y, weights)
        node.feature_index = feature_index
        node.thresh = thresh
        node.left = Node(2, true, left_prediction)
        node.right = Node(2, true, -left_prediction)

        new(node)
    end
end

function split(X, y, weights)
    best_feature = 1
    best_thresh = minimum(X[:, 1])
    best_left_prediction = 1
    error_val = Inf

    for (feature_index, feature_array) in enumerate(eachcol(X))
        sorted_indices = sortperm(feature_array)
        X_sorted = X[sorted_indices, :]
        y_sorted = y[sorted_indices]
        weights_sorted = weights[sorted_indices]
        predictions = zeros(length(y_sorted))

        for left_prediction = (-1, 1) 
            for i = 1:(length(y_sorted) - 1)
                predictions[1:i] .= left_prediction
                predictions[i+1:end] .= -left_prediction
                temp_error_val = misclassification_rate(predictions, y_sorted, weights_sorted)
                if temp_error_val < error_val
                    error_val = temp_error_val
                    best_feature = feature_index
                    best_thresh = (X_sorted[i, feature_index] + X_sorted[i+1, feature_index]) / 2
                    best_left_prediction = left_prediction
                end
            end
        end
    end

    return best_feature, best_thresh, best_left_prediction
end

function misclassification_rate(predictions, y, weights)
    sum((predictions .!= y) .* weights)
end

struct AdaBoostModel 
    num_models
    models
    alphas

    function AdaBoostModel(X, y, num_models)
        # AdaBoost model requires labels being -1 or 1
        @assert Set(y) == Set([-1, 1])
        weights = fill(1.0 / length(y), length(y)) 
        models = WeightedDecisionStump[]
        alphas = Float64[]

        for _ = 1:num_models
            model = WeightedDecisionStump(X, y, weights)
            predictions = predict(model, X)
            mistakes = predictions .!= y
            error = sum(weights .* mistakes) / sum(weights)
            alpha = error == 0.0 ? 1.0 : log((1-error)/error)
            weights = weights .* exp.(alpha .* mistakes) 
            weights = weights ./ sum(weights)

            push!(models, model)
            push!(alphas, alpha)
        end

        new(num_models, models, alphas)
    end
end

function predict(model::WeightedDecisionStump, X)
    root_node = model.root_node
    feature_index = root_node.feature_index
    thresh = root_node.thresh
    predictions = zeros(size(X)[1])

    left_indices = X[:, feature_index] .< thresh 
    predictions[left_indices] .= root_node.left.prediction
    predictions[.!left_indices] .= root_node.right.prediction
    return predictions
end

function predict(model::AdaBoostModel, X)
    models = model.models
    alphas = model.alphas
    predictions = zeros(size(X)[1])
    votes = zeros(size(X)[1]) 

    for (alpha, model) in zip(alphas, models)
        predictions = predict(model, X)
        votes .+= alpha.*predictions
    end

    return sign.(sum(votes, dims=2))
end


