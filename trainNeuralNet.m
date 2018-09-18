function [theta] = trainNeuralNet(X, y, lambda, input_layer_size, hidden_layer_size, num_classes)

cost_grad = @(nn_params, X, y, lambda) cost_grad_NN(nn_params, input_layer_size, hidden_layer_size, num_classes, X, y, lambda);

initial_theta = [rand((1 + input_layer_size) * hidden_layer_size, 1) ; rand(hidden_layer_size + 1, 1)]; % Theta1 and Theta2

% Create "short hand" for the cost function to be minimized
costFunction = @(t) cost_grad(t, X, y, lambda);

% Now, costFunction is a function that takes in only one argument
options = optimset('MaxIter', 10, 'GradObj', 'on');

% Minimize using fmincg
theta = fminunc(costFunction, initial_theta, options);

end
