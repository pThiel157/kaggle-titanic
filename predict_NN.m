function out = predict_NN(X, nn_params, input_layer_size, ...
                                    hidden_layer_size, ...
                                    num_labels)

threshold = 0.5;


Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);

A1 = [ones(m,1) X];
Z2 = A1 * Theta1';
A2 = [ones(m,1) sigmoid(Z2)];
Z3 = A2 * Theta2';
A3 = sigmoid(Z3);

out = A3 > threshold;


end
