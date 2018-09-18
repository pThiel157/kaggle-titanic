% d = fileread("data/train.csv");
% d = strsplit(d, "\n");
% for i = 1:length(d)
%   B(i,:) = strtrim(strsplit(d{i}, ","));
% end
% d = B;

d = csvread("data/train.csv");
d = d(2:end,:);

age    = d(:,7);
pclass = d(:,3);
sibSp  = d(:,8);
fare   = d(:,11);
% sex    = d(:,6)
y      = d(:,2);

X = [age pclass sibSp fare];

% Get the rows where age is undefined:
I = age == 0;

% Filter out all those rows in both the age vector and the survival vector:
X = X(~I, :);
y = y(~I);


% % Visualize the data:
% plot(age, y, 'rx', 'MarkerSize', 20, 'LineWidth', 1);
% xlabel('Age (x)');
% ylabel('Survival (y)');

% Set up X and initial placeholder theta:
X = mapFeature(X);
initial_theta = zeros(size(X,2), 1);

% Set up neural net stuff if we want to:
X = X(:, 2:end); % Get rid of the bias column since we'll add that in later
input_layer_size  = size(X, 2);
hidden_layer_size = 25;          % 25 hidden units
cost_grad = @(nn_params, X, y, lambda) cost_grad_NN(nn_params, input_layer_size, hidden_layer_size, 1, X, y, lambda);
predict   = @(X, theta) predict_NN(X, theta, input_layer_size, hidden_layer_size, 1);
initial_theta = [rand((1 + input_layer_size) * hidden_layer_size, 1) ; rand(hidden_layer_size + 1, 1)]; % Theta1 and Theta2


% Split X and y into train and cross-verification sets:
delim = round((0.8 * size(X,1)));
X_train = X(1:delim, :);
X_cv    = X(delim:end, :);
y_train = y(1:delim);
y_cv    = y(delim:end);


% Set options for fminunc:
options = optimset('GradObj', 'on', 'MaxIter', 10);

% Set lambda:
lambda = 0;

% A quick test for sanity:
% disp("Running computeNumericalGradient...")
% [_, grad] = cost_grad(initial_theta, X_train, y_train, lambda, false);
% costFunc = @(p) cost_grad(p, X_train, y_train, lambda, false);
% numGrad = computeNumericalGradient(costFunc, initial_theta);
% disp([grad numGrad])
% disp('The above should be similar')

% Run fminunc (optimize the cost func):
disp("Training model...")
[theta, cost_train] = fminunc(@(t)(cost_grad(t, X_train, y_train, lambda, false)), initial_theta, options);
% [theta] = trainNeuralNet(X_train, y_train, lambda, input_layer_size, hidden_layer_size, 1);

% % Another quick gradient check for sanity:
% [_, grad] = cost_grad(theta, X_train, y_train, lambda, false);
% costFunc = @(p) cost_grad(p, X_train, y_train, lambda, false);
% numGrad = computeNumericalGradient(costFunc, theta);
% disp([grad numGrad])
% disp('The above should also be similar')


% disp('Theta after optimization:')
% disp(theta)

cost_train = cost_grad(theta, X_train, y_train, lambda, true);
disp('Cost after optimization:')
disp(cost_train)

% Run on our cross-verification set and get the cost:
[cost_cv, grad] = cost_grad(theta, X_cv, y_cv, lambda, true);

disp('CV cost:')
disp(cost_cv)

% Get F1 score:
f1_score = calc_f1(X_cv, y_cv, theta, input_layer_size, hidden_layer_size, 1);

disp("F1 score:")
disp(f1_score)


% Plot learning curve stuff:
max_m = 50;
[error_train, error_val] = learningCurve(X_train, y_train, X_cv, y_cv, lambda, max_m, input_layer_size, hidden_layer_size, 1);
plot(1:max_m, error_train, 1:max_m, error_val);
title('Learning curve')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
axis([0 max_m 0 4*max(cost_train, cost_cv)])



% Make our predictions for the test set and output a csv file:

test_data = csvread("data/test.csv");

ids = test_data(:,1);

age_test    = test_data(:,6);
pclass_test = test_data(:,2);
sibSp_test  = test_data(:,7);
fare_test   = test_data(:,10);

test = [age_test pclass_test sibSp_test fare_test];

test = mapFeature(test);

% Get rid of bias column if we're doing NN stuff:
test = test(:, 2:end);

p = predict(test, theta);
% p = predict_NN(test, theta, input_layer_size, hidden_layer_size, 1);

out = [ids p];

csvwrite("out.csv", out);
