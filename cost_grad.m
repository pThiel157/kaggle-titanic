function [J, grad] = cost_grad(theta, X, y, lambda, isCV)

  m = length(y);

  grad = zeros(size(theta, 1), 1);


      % (1/m) * sum((-y .* log(sigmoid(X * theta))) - ((1-y) .* log(1 - sigmoid(X * theta))));
  if isCV
    J = (1/m) * sum((-y .* log(sigmoid(X * theta))) - ((1-y) .* log(1 - sigmoid(X * theta))));
  else
    J = (1/m) * sum((-y .* log(sigmoid(X * theta))) - ((1-y) .* log(1 - sigmoid(X * theta)))) + lambda/(2*m) * sum(theta(2:end).^2);
    % disp(J)
  end

  % disp(size(X * theta))
  % disp(size(sigmoid(X * theta) - y))
         % (1/m) * sum((sigmoid(X * theta) - y) .* X)'
  % disp(size(lambda/m * theta(2:end)))


  grad(1)     = (1/m) * sum(sigmoid(X * theta) - y)';
  grad(2:end) = (1/m) * sum((sigmoid(X * theta) - y) .* X(:,2:end))' + lambda/m * theta(2:end);

  % disp(grad)
  % disp(theta)
end
