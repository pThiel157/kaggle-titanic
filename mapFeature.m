function out = mapFeature(X)

  n = size(X, 2);

  degree = 1;
  fprintf("Mapping X to degree %i\n", degree);

  out = ones(size(X(:,1)), 1);
  for i = 1:degree
    out(:, end+1:end+n) = X.^i;
  end

  out(:, 2:end) = X = normalize(out(:, 2:end));
end
