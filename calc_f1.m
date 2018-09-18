function f1 = calc_f1(X, y, theta, input_layer_size, hidden_layer_size, num_labels)

preds = predict_NN(X, theta, input_layer_size, hidden_layer_size, num_labels);

adder = preds + y;
subtr = preds - y;
t_pos = length(find(adder == 2));
f_pos = length(find(subtr == 1));
f_neg = length(find(subtr == -1));

prec = t_pos / (t_pos + f_pos)
rec  = t_pos / (t_pos + f_neg)

f1 = 2 * prec * rec / (prec + rec);

end
