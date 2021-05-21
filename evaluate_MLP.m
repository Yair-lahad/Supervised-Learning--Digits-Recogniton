function [err, acc] = evaluate_MLP(Y, Y0, labels)
%EVALUATE_MLP Evaluate a MLP
%   Get the squared error and accuracy of a MLP on a given dataset. 

% Get statistics
err         = mean(sum(0.5.*(Y0 - Y).^2));      % squared error
Y_labels	= output2labels(Y);
acc         = mean(double(Y_labels == labels));	% accuracy

end

