function [labels] = output2labels(Y)
%OUTPUT2LABELS Convert a MLP's classification output to MNIST labels

[~, labels]	= max(Y);       % Get the index of the neuron with the maximal activity
labels      = labels' - 1;	% Correct for matlab indices (start with 1)

end

