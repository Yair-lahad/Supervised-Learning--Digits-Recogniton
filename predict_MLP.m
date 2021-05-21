function [Y] = predict_MLP(Net, X)
%PREDICT_MLP Get the output of a MLP for a given dataset

% Get the MLP's output for the given set
s = X;                         	% set the in[ut
for l = 1:length(Net)        	% forward pass
    % TODO 4: Add a bias neuron
    s(size(s,1)+1,:)=ones(1,size(s,2));
    s = Net(l).g(Net(l).W * s);	% get next layer's activities
end
Y = s;                         	% get the output

end

