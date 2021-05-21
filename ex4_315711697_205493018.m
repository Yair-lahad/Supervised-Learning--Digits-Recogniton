    
    % A MLP learning to classify MNIST handwritten digits

    % Neriya Mizrahi (315711697)
    % Yair Lahad (205493018)

    % Computation and cognition undergrad - ex3
    % See PDF document for instructions

    clear; close all; clc;

%% Get the data

data_dir = 'Data';  % Data directory

% Download the data (if necessary) and load it
[train_images, train_labels, ...
 test_images, test_labels] = MNIST.get_data(data_dir);

%% Split the training data into training and validation sets

% Define the validation set
valid_percent   = 0.2;	% percentage of training data to be used as 
                        % validation set

% TODO 1: Randomly split the training set into training an validation sets.
%         You should create new variables named `valid_images` and
%         `valid_labels`, structured exactly as `train_images` and
%         `train_labels` (respectively), but with different number of
%         samples. You should also remove the samples chosen as the
%         validation set from the training set. 
%valid_images = zeros(size(train_images));
%valid_labels = zeros(size(train_labels));
keepIm =randperm(size(train_images,3),0.2.*(size(train_images,3)));
valid_images=train_images(:,:,keepIm);
valid_labels=train_labels(keepIm,:);
train_images(:,:,keepIm)=[];
train_labels(keepIm,:)=[];

%% Preprocess the data

% Rehsape images into vectors, normalize pixel values to [0,1], convert
% labels to 1-hot encoding
% TODO 2: Explain the importance of the images reshape and the labels
%         encoding as 1-hot (write your answer in the PDF file). 
[X_train, Y_train]  = MNIST.preprocess(train_images, train_labels);
[X_valid, Y_valid]	= MNIST.preprocess(valid_images, valid_labels);
[X_test,  Y_test]   = MNIST.preprocess(test_images,  test_labels);

%% Build and initialize the MLP

% TODO 5: Change the network's architecture and learning parameters as you 
%         see fit. Try to get the accuracy as high as you can. 
%         The parameters you should tweak are: 
%         `N`           NOTE: Keep the number of neurons of the first 
%                             (input) and last (output) layers. 
%         `g_funcs`     NOTE: The activation fuctions can be different for
%                             different layers. Try it out!
%         `eta`
%         `batch_size`
%         NOTE: Please keep the overall runtime under 10 minutes. It should
%               be more than enough. 

% Network structure
N = [784, 100, 10];	% number of neurons per layer
L = length(N) - 1;	% number of layers

assert(N(1) == size(X_train, 1), ...
       "The number of input neurons must match the samples\' input dimension. ")
assert(N(end) == size(Y_train, 1), ...
       "The number of output neurons must match the samples\' output dimension. ")

% Activation functions (per layer)
% NOTE: `g_funcs` can be also defined manually. For example, if `N` defines
%       3 layers (other than the input layer), then `g_funcs` can be 
%       defined as: 
%       g_funcs  = {@ActFuncs.Tanh, @ActFuncs.ReLU, @ActFuncs.Sigmoid};
%       If you choose to define g_funcs this way, make sure that the number
%       of elements corresponds to the length of `N`. 
g_funcs         = cell(1, L);
[g_funcs{1:L}]	= deal(@ActFuncs.Sigmoid);         % all but last layer
assert(length(N) == length(g_funcs) + 1, ...    % sanity check
       'The number of activation functions and the number of layers mismatch. ')

% Initialize the layers' weights and activation functions
% TODO 4: Change the dimensions of W to account for a bias neuron
Net = arrayfun(@(n, n1, g) struct('W', 0.1*randn(n1, n+1), ...	% weights
                                  'g', g), ...                  % activation function
               N(1:L), N(2:L + 1), g_funcs);

% Training parameteres
eta         = 3.0;	% learning rate
n_epochs    = 50;	% number of training epochs
batch_size  = 10;	% mini-batch size

%% Train the MLP

% Training statsitics per epoch
history = cell(n_epochs + 1, 1);

% Get metrics for the training and validation sets
[t_err, t_acc] = predict_evaluate_MLP(Net, X_train, Y_train, train_labels);
[v_err, v_acc] = predict_evaluate_MLP(Net, X_valid, Y_valid, valid_labels);

% Save statistics history
history{1} = struct('train_err', t_err, ... % training error
                    'train_acc', t_acc, ... % training accuracy
                    'valid_err', v_err, ... % validation error
                    'valid_acc', v_acc);    % validation accuracy

% Command log
fprintf('Epoch %d/%d, error = %0.3g, accuracy = %2.1f%%. \n', ...
        0, n_epochs, v_err, 100*v_acc);

% Loop over epochs
for epoch = 1:n_epochs
    
    % Start taking time
    tic
    
    % Get a random order of the samples
    perm = randperm(size(X_train, 2));
    
    % Loop over all training samples (in mini-batches)
    for batch_start = 1:batch_size:length(perm)
        
        % Get the samples' indices for the current batch
        batch_end = min(batch_start + batch_size - 1, length(perm));
        batch_ind = batch_start:batch_end;
        
        % Get the current batch data
        X   = X_train(:, perm(batch_ind));
        Y0  = Y_train(:, perm(batch_ind));
        
        % Temporary neurons activities per layer
        s = cell(L + 1, 1);
        
        % Forward pass
        % NOTE: The layers' activities and derivatives of the current 
        %       mini-batch are stored in matrices, such that each column
        %       represents a different sample. 
        s{1} = struct('act', X, ...                     % set the input
                      'der', zeros(size(X)));           % for completeness
        for l = 1:L
            % TODO 4: Add a bias neuron
            s{l}.act(size(s{l}.act,1)+1,:)=ones(1,size(s{l}.act,2));
            [g, gp] = Net(l).g(Net(l).W * s{l}.act);	% get next layer's activities 
                                                        % (and derivatives)
            s{l+1}  = struct('act', g, ...
                             'der', gp);                % save results per layer
        end
        Y = s{L + 1}.act;                               % get the output
        
        % Back propagation
        delta = (Y - Y0).*s{L + 1}.der;
        for l = L:-1:1
            % TODO 3: Use the learning rules derived in class to get the
            %         change in weights `dW` and the previous layer's 
            %         `delta`. 
            %         NOTE: Although you work on a mini-batch of several
            %               samples, you are NOT ALLOWED to use any loops!
            %               This task should be solved using matlab's
            %               matrix operations syntax. 
            %dW = -(eta/batch_size).*(delta * s{l}.act');
            %delta=((Net(l).W)'*delta).*s{l}.der;
            %Net(l).W=Net(l).W+dW;
            % TODO 4: Modify the updates of `delta` to account for the 
            %         added bias neuron. 
            dW = -(eta/batch_size).*(delta * s{l}.act');         % get the weights update
            delta = ((Net(l).W(:,(1:end-1)))'*delta).*s{l}.der;  % update delta
            Net(l).W = Net(l).W + dW;	% update the weights
        end
        
    end % mini-batches loop

    % Get metrics for the training and validation sets
    [t_err, t_acc] = predict_evaluate_MLP(Net, ...
                                          X_train, Y_train, train_labels);
    [v_err, v_acc] = predict_evaluate_MLP(Net, ...
                                          X_valid, Y_valid, valid_labels);
    
    % Save statistics history
    history{epoch + 1} = struct('train_err', t_err, ... % training error
                                'train_acc', t_acc, ... % training accuracy
                                'valid_err', v_err, ... % validation error
                                'valid_acc', v_acc);    % validation accuracy
    
    % Command log
    fprintf('Epoch %d/%d, error = %0.3g, accuracy = %2.1f%%, time: %.1f[sec]. \n', ...
            epoch, n_epochs, v_err, 100*v_acc, toc);
    
end % epochs loop

%% Get results for the test set
Y_out                   = predict_MLP(Net, X_test);	% MLP's output
Y_labels                = output2labels(Y_out);     % MLP's labels
[test_err, test_acc]    = evaluate_MLP(Y_out, Y_test, test_labels);
fprintf('Test: error = %0.3g, accuracy = %2.1f%%. \n', ...
        test_err, 100*test_acc);

%% Plots

% Training history plot
figure('Name', 'Training history', ...
       'Units', 'normalized', ...
       'Position', [0.035, 0.25, 0.36, 0.5]);
Plots.history(history);
title(sprintf('Test accuracy: %2.1f%%', 100*test_acc));

% Correct classification examples
figure('Name', 'Correct examples', ...
       'Units', 'normalized', ...
       'Position', [0.4, 0.52, 0.6, 0.4]);
corr_idx = Y_labels == test_labels;         % Extract only correct samples
Plots.results(test_images(:, :, corr_idx), ...
             Y_labels(corr_idx), ...
             test_labels(corr_idx));
suptitle('Examples of the MLP''s correct classifications');

% Errors examples
figure('Name', 'Error examples', ...
       'Units', 'normalized', ...
       'Position', [0.4, 0.04, 0.6, 0.4]);
err_idx = Y_labels ~= test_labels;          % Extract only error samples
Plots.results(test_images(:, :, err_idx), ...
             Y_labels(err_idx), ...
             test_labels(err_idx));
suptitle('Examples of the MLP''s errors');
