function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



%Part 1: Cost function J

%FeedForward

%add bias to first layer: a(1) aka X
X = [ones(m, 1) X];
% calculate z(2) 
z2 = Theta1*X';
z2 = z2';
% take sigmoid of z(2) to make activity level of hidden layer 1 aka layer 2
a2 = sigmoid(z2);
%add bias to hidden layer
a2 = [ones(size(a2,1), 1) a2];

%calculate
z3 = Theta2*a2';
a3 = sigmoid(z3);

a3 = a3';

%"unroll" y

% ylarge = zeros(size(y,1),num_labels);
% for t = 1:size(ylarge,1)
% ylarge(t,y(t)) = 1;
% end

all_combos = eye(num_labels);
ylarge = all_combos(y,:);
%Apparently don't need line below
% ylarge = [ylarge(:,10),ylarge(:,1:9)];


% after calculating a3 the activity levels at the output, compare a3 to y
% for all the classes and all the examples.

J = (1/m)*sum(sum(-ylarge.*log(a3)-(1-ylarge).*log(1-a3)));


%Part 2: regularised cost function

Jreg = (lambda/(2*m))*(sum(Theta1(:,2:end).^2,'all')+sum(Theta2(:,2:end).^2,'all'));

J = J+Jreg;

%Part 3: backpropagation
Delta1 = 0;

Delta2 = 0;
% step below necessary for everything to work
z2 = [ones(size(z2,1), 1), z2];
for t = 1:m
    %initialise these two at every iteration for dimensions modifications
    delta3temp = 0;
    delta2temp = 0;
    %calculate error at output
    delta3temp = a3(t,:)- ylarge(t,:);
    %backpropagate error as specified
    delta2temp = delta3temp*Theta2.*sigmoidGradient(z2(t,:));
    %do this step, not sure why yet
    delta2temp = delta2temp(2:end);
    %accumulate error in this big Delta vectors
    Delta1 = Delta1+delta2temp'*X(t,:);
    Delta2 = Delta2+delta3temp'*a2(t,:);
end
%pass into grad vectors by averaging over number of examples
Theta1_grad = (1/m)*Delta1+(lambda/m)*Theta1;
Theta2_grad = (1/m)*Delta2+(lambda/m)*Theta2;

Theta1_grad(:,1) = (1/m)*Delta1(:,1);
Theta2_grad(:,1) = (1/m)*Delta2(:,1);
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
