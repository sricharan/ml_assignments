function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


[J_old, grad_old] = costFunction(theta,X,y);

disp("----------");
disp(theta);

disp("------------");
theta_matrix = theta(2:end);
 
theta_matrix_new = [0; theta_matrix];

disp(theta_matrix_new);

sq_theta_matrix = theta_matrix_new.^2;

J = J_old + (lambda*sum(sq_theta_matrix))/(2*m);

grad = grad_old + lambda*theta_matrix_new/m;



% =============================================================

end
