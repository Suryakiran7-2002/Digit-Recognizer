function [J grad] = nncostfun(X,y,theta,n_input,n_hidden,n_output,lambda)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


theta1 = reshape(theta(1:n_hidden * (n_input+1)),n_hidden,n_input+1);
theta2 = reshape(theta((n_hidden * (n_input+1)) + 1:end),n_output,n_hidden+1);

theta1_grad = zeros(size(theta1));
theta2_grad = zeros(size(theta2));

J = 0;

m = size(X,1);
X = [ones(m,1) X];

z2 = X * theta1';
a2 = [sigmoid(z2)];
s_a2 = size(a2,1);
a2 = [ones(s_a2,1) a2];
z3 = a2 * theta2';
a3 = sigmoid(z3);



y_vec = (1:n_output) == y;

h = a3;

J = (-1/m) * sum(sum((y_vec .* log(h)) + ((1-y_vec) .* log(1-h))));


for i = 1:m
    a1 = X(i,:)';
   
    
    
    z2 = theta1 * a1;
    a2 = sigmoid(z2);
    
    a2 = [1 ; a2];
    
    z3 = theta2 * a2;
    a3 = sigmoid(z3);
    
    delta3 = a3 - y_vec(i,:)';
    
    delta2  = (theta2' * delta3) .* [1;sigmoidgrad(z2)];
    delta2 = delta2(2:end);
    
    theta1_grad = theta1_grad + (delta2 * a1');
    theta2_grad = theta2_grad + (delta3 * a2');
    
end

theta2_grad = (1/m) * theta2_grad;
theta1_grad = (1/m) * theta1_grad;

reg_term = (lambda/(2*m)) * (sum(sum(theta1(:,2:end).^2)) + sum(sum(theta2(:,2:end).^2))); %scalar
  
  %Costfunction With regularization
  J = J + reg_term; %scalar
  
  %Calculating gradients for the regularization
  theta1_grad_reg_term = (lambda/m) * [zeros(size(theta1, 1), 1) theta1(:,2:end)]; % 25 x 401
  theta2_grad_reg_term = (lambda/m) * [zeros(size(theta2, 1), 1) theta2(:,2:end)]; % 10 x 26
  
  %Adding regularization term to earlier calculated Theta_grad
  theta1_grad = theta1_grad + theta1_grad_reg_term;
  theta2_grad = theta2_grad + theta2_grad_reg_term;

grad = [theta1_grad(:);theta2_grad(:)];


end
