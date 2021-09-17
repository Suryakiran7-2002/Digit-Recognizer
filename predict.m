function I = predict(X,theta1,theta2)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
m = size(X,1);
X = [ones(m,1) X];

z2 = X * theta1';
a2 = sigmoid(z2);
m1 = size(a2,1);
a2 = [ones(m1,1) a2];

z3 = a2 * theta2';
a3 = sigmoid(z3);


[M,I] = max(a3,[],2);


end
