function g = sigmoid(z)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
g = 1 ./ (1 + exp(-z));
end
