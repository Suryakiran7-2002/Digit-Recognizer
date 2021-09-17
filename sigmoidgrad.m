function g = sigmoidgrad(z)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
g = sigmoid(z) .* (1 - sigmoid(z));
end
