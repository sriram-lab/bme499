function [cost] = CostFunction(AL, Y)
    % CostFunction implements a cost function for n layer neural networks
    m = shape(Y, 2);
    A = 
    cost = -1*sum(dot(Y, log(AL') + dot(1-Y, log(1-A))))/m;

end