function [A, cache] = linear_activation_forward(A_prev, W, b, type)

    data = zeros(size(prior));
    [Z, cache] = linear_forward(A_prev, W, b);
    switch type
        case 'sigmoid'
            A = 1.0 ./ (1.0 + exp(-A_prev));
        case 'relu'
            A = max(A_prev, 0);
        case 'leaky_relu'
    end
    
end