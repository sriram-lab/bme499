function [Z, cache] = linear_forward(A, W, b)
    Z = dot(A, W) + b;
    vars = {A, W, b};
    names = {'A', 'W', 'b'};
    cache = make_structure(names, vars);
end