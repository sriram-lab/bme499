function [] = GradientDescent(A, theta)
    
% GradientDescent returns a numerical estimate of the gradient
    gradient = zeros(size(A));
    noise = zeros(size(theta));
    window = 1E-4;
    
    for value = 1:numel(theta)
        noise(value) = window;
        negLoss = A(theta - noise);
        posLoss = A(theta + noise);
        
        gradient(p) = (posLoss - negLoss) / (2*window);
        noise(value) = 0;
        
    end
    
end