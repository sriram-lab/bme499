function structure = make_structure(field, vars)
    for i=1:length(field)
        structure.(field{i}) = vars{i};
    end
end