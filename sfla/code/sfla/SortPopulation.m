function [pop, SortOrder] = SortPopulation(pop)

    % Get fits
    fits = [pop.fit];
    
    % Sort the fits Vector
    [~, SortOrder] = sort(fits,'descend');
    
    % Apply the Sort Order to Population
    pop = pop(SortOrder);

end