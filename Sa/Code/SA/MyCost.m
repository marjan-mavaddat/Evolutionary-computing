function [z, sol]=MyCost(q,model)

    sol=ParseSolution(q,model);
    
    z=sol.Cmax;

end