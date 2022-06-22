function [z sol]=KnapsackFitness(x,model)

    global NFE;
    if isempty(NFE)
        NFE=0;
    end
    NFE=NFE+1;
    
    v=model.v;
    w=model.w;
    W=model.W;
    
    z=sum(x.*v);
    c=max(sum(x.*w)-W,0);
    z=z-100*c;
end