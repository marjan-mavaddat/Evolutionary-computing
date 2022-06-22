function fobj = Get_Functions_details(F)
switch F
    case 'F1'
        fobj = @F1;
 end
end

% F1
function f = F1(malakh,G,ub)
    f=0;
    y=0;
    for i=1:ub
        if sum(i==malakh)==0
            f=0;
            return;
        end
    end   
    for i=1:size(G,1)
        for j=1:size(G,2)
            if G(i,j)==1
                y=y+1;
                if malakh(i)==malakh(j)
                    f=f+1;
                end
            end
        end
    end
   f=(y/2)-(f/2);
end

