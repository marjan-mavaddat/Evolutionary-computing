function model=CreateModel()
     %% Input Graph:
        N_nodes=input('Enter Number of Nodes:');
        N_Colors=input('Enter Number of Colors:');
        G=zeros(N_nodes);
        type=input('Graph Type: Custom(1) / Random(2)?');
        if type==1
            % Custom Graph:
            for i=1:N_nodes
                for j=i:N_nodes
                    if i~=j
                        connectivity=input(['Is there a link between nodes ',num2str(i),' and ',num2str(j), ' ? (0/1) : ']);
                        if connectivity==1
                            G(i,j)=1;
                            G(j,i)=1;
                        end
                    end
                end
            end
        else
            % Random Graph:
            for i=1:N_nodes
                for j=i:N_nodes
                    if i~=j
                        if rand>0.5
                            G(i,j)=1;
                            G(j,i)=1;
                        end
                    end
                end
            end
        end
        %% Drawing Graph:
        position=randsrc(2,N_nodes,1:1000); 
        figure(1),
        plot(position(1,:),position(2,:),'o');
        title('Input Graph');
        hold on
        tedadyal=0;
        for i=1:N_nodes
            for j=1:N_nodes
                if G(i,j)==1
                     line([position(1,i),position(1,j)],[position(2,i),position(2,j)]);
                     tedadyal=tedadyal+1;
                end
            end
        end
        hold off
    model.N_nodes=N_nodes;
    model.N_Colors=N_Colors;
    model.G=G;
    model.position=position;
    model.tedadyal=tedadyal/2;
end