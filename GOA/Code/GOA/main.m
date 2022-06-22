clear all 
clc

SearchAgents_no=8; % Number of search agents

model=CreateModel();

N_nodes=model.N_nodes;
N_Colors=model.N_Colors;
G=model.G;
Position=model.position;
tedadyal=model.tedadyal;

lb=1;
ub=N_Colors;
dim=N_nodes;

Function_name='F1'; % Name of the test function that can be from F1 to F23 (Table 1,2,3 in the paper)

Max_iteration=25; % Maximum numbef of iterations

% Load details of the selected benchmark function
fobj=Get_Functions_details(Function_name);

[Target_score,Target_pos,GOA_cg_curve, Trajectories,fitness_history, position_history]=GOA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj,G);
%% Create Color Table:
    w=9;
    Color={[0 0 0],[0 0 1],[0 1 0],[0 1 1],[1 0 0],[1 0 1],[1 1 0],[1 1 1]};
    stepsize=0.1;
    for i=0.1:stepsize:0.9
        for j=0.1:stepsize:0.9
            for k=0.1:stepsize:0.9
                Color{w}=[i j k];
                w=w+1;
            end
        end
    end
    %% Drawing Colored Graph:
    figure(2),
    for i=1:dim
        for j=1:dim
            if G(i,j)==1
                 line([Position(1,i),Position(1,j)],[Position(2,i),Position(2,j)]);
                 hold on
            end
        end
    end
for i=1:N_nodes 
    plot(Position(1,i),Position(2,i),'o','MarkerFaceColor',Color{Target_pos(i)},'MarkerEdgeColor',Color{Target_pos(i)},'MarkerSize',11);
    text(Position(1,i),Position(2,i),num2str(Target_pos(i)),'color','r')
    hold on
end
title('Colored Graph');
display(['The tedadyal grapg is : ', num2str(tedadyal)]);
display(['The best solution obtained by GOA is : ', num2str(Target_pos)]);
display(['The best optimal value of the objective funciton found by GOA is : ', num2str(Target_score)]);
