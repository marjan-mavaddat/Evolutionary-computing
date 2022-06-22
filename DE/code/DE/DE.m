clc;
clear;
close all;
%% Problem Definition
global NFE;
NFE=0;

model=CreateModel();    % Create Knapsack Model

FitnessFunction=@(x) KnapsackFitness(x,model);    % Cost Function

nVar=model.n;            % Number of Decision Variables
VarSize=[1 nVar];   % Decision Variables Matrix Size

VarMin=0;          % Lower Bound of Decision Variables
VarMax=1;          % Upper Bound of Decision Variables
%% DE Parameters
MaxIt=500;      % Maximum Number of Iterations
nPop=50;        % Population Size
beta_min=0.2;   % Lower Bound of Scaling Factor
beta_max=0.8;   % Upper Bound of Scaling Factor
pCR=0.2;        % Crossover Probability
%% Initialization
empty_individual.Position=[];
empty_individual.Fitness=[];
BestSol.Fitness=0;
pop=repmat(empty_individual,nPop,1);
for i=1:nPop

    pop(i).Position=randi([0 1],VarSize);
    
    pop(i).Fitness=FitnessFunction(pop(i).Position);
    
    if pop(i).Fitness>BestSol.Fitness
        BestSol=pop(i);
    end
    
end
BestFitness=zeros(MaxIt,1);
%% DE Main Loop
for it=1:MaxIt
    
    for i=1:nPop
        
        x=pop(i).Position;
        
        A=randperm(nPop);
        
        A(A==i)=[];
        
        a=A(1);
        b=A(2);
        c=A(3);
        
        % Mutation
        %beta=unifrnd(beta_min,beta_max);
        beta=unifrnd(beta_min,beta_max,VarSize);
        y=pop(a).Position+beta.*(pop(b).Position-pop(c).Position);
        y = max(y, VarMin);
		y = min(y, VarMax);
		
        % Crossover
        z=zeros(size(x));
        j0=randi([1 numel(x)]);
        for j=1:numel(x)
            if j==j0 || rand>=pCR
                z(j)=y(j);
            else
                z(j)=x(j);
            end
        end
        
        NewSol.Position=z;
        NewSol.Fitness=FitnessFunction(NewSol.Position);
        
        if NewSol.Fitness>pop(i).Fitness
            pop(i)=NewSol;
            
            if pop(i).Fitness>BestSol.Fitness
               BestSol=pop(i);
            end
        end
        
    end
    
    % Update Best Fitness
    BestFitness(it)=BestSol.Fitness;
    
    % Show Iteration Information
    disp(['Iteration ' num2str(it) ': Best Fitness = ' num2str(BestFitness(it))]);
    
end
%% Show Results
figure;
%plot(BestFitness);
semilogy(BestFitness, 'LineWidth', 2);
xlabel('Iteration');
ylabel('Best Fitness');
grid on;
