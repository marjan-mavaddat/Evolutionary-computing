clc;
clear;
close all;

%% Problem Definition

model = CreateModel();  % Create Bin Packing Model

FitFunction = @(x) BinPackingFit(x, model);  % Objective Function

nVar = 2*model.n-1;     % Number of Decision Variables
VarSize = [1 nVar];     % Decision Variables Matrix Size

VarMin = 0;     % Lower Bound of Decision Variables
VarMax = 1;     % Upper Bound of Decision Variables


%% Firefly Algorithm Parameters

MaxIt=150;         % Maximum Number of Iterations

nPop=20;            % Number of Fireflies (Swarm Size)

gamma=1;            % Light Absorption Coefficient

beta0=2;            % Attraction Coefficient Base Value

alpha=0.2;          % Mutation Coefficient

alpha_damp=0.98;    % Mutation Coefficient Damping Ratio

delta=0.05*(VarMax-VarMin);     % Uniform Mutation Range

m=2;

if isscalar(VarMin) && isscalar(VarMax)
    dmax = (VarMax-VarMin)*sqrt(nVar);
else
    dmax = norm(VarMax-VarMin);
end

nMutation = 3;      % Number of Additional Mutation Operations

%% Initialization

% Empty Firefly Structure
firefly.Position=[];
firefly.Fit=[];
firefly.Sol=[];

% Initialize Population Array
pop=repmat(firefly,nPop,1);

% Initialize Best Solution Ever Found
BestSol.Fit=-inf;

% Create Initial Fireflies
for i=1:nPop
   pop(i).Position=unifrnd(VarMin,VarMax,VarSize);
   [pop(i).Fit, pop(i).Sol]=FitFunction(pop(i).Position);
   
   if pop(i).Fit>=BestSol.Fit
       BestSol=pop(i);
   end
end

% Array to Hold Best Fit Values
BestFit=zeros(MaxIt,1);

%% Firefly Algorithm Main Loop

for it=1:MaxIt
    
    newpop=repmat(firefly,nPop,1);
    for i=1:nPop
        newpop(i).Fit = -inf;
        for j=1:nPop
            if pop(j).Fit > pop(i).Fit || i==j
                rij=norm(pop(i).Position-pop(j).Position)/dmax;
                beta=beta0*exp(-gamma*rij^m);
                e=delta*unifrnd(-1,+1,VarSize);
                %e=delta*randn(VarSize);
                
                newsol.Position = pop(i).Position ...
                                + beta*rand(VarSize).*(pop(j).Position-pop(i).Position) ...
                                + alpha*e;
                
                newsol.Position=max(newsol.Position,VarMin);
                newsol.Position=min(newsol.Position,VarMax);
                
                [newsol.Fit, newsol.Sol]=FitFunction(newsol.Position);
                
                if newsol.Fit >= newpop(i).Fit
                    newpop(i) = newsol;
                    if newpop(i).Fit>=BestSol.Fit
                        BestSol=newpop(i);
                    end
                end
                
            end
        end
        
        % Perform Mutation
        for k=1:nMutation
            newsol.Position = Mutate(pop(i).Position);
            [newsol.Fit, newsol.Sol]=FitFunction(newsol.Position);
            if newsol.Fit >= newpop(i).Fit
                newpop(i) = newsol;
                if newpop(i).Fit>=BestSol.Fit
                    BestSol=newpop(i);
                end
            end
        end
                
    end
    
    % Merge
    pop=[pop
         newpop];  %#ok
    
    % Sort
    [~, SortOrder]=sort([pop.Fit],'descend');
    pop=pop(SortOrder);
    
    % Truncate
    pop=pop(1:nPop);
    
    % Store Best Fit Ever Found
    BestFit(it)=BestSol.Fit;
    
    % Show Iteration Information
    disp(['Iteration ' num2str(it) ': Best Fit = ' num2str(BestFit(it))]);
    
    % Damp Mutation Coefficient
    alpha = alpha*alpha_damp;

end

%% Results

figure;
plot(BestFit,'LineWidth',2);
xlabel('Iteration');
ylabel('Best Fit');
grid on;
