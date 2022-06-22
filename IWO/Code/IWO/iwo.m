clc;
clear;
close all;

%% Problem Definition

model = CreateModel();  % Create Bin Packing Model

fitFunction = @(x) BinPackingFit(x, model);  % Objective Function

nVar = 2*model.n-1;     % Number of Decision Variables
VarSize = [1 nVar];     % Decision Variables Matrix Size

VarMin = 0;     % Lower Bound of Decision Variables
VarMax = 1;     % Upper Bound of Decision Variables

%% IWO Parameters

MaxIt = 500;    % Maximum Number of Iterations

nPop0 = 20;     % Initial Population Size
nPop = 100;     % Maximum Population Size

Smin = 0;       % Minimum Number of Seeds
Smax = 5;       % Maximum Number of Seeds

Exponent = 2;           % Variance Reduction Exponent
sigma_initial = 1;      % Initial Value of Standard Deviation
sigma_final = 0.001;	% Final Value of Standard Deviation

%% Initialization

% Empty Plant Structure
empty_plant.Position = [];
empty_plant.fit = [];
empty_plant.Sol = [];

pop = repmat(empty_plant, nPop0, 1);    % Initial Population Array

for i = 1:numel(pop)
    
    % Initialize Position
    pop(i).Position = unifrnd(VarMin, VarMax, VarSize);
    
    % Evaluation
    [pop(i).fit, pop(i).Sol] = fitFunction(pop(i).Position);
    
end

% Initialize Best fitness History
Bestfits = zeros(MaxIt, 1);

%% IWO Main Loop

for it = 1:MaxIt
    
    % Update Standard Deviation
    sigma = ((MaxIt - it)/(MaxIt - 1))^Exponent * (sigma_initial - sigma_final) + sigma_final;
    
    % Get Best and Worst fit Values
    fits = [pop.fit];
    Bestfit = max(fits);
    Worstfit = min(fits);
    
    % Initialize Offsprings Population
    newpop = [];
    
    % Reproduction
    for i = 1:numel(pop)
        
        ratio = (pop(i).fit - Worstfit)/(Bestfit - Worstfit);
        S = floor(Smin + (Smax - Smin)*ratio);
        
        for j = 1:S
            
            % Initialize Offspring
            newsol = empty_plant;
            
            % Generate Random Location
            newsol.Position = pop(i).Position + sigma * randn(VarSize);
            
            % Apply Lower/Upper Bounds
            newsol.Position = max(newsol.Position, VarMin);
            newsol.Position = min(newsol.Position, VarMax);
            
            % Evaluate Offsring
            [newsol.fit, newsol.Sol] = fitFunction(newsol.Position);
            
            % Add Offpsring to the Population
            newpop = [newpop
                      newsol];  %#ok
            
        end
        
    end
    
    % Merge Populations
    pop = [pop
           newpop];
    
    % Sort Population
    [~, SortOrder]=sort([pop.fit],'descend');
    pop = pop(SortOrder);

    % Competitive Exclusion (Delete Extra Members)
    if numel(pop)>nPop
        pop = pop(1:nPop);
    end
    
    % Store Best Solution Ever Found
    BestSol = pop(1);
    
    % Store Best fit History
    Bestfits(it) = BestSol.fit;
    
    % Display Iteration Information
    disp(['Iteration ' num2str(it) ': Best fit = ' num2str(Bestfits(it))]);
    
end

%% Results

figure;
plot(Bestfits,'LineWidth',2);
xlabel('Iteration');
ylabel('Best fit');
