clc;
clear;
close all;

%% Problem Definition

model = CreateModel();

FitFunction = @(x) MyFit(x, model);

nVar = model.n;


%% ACO Parameters

MaxIt = 300;      % Maximum Number of Iterations

nAnt = 40;        % Number of Ants (Population Size)

Q = 1;

tau0 = 0.1;        % Initial Phromone

alpha = 1;        % Phromone Exponential Weight
beta = 0.02;      % Heuristic Exponential Weight

rho = 0.1;        % Evaporation Rate


%% Initialization

N = [0 1];

eta = [model.w./model.v
     model.v./model.w];           % Heuristic Information

tau = tau0*ones(2, nVar);      % Phromone Matrix

BestFit = zeros(MaxIt, 1);    % Array to Hold Best Fit Values

% Empty Ant
empty_ant.Tour = [];
empty_ant.x = [];
empty_ant.Fit = [];
empty_ant.Sol = [];

% Ant Colony Matrix
ant = repmat(empty_ant, nAnt, 1);

% Best Ant
BestSol.Fit = 0;


%% ACO Main Loop

for it = 1:MaxIt
    
    % Move Ants
    for k = 1:nAnt
        
        ant(k).Tour = [];
        
        for l = 1:nVar
            
            P = tau(:, l).^alpha.*eta(:, l).^beta;
            
            P = P/sum(P);
            
            j = RouletteWheelSelection(P);
            
            ant(k).Tour = [ant(k).Tour j];
            
        end
        
        ant(k).x = N(ant(k).Tour);
        
        [ant(k).Fit, ant(k).Sol] = FitFunction(ant(k).x);
        
        if ant(k).Fit>BestSol.Fit
            BestSol = ant(k);
        end
        
    end
    
    % Update Phromones
    for k = 1:nAnt
        
        tour = ant(k).Tour;
        
        for l = 1:nVar
            
            tau(tour(l), l) = tau(tour(l), l)+Q/ant(k).Fit;
            
        end
        
    end
    
    % Evaporation
    tau = (1-rho)*tau;
    
    % Store Best Fit
    BestFit(it) = BestSol.Fit;
    
    % Show Iteration Information
    if BestSol.Sol.IsFeasible
        FeasiblityFlag = '*';
    else
        FeasiblityFlag = '';
    end
    disp(['Iteration ' num2str(it) ': Best Fit = ' num2str(BestFit(it)) ' ' FeasiblityFlag]);
    
end

%% Results

figure;
plot(BestFit, 'LineWidth', 2);
xlabel('Iteration');
ylabel('Best Fit');
grid on;
