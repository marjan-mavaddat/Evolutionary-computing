clc;
clear;
close all;

%% Problem Definition

model=CreateModel();

fitFunction=@(s) Myfitt(s,model);        % fit Function

nVar=model.n;             % Number of Decision Variables

VarSize=[1 nVar];   % Decision Variables Matrix Size

VarMin=0;         % Lower Bound of Variables
VarMax=1;         % Upper Bound of Variables

%% SFLA Parameters

MaxIt = 50;        % Maximum Number of Iterations

nPopMemeplex = nVar*2;                          % Memeplex Size
nPopMemeplex = max(nPopMemeplex, nVar+1);   % Nelder-Mead Standard

nMemeplex = nVar;                  % Number of Memeplexes
nPop = nMemeplex*nPopMemeplex;	% Population Size

I = reshape(1:nPop, nMemeplex, []);

% FLA Parameters
fla_params.q = max(round(0.3*nPopMemeplex), 2);   % Number of Parents
fla_params.alpha = 10;   % Number of Offsprings
fla_params.beta = 5;    % Maximum Number of Iterations
fla_params.sigma = 1;   % Step Size
fla_params.fitFunction = fitFunction;
fla_params.VarMin = VarMin;
fla_params.VarMax = VarMax;

%% Initialization

% Empty Individual Template
empty_individual.Position = [];
empty_individual.fit = [];
empty_individual.Sol=[];
% Initialize Population Array
pop = repmat(empty_individual, nPop, 1);

% Initialize Population Members
for i = 1:nPop
    pop(i).Position = unifrnd(VarMin, VarMax, VarSize);
    [ pop(i).fit  pop(i).Sol] = fitFunction(pop(i).Position);
end

% Sort Population
pop = SortPopulation(pop);

% Update Best Solution Ever Found
BestSol = pop(1);

% Initialize Best fits Record Array
Bestfits = nan(MaxIt, 1);

%% SFLA Main Loop

for it = 1:MaxIt
    
    fla_params.BestSol = BestSol;

    % Initialize Memeplexes Array
    Memeplex = cell(nMemeplex, 1);
    
    % Form Memeplexes and Run FLA
    for j = 1:nMemeplex
        % Memeplex Formation
        Memeplex{j} = pop(I(j, :));
        
        % Run FLA
        Memeplex{j} = RunFLA(Memeplex{j}, fla_params);
        
        % Insert Updated Memeplex into Population
        pop(I(j, :)) = Memeplex{j};
    end
    
    % Sort Population
    pop = SortPopulation(pop);
    
    % Update Best Solution Ever Found
    BestSol = pop(1);
    
    % Store Best fit Ever Found
    Bestfits(it) = BestSol.fit;
    
    % Show Iteration Information
    disp(['Iteration ' num2str(it) ': Best fit = ' num2str(Bestfits(it))]);
    % Plot Solution
    figure(1);
    PlotSolution(BestSol.Sol.tour,model);
end

%% Results

figure;
%plot(Bestfits, 'LineWidth', 2);
semilogy(Bestfits, 'LineWidth', 2);
xlabel('Iteration');
ylabel('Best fit');
grid on;
