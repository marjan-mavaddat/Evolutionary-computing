clc;
clear;
close all;

%% Problem Definition
data = load('mydata');
X = data.X;
k = 3;

fitFunction=@(m) Clusteringfit(m, X);     % fit Function

VarSize=[k size(X,2)];  % Decision Variables Matrix Size

nVar=prod(VarSize);     % Number of Decision Variables

VarMin= repmat(min(X),k,1);      % Lower Bound of Variables
VarMax= repmat(max(X),k,1);      % Upper Bound of Variables


%% ABC Settings

MaxIt = 100;              % Maximum Number of Iterations

nPop = 100;               % Population Size (Colony Size)

nOnlooker = nPop;         % Number of Onlooker Bees

L = round(0.6*nVar*nPop); % Abandonment Limit Parameter (Trial Limit)

a = 1;                    % Acceleration Coefficient Upper Bound

%% Initialization

% Empty Bee Structure
empty_bee.Position = [];
empty_bee.fit = [];
empty_bee.Out = [];
% Initialize Population Array
pop = repmat(empty_bee, nPop, 1);

% Initialize Best Solution Ever Found
BestSol.fit = -inf;

% Create Initial Population
for i = 1:nPop
    pop(i).Position = unifrnd(VarMin, VarMax, VarSize);
    [pop(i).fit,  pop(i).Out]= fitFunction(pop(i).Position);
    if pop(i).fit >= BestSol.fit
        BestSol = pop(i);
    end
end

% Abandonment Counter
C = zeros(nPop, 1);

% Array to Hold Best fit Values
Bestfit = zeros(MaxIt, 1);

%% ABC Main Loop

for it = 1:MaxIt
    
    % Recruited Bees
    for i = 1:nPop
        
        % Choose k randomly, not equal to i
        K = [1:i-1 i+1:nPop];
        k = K(randi([1 numel(K)]));
        
        % Define Acceleration Coeff.
        phi = a*unifrnd(-1, +1, VarSize);
        
        % New Bee Position
        newbee.Position = pop(i).Position+phi.*(pop(i).Position-pop(k).Position);
        
        % Apply Bounds
        newbee.Position = max(newbee.Position, VarMin);
        newbee.Position = min(newbee.Position, VarMax);

        % Evaluation
        [newbee.fit, newbee.Out]= fitFunction(newbee.Position);
        
        % Comparision
        if newbee.fit >= pop(i).fit
            pop(i) = newbee;
        else
            C(i) = C(i)+1;
        end
        
    end
    
    % Calculate Fitness Values and Selection Probabilities
    F = zeros(nPop, 1);
    Meanfit = mean([pop.fit]);
    for i = 1:nPop
        F(i) = exp(-pop(i).fit/Meanfit); % Convert fit to Fitness
    end
    P = F/sum(F);
    
    % Onlooker Bees
    for m = 1:nOnlooker
        
        % Select Source Site
        i = RouletteWheelSelection(P);
        
        % Choose k randomly, not equal to i
        K = [1:i-1 i+1:nPop];
        k = K(randi([1 numel(K)]));
        
        % Define Acceleration Coeff.
        phi = a*unifrnd(-1, +1, VarSize);
        
        % New Bee Position
        newbee.Position = pop(i).Position+phi.*(pop(i).Position-pop(k).Position);
        
        % Apply Bounds
        newbee.Position = max(newbee.Position, VarMin);
        newbee.Position = min(newbee.Position, VarMax);
        
        % Evaluation
        [newbee.fit, newbee.Out]= fitFunction(newbee.Position);
        
        % Comparision
        if newbee.fit >= pop(i).fit
            pop(i) = newbee;
        else
            C(i) = C(i) + 1;
        end
        
    end
    
    % Scout Bees
    for i = 1:nPop
        if C(i) >= L
            pop(i).Position = unifrnd(VarMin, VarMax, VarSize);
            [pop(i).fit,pop(i).Out] = fitFunction(pop(i).Position);
            C(i) = 0;
        end
    end
    
    % Update Best Solution Ever Found
    for i = 1:nPop
        if pop(i).fit >= BestSol.fit
            BestSol = pop(i);
        end
    end
    
    % Store Best fit Ever Found
    Bestfit(it) = BestSol.fit;
    
    % Display Iteration Information
    disp(['Iteration ' num2str(it) ': Best fit = ' num2str(Bestfit(it))]);
        
    % Plot Solution
    figure(1);
    PlotSolution(X, BestSol);
    pause(0.01);
end
    
%% Results

figure;
%plot(Bestfit, 'LineWidth', 2);
semilogy(Bestfit, 'LineWidth', 2);
xlabel('Iteration');
ylabel('Best fit');
grid on;
