%% Cuckoo
function [Iout,bestnest,fmax,time]=Cuckoo(n)
tic;
if nargin<1,
% Number of nests (or different solutions)
n=25;%(i.e cuckoos( new solution) can lay eggs in any of these n nest)
end
 
% Discovery rate of alien eggs/solutions
pa=0.5;%(how well the host birdscan detect alian eggs)
 
%% Change this if you want to get better results
N_IterTotalR=100;
N_IterTotalG=100;
N_IterTotalB=100;
 
%Data

I=imread('image.jpg');
%I=rgb2gray(I);
Lmax= 255;
Nt=size(I,1)*size(I,2);

nd=5;% number of thresholds required 
 

 
if size(I,3)==1 %grayscale image
    [n_countR,x_valueR] = imhist(I(:,:,1));
elseif size(I,3)==3 %RGB image
    [n_countR,x_valueR] = imhist(I(:,:,1));
    [n_countG,x_valueG] = imhist(I(:,:,2));
    [n_countB,x_valueB] = imhist(I(:,:,3));
end
 
%PDF
for i=1:Lmax
    if size(I,3)==1 %grayscale image
        probR(i)=n_countR(i)/Nt;
    elseif size(I,3)==3 %RGB image    
        probR(i)=n_countR(i)/Nt;
        probG(i)=n_countG(i)/Nt;
        probB(i)=n_countB(i)/Nt;
    end
end
 
 
if size(I,3)==1 %grayscale image
%Lower bounds and Upper bounds
LbR=zeros(1,nd); 
UbR=Lmax*ones(1,nd);  %(here it is from 0 to 255)
fitnessR=zeros(n,1);

% Random initial solutions
for i=1:n,
nestR(i,:)=LbR+(UbR-LbR).*(rand(size(LbR)));% size(nest)=25X5....ie nXnd
end
for si=1:length(nestR)%No. of rows=N..ie similar si=1:N
nestR(si,:)=sort(nestR(si,:)); % sorting the xR generated randomly as above each row in ascending order
end
nestR=fix(nestR);
% initialized the population with n=25 nests, and nd=5 birds in each
% nest.the Lb and Ub decides the value bounds that can be assigned to each
% bird.
% Get the current best(finding the fittest one in each nest)
[fmaxR,bestnestR,nestR,fitnessR]=get_best_nest(nestR,nestR,fitnessR,nd,probR);
 
 %% Same as above for RGB images treating each components seperately and initializing
elseif size(I,3)==3 %RGB image    
LbR=ones(1,nd); 
LbG=ones(1,nd); 
LbB=ones(1,nd); 
% Upper bounds
UbR=Lmax*ones(1,nd); 
UbG=Lmax*ones(1,nd);
UbB=Lmax*ones(1,nd);
 
fitnessR=zeros(n,1);
fitnessG=zeros(n,1);
fitnessB=zeros(n,1);
 
% Random initial solutions
for i=1:n,
nestR(i,:)=LbR+(UbR-LbR).*(rand(size(LbR)));
nestG(i,:)=LbR+(UbR-LbR).*(rand(size(LbR)));
nestB(i,:)=LbR+(UbR-LbR).*(rand(size(LbR)));
end
for si=1:length(nestR)
nestR(si,:)=sort(nestR(si,:)); 
nestG(si,:)=sort(nestG(si,:)); 
nestB(si,:)=sort(nestB(si,:)); 
end
nestR=fix(nestR);
nestG=fix(nestG);
nestB=fix(nestB);
 
% Get the current best(finding the fittest one in each nest)
[fmaxR,bestnestR,nestR,fitnessR]=get_best_nest(nestR,nestR,fitnessR,nd,probR);
[fmaxG,bestnestG,nestG,fitnessG]=get_best_nest(nestG,nestG,fitnessG,nd,probG);
[fmaxB,bestnestB,nestB,fitnessB]=get_best_nest(nestB,nestB,fitnessB,nd,probB);
end
N_iterR=0;
N_iterG=0;
N_iterB=0;
%% Starting iterations
if size(I,3)==1 %grayscale image
for iter=1:N_IterTotalR,
    % Generate new solutions (but keep the current best)
     new_nestR=get_cuckoos(nestR,bestnestR,LbR,UbR,nd);   
     [fmax1R,bestR,nestR,fitnessR]=get_best_nest(nestR,new_nestR,fitnessR,nd,probR);
    % Update the counter(after one go through all nests)
      N_iterR=N_iterR+n; 
    % Discovery and randomization
      new_nestR=empty_nests(nestR,LbR,UbR,pa) ;
    
    % Evaluate fitness for this set of solutions
      [fmax1R,bestR,nestR,fitnessR]=get_best_nest(nestR,new_nestR,fitnessR,nd,probR);
    % Update the counter again
      N_iterR=N_iterR+n;
    % Find the best objective so far  
    if fmax1R>fmaxR,
        fmaxR=fmax1R;
        bestnestR=bestR;
    end
end %% End of iterations
elseif size(I,3)==3 %RGB image    
for iter=1:N_IterTotalR,
    % Generate new solutions (but keep the current best)
     new_nestR=get_cuckoos(nestR,bestnestR,LbR,UbR,nd);   
     [fmax1R,bestR,nestR,fitnessR]=get_best_nest(nestR,new_nestR,fitnessR,nd,probR);
    % Update the counter(after one go through all nests)
      N_iterR=N_iterR+n; 
    % Discovery and randomization
      new_nestR=empty_nests(nestR,LbR,UbR,pa) ;
    
    % Evaluate fitness for this set of solutions
      [fmax1R,bestR,nestR,fitnessR]=get_best_nest(nestR,new_nestR,fitnessR,nd,probR);
    % Update the counter again
      N_iterR=N_iterR+n;
    % Find the best objective so far  
    if fmax1R>fmaxR,
        fmaxR=fmax1R;
        bestnestR=bestR;
    end
end %%
for iter=1:N_IterTotalG,
    % Generate new solutions (but keep the current best)
     new_nestG=get_cuckoos(nestG,bestnestG,LbG,UbG,nd);   
     [fmax1G,bestG,nestG,fitnessG]=get_best_nest(nestG,new_nestG,fitnessG,nd,probG);
    % Update the counter(after one go through all nests)
      N_iterG=N_iterG+n; 
    % Discovery and randomization
      new_nestG=empty_nests(nestG,LbG,UbG,pa) ;
    
    % Evaluate fitness for this set of solutions
      [fmax1G,bestG,nestG,fitnessG]=get_best_nest(nestG,new_nestG,fitnessG,nd,probG);
    % Update the counter again
      N_iterG=N_iterG+n;
    % Find the best objective so far  
    if fmax1G>fmaxG,
        fmaxG=fmax1G;
        bestnestG=bestG;
    end
end %%
for iter=1:N_IterTotalB,
    % Generate new solutions (but keep the current best)
     new_nestB=get_cuckoos(nestB,bestnestB,LbB,UbB,nd);   
     [fmax1B,bestB,nestB,fitnessB]=get_best_nest(nestB,new_nestB,fitnessB,nd,probB);
    % Update the counter(after one go through all nests)
      N_iterB=N_iterB+n; 
    % Discovery and randomization
      new_nestB=empty_nests(nestB,LbB,UbB,pa) ;
    
    % Evaluate fitness for this set of solutions
      [fmax1B,bestB,nestB,fitnessB]=get_best_nest(nestB,new_nestB,fitnessB,nd,probB);
    % Update the counter again
      N_iterB=N_iterB+n;
    % Find the best objective so far  
    if fmax1B>fmaxB,
        fmaxB=fmax1B;
        bestnestB=bestB;
    end
end %%
end

%% Displaying segmented output
if size(I,3)==1 %grayscale image
 bestR=sort(bestR);
 Iout=imageGRAY(I,bestR);
 bestnest=bestnestR  ;   %return optimal intensity
 fmax=fmaxR;%
elseif size(I,3)==3 %RGB image
     bestR=sort(bestR);
     bestG=sort(bestG);
     bestB=sort(bestB);
    Iout=imageRGB(I,bestR,bestG,bestB);
    bestnest=[bestnestR; bestnestG; bestnestB];
    fmax=[fmaxR; fmaxG; fmaxB]; 
end
 ax(1)=subplot(1,2,1)
 imshow(I,[])
 ax(2)=subplot(1,2,2)
 imshow(Iout,[])
  linkaxes()
 
  time=toc 
 
 
 function imgOut=imageRGB(img,Rvec,Gvec,Bvec)%img=original image;Rvec=xR;Gvec=xG,Bvec=xB
imgOutR=img(:,:,1);
imgOutG=img(:,:,2);
imgOutB=img(:,:,3);
 
Rvec=[0 Rvec 256];
for iii=1:size(Rvec,2)-1
    at=find(imgOutR(:,:)>=Rvec(iii) & imgOutR(:,:)<Rvec(iii+1));
    imgOutR(at)=Rvec(iii);
end
 
Gvec=[0 Gvec 256];
for iii=1:size(Gvec,2)-1
    at=find(imgOutG(:,:)>=Gvec(iii) & imgOutG(:,:)<Gvec(iii+1));
    imgOutG(at)=Gvec(iii);
end
 
Bvec=[0 Bvec 256];
for iii=1:size(Bvec,2)-1
    at=find(imgOutB(:,:)>=Bvec(iii) & imgOutB(:,:)<Bvec(iii+1));
    imgOutB(at)=Bvec(iii);
end
 
imgOut=img;
 
imgOut(:,:,1)=imgOutR;
imgOut(:,:,2)=imgOutG;
imgOut(:,:,3)=imgOutB;
 
 function imgOut=imageGRAY(img,Rvec)
% imgOut=img;
limites=[0 Rvec 255];
tamanho=size(img);
imgOut(:,:)=img*0;
cores=colormap(lines)*255;
close all;
%tic
k=1;
    for i= 1:tamanho(1,1)
        for j=1:tamanho(1,2)
            while(k<size(limites,2))
                if(img(i,j)>=limites(1,k) && img(i,j)<=limites(1,k+1))
                    imgOut(i,j,1)=limites(1,k);
%                     imgOut(i,j,2)=cores(k,2);
%                     imgOut(i,j,3)=cores(k,3);
                end
                k=k+1;
            end
            k=1;
        end
    end
    
 
%% --------------- All subfunctions are list below ------------------
%% Get cuckoos by random walk
function nest=get_cuckoos(nest,best,Lb,Ub,nd)
% Levy flights
n=size(nest,1);
% Levy exponent and coefficient
% For details, see equation (2.21), Page 16 (chapter 2) of the book
% X. S. Yang, Nature-Inspired Metaheuristic Algorithms, 2nd Edition, Luniver Press, (2010).
beta=3/2;
sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);

for j=1:n,
    s=nest(j,:);
    % This is a simple way of implementing Levy flights
    % For standard random walks, use step=1;
    %% Levy flights by Mcculloch's algorithm
   x = stabrnd(.5,1, 1, 1,1, nd);
   % Now the actual random walks or flights
    s=s+x;
   % Apply simple bounds/limits
   nest(j,:)=simplebounds(s,Lb,Ub);% assuring the new updated solution is within the bounds and replacing the corresponding nest values
for si=1:n%No. of rows=N..ie similar si=1:N
nest(si,:)=sort(nest(si,:)); % sorting the xR generated randomly as above each row in ascending order
end
   nest(j,:)=fix(nest(j,:));
end
 
%% Find the current best nest
function [fmax,best,nest,fitness]=get_best_nest(nest,newnest,fitness,nd,probR)
% Evaluating all new solutions
for j=1:size(nest,1),
    fnew=fobj(newnest(j,:),nd,probR);
    if fnew>=fitness(j),
       fitness(j)=fnew;
       nest(j,:)=newnest(j,:);
    end
end
% Find the current best
[fmax,K]=max(fitness) ;%fmin=minimum fitness;K=corresponding row.ie nest
best=nest(K,:);
% copying the nest with 'nd' birds which gave the fittest soln
 
%% Replace some nests by constructing new solutions/nests
function new_nest=empty_nests(nest,Lb,Ub,pa)
% A fraction of worse nests are discovered with a probability pa
n=size(nest,1);
% Discovered or not -- a status vector
K=rand(size(nest))>pa;
 

%% New solution by biased/selective random walks
stepsize=rand*(nest(randperm(n),:)-nest(randperm(n),:));
new_nest=nest+stepsize.*K;
for j=1:size(new_nest,1)
    s=new_nest(j,:);
    new_nest(j,:)=simplebounds(s,Lb,Ub); 
end    
for si=1:n%No. of rows=N..ie similar si=1:N
nest(si,:)=sort(new_nest(si,:)); % sorting the xR generated randomly as above each row in ascending order
end
new_nest=fix(nest);
 
% Application of simple constraints
function s=simplebounds(s,Lb,Ub)
  % Apply the lower bound
  ns_tmp=s;
  I=ns_tmp<Lb;
  ns_tmp(I)=Lb(I);
  
  % Apply the upper bounds 
  J=ns_tmp>Ub;
  ns_tmp(J)=Ub(J);
  % Update this new move 
  s=ns_tmp;
 
%% fitness function
 
function fnew=fobj(u,nd,probR)
j=1;
fitR=sum(probR(1:u(j,1)))*(sum((1:u(j,1)).*probR(1:u(j,1))/sum(probR(1:u(j,1)))) - sum((1:255).*probR(1:255)) )^2;
for jlevel=2:nd
fitR=fitR+sum(probR(u(j,jlevel-1)+1:u(j,jlevel)))*(sum((u(j,jlevel-1)+1:u(j,jlevel)).*probR(u(j,jlevel-1)+1:u(j,jlevel))/sum(probR(u(j,jlevel-1)+1:u(j,jlevel))))- sum((1:255).*probR(1:255)))^2;
end
fitR=fitR+sum(probR(u(j,nd)+1:255))*(sum((u(j,nd)+1:255).*probR(u(j,nd)+1:255)/sum(probR(u(j,nd)+1:255)))- sum((1:255).*probR(1:255)))^2;
fnew=fitR;



 % Stable Random Number Generator (McCulloch 12/18/96)
%------------------------------------------------------------------------------
%------------------------------------------------------------------------------

 function [x] = stabrnd(alpha, beta, c, delta, m, n)


 % Errortraps:
 if alpha < .1 | alpha > 2
 disp('Alpha must be in [.1,2] for function STABRND.')
 alpha
 x = NaN * zeros(m,n);
 return
 end
 if abs(beta) > 1
 disp('Beta must be in [-1,1] for function STABRND.')
 beta
 x = NaN * zeros(m,n);
 return
 end

% Generate exponential w and uniform phi:
 w = -log(rand(m,n));
phi = (rand(m,n)-.5)*pi;

 % Gaussian case (Box-Muller):
 if alpha == 2
 x = (2*sqrt(w) .* sin(phi));
 x = delta + c*x;
 return
 end

 % Symmetrical cases:
 if beta == 0
if alpha == 1 % Cauchy case
 x = tan(phi);
 else
 x = ((cos((1-alpha)*phi) ./ w) .^ (1/alpha - 1) ...
 .* sin(alpha * phi) ./ cos(phi) .^ (1/alpha));
 end

 % General cases:
 else
 cosphi = cos(phi);
 if abs(alpha-1) > 1.e-8
 zeta = beta * tan(pi*alpha/2);
 aphi = alpha * phi;
 a1phi = (1 - alpha) * phi;
 x = ((sin(aphi) + zeta * cos(aphi)) ./ cosphi) ...
 .* ((cos(a1phi) + zeta * sin(a1phi)) ...
 ./ (w .* cosphi)) .^ ((1-alpha)/alpha);
 else
 bphi = (pi/2) + beta * phi;
 x = (2/pi) * (bphi .* tan(phi) - beta * log((pi/2) * w ...
 .* cosphi ./ bphi));
 if alpha ~= 1
 x = x + beta * tan(pi * alpha/2);
 end
 end
end

 % Finale:
x = delta + c * x;
 return
% End of STABRND.M 