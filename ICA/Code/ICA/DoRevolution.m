function emp=DoRevolution(emp)

    global ProblemSettings;
    CostFunction=ProblemSettings.CostFunction;
    nVar=ProblemSettings.nVar;
    VarSize=ProblemSettings.VarSize;
    VarMin=ProblemSettings.VarMin;
    VarMax=ProblemSettings.VarMax;
    
    global ICASettings;
    pRevolution=ICASettings.pRevolution;
    mu=ICASettings.mu;
    
    nmu=ceil(mu*nVar);
    
    sigma=0.1*(VarMax-VarMin);
    
    nEmp=numel(emp);
    for k=1:nEmp
        
        NewPos = emp(k).Imp.Position + sigma*randn(VarSize);
        NewPos = max(NewPos,VarMin);
		NewPos = min(NewPos,VarMax);
		
        jj=randsample(nVar,nmu)';
        NewImp=emp(k).Imp;
        NewImp.Position(jj)=NewPos(jj);
        NewImp.Cost=CostFunction(NewImp.Position);
        if NewImp.Cost<emp(k).Imp.Cost
            emp(k).Imp = NewImp;
        end
        
        for i=1:emp(k).nCol
            if rand<=pRevolution

                NewPos = emp(k).Col(i).Position + sigma*randn(VarSize);
				NewPos = max(NewPos,VarMin);
				NewPos = min(NewPos,VarMax);
                
                jj=randsample(nVar,nmu)';
                emp(k).Col(i).Position(jj) = NewPos(jj);

                emp(k).Col(i).Cost = CostFunction(emp(k).Col(i).Position);

            end
        end
    end

end