function [g]=ipm_UE_corr(fA,fAt,m,Q2,n,CS)
% % % % % % % % % % % % % % % % % % % % % % % % % % %
% IPM method for UNA Europa project with 
% multiple centrality correctors
%
% fA:   function that applies A 
% fAt:  function that applies A'
% m:    measurement vector
% Q2:   regularization matrix
% n:    size of g
% CS:   struct with coefficients
%       CS.e11 = alpha+(c11^2+c21^2)*rho
%       CS.e12 = beta+(c11*c12+c21*c22)*rho
%       CS.e22 = alpha+(c12^2+c22^2)*rho
%       where rho is the mean diagonal element 
%       of A'*A
%
% Author: Filippo Zanetti, 2020
% 
% % % % % % % % % % % % % % % % % % % % % % % % % % %

%% Parameters
%Tolerance
tol=1e-8;
%Max number of iterations
maxit=500;
%Number of centrality correctors
max_cc=3;
%Symmetric neighborhood coefficient
gamma=0.2;
%sigma in affine scaling direction
sigma_aff=0.05;         

%% Initialization
g=10*ones(n,1);
s=10*ones(n,1);
iter = 0;
IPM_time=0;
iterPCG=0;
rho = 0.995;
tol_cg=1e-2;

%% Iteration
% % % % % % % % % % % % % % % % % % % % % % % % % %
% Newton system
%   [ Q -I ] (dg) = r1
%   [ S  G ] (ds) = r2
%
% Q=Q1+Q2
% r1=A'*m+s-Q*g
% r2=sigma*mu*e-g.*s
% mu=g'*s/n;
% % % % % % % % % % % % % % % % % % % % % % % % % %  

Atm=fAt(m);
nAtm=norm(Atm);
r1 = Atm+s-fAt(fA(g))-Q2*g;
mu = (g'*s)/n;

fprintf('\nphase      infeas       mu      PCG_iter    alpha_g   alpha_s\n')

while iter<maxit
    
    if norm(r1)/norm(Atm)<tol && mu<tol
       fprintf('\n*** Optimal solution found ***\n')
       break
    end
    
    iter=iter+1;
    fprintf('\nIPM iteration: %d\n',iter)
    
    % % % % % % % % % % % % % % % % % % % % % % % % % %
    % Normal equations
    % (Q+G^-1*S)*dg = r1+G^-1*r2
    % ds = G^-1*(r2-S*dg)
    %
    % Solved using pcg with 2x2 diagonal block preconditioner
    % % % % % % % % % % % % % % % % % % % % % % % % % %
    
    %Heuristic for choosing mu
    if iter==1 || min(alpha_g,alpha_s)<0.3
        mu_target=mu*0.7;
    else
       mu_target=mu*0.1;
    end
    
    %Heuristic for choosing the tolerance of pcg.
    if (tol_cg==1e-2) && (iter>3 && min(alpha_g,alpha_s)>0.2)
        tol_cg=1e-4;
    end
    if (tol_cg==1e-4) && (iter>12 && min(alpha_g,alpha_s)>0.7)
        tol_cg=1e-6;
    end
    
    %% Affine scaling direction
    
    %build right hand side
    r2=-g.*s+sigma_aff*mu_target;
    rhs=r1+r2./g;
    
    %solve the normal equations
    tic
    [dg_p,flag,~,iterpcg]=pcg(@(x) applyNE(x,fA,fAt,Q2,g,s),rhs,tol_cg,400,@(x) apply_prec(x,g,s,CS,n));
    ds_p=(r2-dg_p.*s)./g;
    itertime=toc;
    IPM_time=IPM_time+itertime;
    iterPCG=iterPCG+iterpcg;
    
    %find steplength
    idg = dg_p < 0;
    ids = ds_p < 0;
    alpha_g = min([1;-g(idg)./dg_p(idg)]);
    alpha_s = min([1;-s(ids)./ds_p(ids)]);
    g_aff = g+alpha_g*dg_p; 
    s_aff = s+alpha_s*ds_p;
    
    %print output
    r1=Atm+s_aff-fAt(fA(g_aff))-Q2*g_aff;
    mu_aff=(g_aff'*s_aff)/n;
    fprintf('AffSca  %10.2e %10.2e %6d (%1d) %10.5f %9.5f\n',norm(r1)/nAtm,mu_aff,iterpcg,flag,alpha_g,alpha_s)
    
    %% Centrality correctors
    
    %iterate centrality correctors
    for cc=1:max_cc
        
        %find trial point
        alpha_g_tilde=min(1.5*alpha_g+0.3,1);
        alpha_s_tilde=min(1.5*alpha_s+0.3,1);
        g_tilde=g+alpha_g_tilde*dg_p;
        s_tilde=s+alpha_s_tilde*ds_p;
        
        %complementarity of trial point
        v_tilde=g_tilde.*s_tilde;
        t=zeros(n,1);
        for i=1:n
            if v_tilde(i)<=gamma*mu_target
               t(i)=gamma*mu_target-v_tilde(i);
            elseif v_tilde(i)>=mu_target/gamma
                t(i)=mu_target/gamma-v_tilde(i);
            end
        end
        
        %avoid large elements in t
        t(t<-mu_target/gamma)=-mu_target/gamma;
        t(t>2*mu_target/gamma)=2*mu_target/gamma;
        rhs=t./g;

        %solve normal equations
        tic
        [dg_cor,flag,~,iterpcg]=pcg(@(x) applyNE(x,fA,fAt,Q2,g,s),rhs,tol_cg,400,@(x) apply_prec(x,g,s,CS,n));
        ds_cor=(t-dg_cor.*s)./g;
        itertime=toc;
        IPM_time=IPM_time+itertime;
        iterPCG=iterPCG+iterpcg;
        
        %find steplength
        dg_p=dg_p+dg_cor;
        ds_p=ds_p+ds_cor;
        idg = dg_p < 0;
        ids = ds_p < 0;
        alpha_g = min([1;-g(idg)./dg_p(idg)]);
        alpha_s = min([1;-s(ids)./ds_p(ids)]);
       
        %print output
        g_cor=g+alpha_g*dg_p;
        s_cor=s+alpha_s*ds_p;
        r1=Atm+s_cor-fAt(fA(g_cor))-Q2*g_cor;
        mu_cor=(g_cor'*s_cor)/n;
        fprintf('CenCor  %10.2e %10.2e %6d (%1d) %10.5f %9.5f\n',norm(r1)/nAtm,mu_cor,iterpcg,flag,alpha_g,alpha_s)
        
        %interrupt correctors if steplength is large
        if alpha_g>0.99 && alpha_s>0.99
            break
        end
       
    end
    
    %prepare for next ipm iteration
    g = g+rho*alpha_g*dg_p; 
    s = s+rho*alpha_s*ds_p;   
    r1=Atm+s-fAt(fA(g))-Q2*g;
    mu=(g'*s)/n;
    
end

%% Print output
fprintf('\nTime: %f\n',IPM_time)
fprintf('IPM iter: %d\n',iter)
fprintf('PCG iter: %d\n',iterPCG)






