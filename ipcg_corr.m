function [dg,iter,res,flag]=ipcg_corr(fq,b,tol,itmax,fp,r1,r2,g,s,mu,nAtm,dg_p,ds_p)
%-----------------------------------------------------
% This function implements the IPCG: PCG for IPM with 
% a smart stop criterion that estimates some convergence
% indicators. 
% UNA Europa version for centrality corrector direction
% 
% INPUT
% fq    = function that applies matrix Q  
% b     = rhs
% tol   = tolerance
% itmax = maximum number of iterations
% fp    = function that applies the preconditioner
% r1    = dual infeasibility from IPM (A'*m-Q*g+s)
% r2    = vector from IPM (mu*e-G*S*e)
% g     = primal variable from IPM
% s     = slack variable from IPM
% mu    = previous centrality measure from IPM
% nAtm  = norm of A'*m from IPM
% OUTPUT
% dg     = solution
% iter   = number of iterations
% res    = final residual norm
% flag   = 0 - converged within maxit / 1 - not converged
% Ivec   = vector of infeasibilities throughout the PCG
% Muvec  = vector of centralities throughout the PCG
%
% Author: Filippo Zanetti, 2021
% 
%----------------------------------------------------

v2=r2./g+ds_p;
n=length(g);
Rho=1;
tolvar=1e-3;
itmin=5;

dg0=zeros(n,1);
xi1=(s.*dg0)./g;
xi2=fq(dg0)+xi1;
xi_p=fq(dg_p);
r=b-xi2;
r0=norm(r);
z=fp(r);
p=z;
iter=0;
dg=dg0;
rhon=dot(r,z);
flag=0;

I=norm(r1/nAtm);Mu=mu;
vari=[];varmu=[];

while norm(r)>tol*r0 && iter<itmax
    rho=rhon;
    w1=(s.*p)./g;
    w2=fq(p)+w1;
    alfa=rho/dot(w2,p);
    dg=dg+alfa*p;
    xi1=xi1+alfa*w1;
    xi2=xi2+alfa*w2;
    r=r-alfa*w2;
    z=fp(r);
    rhon=dot(r,z);
    beta=rhon/rho;
    p=z+beta*p;
    iter=iter+1;
    
    if iter>=itmin-5
        %IPM convergence indicators
        dgn=dg+dg_p;
        dsn=v2-xi1;

        idg = dgn < 0;
        ids = dsn < 0;
        alpha_g = Rho*min([1;-g(idg)./dgn(idg)]);
        alpha_s = Rho*min([1;-s(ids)./dsn(ids)]);

        infN=(r1+alpha_s*v2-alpha_g*(xi2+xi_p)+(alpha_g-alpha_s)*xi1)/nAtm;
        muN=(g+alpha_g*dgn)'*(s+alpha_s*dsn)/n;

        Iold=I;
        I=norm(infN);
        Muold=Mu;
        Mu=muN;
        
        vari=[vari;abs(Iold-I)/Iold];
        varmu=[varmu;abs(Muold-Mu)/Muold];

        if iter>=itmin && mean(vari(end-4:end))<tolvar && mean(varmu(end-4:end))<tolvar
            break
        end
    end
end

res=norm(r);

if iter==itmax && norm(r)>tol*r0
    flag=1;
end


