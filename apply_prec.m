function y=apply_prec(x,g,s,CS,n)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% Applies 2x2 diagonal block preconditioner
%
% P = G^-1*S + [ alpha+rho*(c11^2+c21^2)     beta+rho*(c11*c12+c21*c22) ]
%              [ beta+rho*(c11*c12+c21*c22)  alpha+rho*(C12^2+c22^2)    ] 
%
% to vector x:
% y = P\x
%
% Author: Filippo Zanetti, 2020
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

x1=x(1:n/2);
x2=x(n/2+1:end);

Dgs=s./g;
Dgs1=Dgs(1:n/2);
Dgs2=Dgs(n/2+1:end);

D11=Dgs1+CS.e11;    %vector
D22=Dgs2+CS.e22;    %vector
D12=CS.e12;         %number

rhs=x2-D12.*(x1./D11);
y2=rhs./(D22-(D12.^2)./D11);
y1=(x1-D12.*y2)./D11;
y=[y1;y2];

