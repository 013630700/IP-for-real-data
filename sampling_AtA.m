function [S,rho]=sampling_AtA(dim,N,m,ang)
% % % % % % % % % % % % % % % % % % % % % % % % % %
% S = sampling_AtA (dim,N,m,ang)
% Random sampling of the entries of matris A'*A.
% S is a (dim)x(dim) submatrix of A'*A that can
% be used to determine the coefficient rho
% (i.e. the average diagonal entry of A'*A).
% 
% Author: Filippo Zanetti, 2020
% % % % % % % % % % % % % % % % % % % % % % % % % %

I=randperm(N^2,dim);
S=zeros(dim,dim);

for ni=1:dim
    i=I(ni);
    ei=zeros(N^2,1);
    ei(i)=1;
    AtAei=AtA_mult(ei,N,m,ang);
    for nj=1:dim
        j=I(nj);
        ej=sparse(N^2,1);
        ej(j)=1;
        S(ni,nj)=ej'*AtAei;
    end
end
rho=mean(diag(S));

end

function y=AtA_mult(x,N,m,ang)

%performs matrix vector product (A'*A)*x using radon transform.

g=reshape(x,[N N]);
Ag=radon(g,ang);
Ag=Ag(:);

m1 = reshape(Ag, [length(m)/(2*length(ang)) length(ang)]);
corxn = 7.65;

AtAx = iradon(m1,ang,'none');
AtAx = AtAx(2:end-1,2:end-1);
AtAx = corxn*AtAx;
y=AtAx(:);

end


