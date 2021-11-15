function y=applyNE(x,fA,fAt,Q2,g,s)
% % % % % % % % % % % % % % % % % % % %
% Applies Normal Equations matrix
%
% Q + G^-1*S = A'*A + Q2 + G^-1*S
%
% to vector x
%
% Author: Filippo Zanetti, 2020
 % % % % % % % % % % % % % % % % % % %

y=fAt(fA(x));
y=y+Q2*x;
y=y+(x.*s)./g;