function y=applyQ(x,fA,fAt,Q2)
% % % % % % % % % % % % % % % % % % % %
% Applies Normal Equations matrix
%
% Q = A'*A + Q2
%
% to vector x
%
% Author: Filippo Zanetti, 2021
 % % % % % % % % % % % % % % % % % % %

y=fAt(fA(x));
y=y+Q2*x;
