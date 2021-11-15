% Example computations related to material decomposition X-ray tomography.
% WITH REAL DATA: CHALKPOXY
% Here we apply (Tikhonov regularization) and new regularization term to solve 
% the normal equations modified for two image system, using the interior 
% point optimization method.
%
% Needs:
% phantoms for two materials
% functions: A2x2mult_matrixfree,
% A2x2mult_matrixfree_rotang,A2x2Tmult_matrixfree
%
% Jennifer Mueller and Samuli Siltanen, October 2012
% Modified by Salla 6.10.2020

clear all;
close all;

% Measure computation time later; start clocking here
tic
%% Choises for the user
% Choose the size of the unknown. The image has size NxN.
N       = 64;
% Choose the regularization parameters
alpha  = 150;             
beta   = 0.8*alpha;

% % Choose measurement angles (given in degrees, not radians). 
Nang       = 720; % odd number is preferred
ang        = [0:(Nang-1)]*720/Nang;

% %% Attenuation coefficients from NIST-database (divided by density).
% c11     = 1.491; % PVC    30kV  (Low energy)
% c12     = 8.561; % Iodine 30kV
% c21     = 0.456; % PVC    50kV  (High energy)
% c22     = 12.32; % Iodine 50kV

% %% Construct phantom
% % Option 4: Bone
% g1      = imread('phantom_bone1.bmp');
% g2      = imread('phantom_bone2.bmp');
% %%
% % Select one of the channels
% g1      = g1(:,:,1);
% g2      = g2(:,:,1);
% 
% g1 = imbinarize(g1);
% g2 = imbinarize(g2);
% 
% % Change to double
% g1      = double(g1);
% g2      = double(g2);
% 
% % Resize the image
% g1      = imresize(g1, [N N], 'nearest');
% g2      = imresize(g2, [N N], 'nearest');
% %%
% % % Avoid inverse crime by rotating the object (interpolation)
% % g1      = imrotate(M1,rotang,'bilinear','crop');
% % g2      = imrotate(M2,rotang,'bilinear','crop');
% 
% % Combine the vectors
% g      = [g1(:);g2(:)];

%% Start reconstruction
% Simulate noisy measurements avoiding inverse crime 
%m       = A2x2mult_matrixfree(c11,c12,c21,c22,g,ang,N); 
% Load the high energy sinogram
binningFactor = 8;
filename = ['chalkpoxy_sinogram_binning_' num2str(binningFactor)];
load(filename);
mhigh = sinogram;
% Load the low energy sinogram
filename = ['chalkpoxy_sinogram_binning_low' num2str(binningFactor)];
load(filename);
mlow = sinogram;
m      = [mhigh(:);mlow(:)];
 
% Add noise/poisson noise
% m  = m + noiselevel*max(abs(m(:)))*randn(size(m));
% m = imnoise(m,'poisson');


%%
% Kokeillaan saadaanko tuotettua ASTRA matriisi

addpath(genpath('Z:\Documents\MATLAB\astra-1.8'))
addpath(genpath('Z:\Documents\MATLAB\spot-1.2'))
addpath(genpath('Z:\Documents\MATLAB\HelTomo'))
load('CTData');

%% Make reconstruction!!!!
% Decide the dimensions?
xDim = 4096;
yDim = xDim;

% %% Create a reconstruction with ASTRA cuda FBP
[ A ] = create_ct_operator_2d_fan_astra_cuda( CtData, xDim, yDim ); % No need for this
% [ recon2 ] = tomorecon_2d_fan_fbp_astra_cuda( CtData, xDim, yDim );
% figure('Name', 'Reconstruction from ASTRA CUDA fbp');
% imshow(recon2,[])
%% Solve the problem

%*** New Q2-regularization term****
% We create matrix Q2 by writing normal matrix M=[alpha, beta; beta, alpha]
% and then taking a kronecker product with opEye. Every element of M will 
% be multiplied with opEye which results a block matrix Q2.
pMatrix = [alpha beta; beta alpha];
opMatrix = speye(N^2);
Q2 = kron(pMatrix,opMatrix);


% Build struct with coefficients for preconditioner.
% rho is the average diagonal element of A'*A, obtained with 
% random sampling.
CS=struct();
[~,rho]=sampling_AtA(200,N,m,ang);
CS.e11=alpha+(c11^2+c21^2)*rho;
CS.e12=beta+(c11*c12+c21*c22)*rho;
CS.e22=alpha+(c12^2+c22^2)*rho;

%Solve QP with IPM
[g]=ipm_UE_corr(@(x) A2x2mult_matrixfree(c11,c12,c21,c22,x,ang,N),...
           @(x) A2x2Tmult_matrixfree(c11,c12,c21,c22,x,ang),...
           m,Q2,2*N^2,CS); %gives out g as a vector

       

%% Separate the reconstructions
IPM1 = reshape(g(1:(end/2),1:end),N,N);
IPM2 = reshape(g((end/2)+1:end,1:end),N,N);

%% % Save result to file
%save XRsparse_aIPM_Bone IPM1 IPM2 alpha M1 M2

%% Calculate the error
% Target 1
E1    = norm(g1(:)-IPM1(:))/norm(g1(:)); % Square error
SSIM1      = ssim(IPM1,g1); % Structural similarity index
RMSE1      = sqrt(mean((g1(:) - IPM1(:)).^2));  % Root Mean Squared Error
% Target 2
E2    = norm(g2(:)-IPM2(:))/norm(g2(:)); % Square error
SSIM2      = ssim(IPM2,g2); % Structural similarity index
RMSE2      = sqrt(mean((g2(:) - IPM2(:)).^2));  % Root Mean Squared Error
% Total errors calculated as mean of the errors of both reconstructions
E_mean      = sqrt(E1*E2);
SSIM        = (SSIM1+SSIM2)/2;
RMSE        = (RMSE1 + RMSE2)/2;

%% Save image
% Samu's version for saving:
% normalize the values of the images between 0 and 1
im1        = g1;     % material 1: PVC
im2        = g2;     % material 2: Iodine
im3        = IPM1;   % reco of PVC
im4        = IPM2;   % reco of Iodine

MIN        = min([min(im1(:)),min(im2(:)),min(im3(:)),min(im4(:))]);
MAX        = max([max(im1(:)),max(im2(:)),max(im3(:)),max(im4(:))]);
im1        = im1-MIN;
im1        = im1/(MAX-MIN);
im2        = im2-MIN;
im2        = im2/(MAX-MIN);
im3        = im3-MIN;
im3        = im3/(MAX-MIN);
im4        = im4-MIN;
im4        = im4/(MAX-MIN);
imwrite(uint8(255*im3),'IPM_reco_Bone1.png')
imwrite(uint8(255*im4),'IPM_reco_Bone2.png')

imwrite(uint8(255*im1),'IPM_phantom_Bone1.png')
imwrite(uint8(255*im2),'IPM_phantom_Bone2.png')

save IPM_Bone_for_segmentations im1 im2 im3 im4 N

% HaarPSI index:
% the HaarPSI of two identical images will be exactly one and the
% HaarPSI of two completely different images will be close to zero. 
% Please make sure that the values are given in the [0,255] interval!
Haarpsi1 = HaarPSI(255*im1,255*im3,0);
Haarpsi2 = HaarPSI(255*im2,255*im4,0);
Haarpsi  = (Haarpsi1+Haarpsi2)/2;
%% Take a look at the results
figure(4);
% Original phantom1
subplot(2,2,1);
imagesc(g1);
colormap gray;
axis square;
axis off;
title({'Phantom Bone1, Ground truth'});
% Reconstruction of phantom1
subplot(2,2,2)
imagesc(IPM1);
colormap gray;
axis square;
axis off;
title(['Approximate error ', num2str(round(E1*100,1)), '%, \alpha=', num2str(alpha), ', \beta=', num2str(beta)]);
% Original target2
subplot(2,2,3)
imagesc(g2);
colormap gray;
axis square;
axis off;
title({'Phantom Bone2, Ground truth'});
% Reconstruction of target2
subplot(2,2,4)
imagesc(IPM2);
%imagesc(imrotate(CG2,-45,'bilinear','crop'));
colormap gray;
axis square;
axis off;
title(['Approximate error ' num2str(round(E2*100,1)),'%']);

%% Print the error

fprintf('Error calculations for IPM approach: \n');
fprintf('E1 %.2f     RMSE1 %.2f      SSIM1 %.2f     haarPSI1 %.2f  alpha %.2f beta %.2f\n', E1,RMSE1,SSIM1,Haarpsi1,alpha, beta)
fprintf('E2 %.2f     RMSE2 %.2f      SSIM2 %.2f     haarPSI2 %.2f  alpha %.2f beta %.2f\n', E2,RMSE2,SSIM2,Haarpsi2,alpha, beta)
fprintf('\n');
fprintf('Mean error calculations for IPM approach: \n');
fprintf('E_mean %.2f     RMSE %.2f      SSIM %.2f     haarPSI %.2f  alpha %.2f beta %.2f\n', E_mean,RMSE,SSIM,Haarpsi,alpha, beta)