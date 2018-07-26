%% test 3D TV minimization
clc
clear
close all
%%
N = 128;
x0 = phantom3d(N);
x = x0(:,1:end-20,end/2-60:end/2+59);
figure;imshow(x(:,:,end/2),[]);
%%
n = rand(size(x))/10;
figure;imshow(n(:,:,end/2),[]);
xn = x + n;
figure;imshow(xn(:,:,end/2),[]);
%%
alpha = 0.01;
TViter = 20;
pvty = 1;
tic;
fxn = imgTVGradMin3D(xn,alpha,TViter,pvty);
toc;
figure;imshow(xn(:,:,end/2),'InitialMagnification',300);
figure;imshow(fxn(:,:,end/2),'InitialMagnification',300);

% for ii = 1:N
% figure;imshow(fxn(:,:,ii),'InitialMagnification',300);
% close all
% end