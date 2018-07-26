%% reconstruction by POCS + BM sparisty
%% a simple demo 
%% 
clc;
close all;
path(path,genpath(pwd));
%% 扫描几何参数
%%
dwnsmp = 1;
dt = 0.148*dwnsmp;%%探测器像素大小
sod = 405.9410;
sdd = 906.2510;
%% 图像参数
%% 
%% 设置重建参数
n = 512;
load('P10240000_head_slice190.mat')
I3d = img_slice190/max(img_slice190(:));
[Nx,Ny,Nz] = size(I3d);
% Nz = 1;
view_num = 180;
Uy = 2*Nx;
Vz = 2*Nz;
d = sod/sdd*dt*Uy;
voxel = d/Nx;
%%
para.nx = Nx;
para.ny = Ny;
para.nz = Nz;
para.SO = sod;
para.OD = sdd - sod;
para.vxlsize = voxel;
para.detsize = dt;
para.na = Uy;
para.nb = Vz;
para.nv = view_num;
%%
t=180/view_num;
theta_vecT=[1:1:view_num]*t;
N_views = length(theta_vecT);
W_c = cell(N_views,1);
sumC_c = cell(N_views,1);
sumR_c = cell(N_views,1);
para.phi_vec = theta_vecT;
%%
P = zeros(para.na*para.nb,para.nv);
for ii = 1:length(theta_vecT)     
    theta_i = theta_vecT(ii)/180*pi;
    para.phi = -1*theta_i;
    [prj,ttmmpp] = Ax_cone_gpu(single(I3d(:)),para);
    imshow(reshape(prj,para.na,para.nb),[]);
    P(:,ii) = double(prj(:));
end
%%
para.maxnumIter = 400;%4000
para.alpha = 0.2;%0.2
para.alpha_red = 0.95;%0.995
para.beta_red = 1.0;
para.I0 = I3d;
[ m2, out] = ASD_POCS_cuda(P, para);
figure;
subplot(1,2,2);
imshow(m2,[0.3 0.5]);
subplot(1,2,1);
imshow(I3d,[0.3 0.5]);
figure;
subplot(1,2,2);
imshow(m2(120:220,270:370)',[0.3 0.5]);
subplot(1,2,1);
imshow(I3d(120:220,270:370)',[0.3 0.5]);
figure;imshow(abs(m2-I3d),[0 0.01]);
% figure;imshow(m,[0.195 0.205]);
figure;plot(out.RMSE);


