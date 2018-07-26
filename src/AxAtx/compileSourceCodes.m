%% mex 
% Please set the correct path for include files and library files
% Please choose proper compute_xx and sm_xx  according to the NVIDIA GPU
% device
% 2017.10.16
% caiailong
close all;clear;clc;
!"%VS110COMNTOOLS%vsvars32.bat" & nvcc -c -m64 -arch compute_61 -code sm_61 Ax_cone_gpu.cu
Lpath='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64';
Ipathname = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include';
mex('-output','Ax_cone_gpu.mexw64',['-L' Lpath],['-I' Ipathname],'-lcudart','Ax_main_test.cpp','Ax_cone_gpu.obj')
mex('-output','Atx_cone_gpu.mexw64',['-L' Lpath],['-I' Ipathname],'-lcudart','Atx_main_test.cpp','Ax_cone_gpu.obj')
