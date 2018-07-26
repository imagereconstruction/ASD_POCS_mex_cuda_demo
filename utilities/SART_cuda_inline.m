function [ m, out] = SART_cuda_inline(P, para)
%SART_CUDA Summary of this function goes here
%   SART reconstruction for fan/cone beam CT
%%
% para.nx = Nx;
% para.ny = Ny;
% para.nz = Nz;
% para.SO = sod;
% para.OD = sdd - sod;
% para.vxlsize = voxel;
% para.detsize = dt;
% para.na = Uy;
% para.nb = 1;
% para.phi_vec = phi_vec; vector in degrees

%%
if isfield(para,'m0') && ~isempty(para.m0)
    m0 = para.m0;
else
    m0 = zeros(para.nx*para.ny*para.nz,1);
end

if isfield(para,'stopRelChg') && ~isempty(para.stopRelChg)
    stopRelChg = para.stopRelChg;
else
    stopRelChg = eps;
end

if isfield(para,'maxnumIter') && ~isempty(para.maxnumIter)
    maxnumIter = para.maxnumIter;
else
    maxnumIter = 5;
end

if isfield(para,'lambda') && ~isempty(para.lambda)
    lambda = para.lambda;
    if (lambda>=2.0 || lambda<=0.0)
        warning('Bad value for para.lambda!');
        warning('For convergence, lambda should be in (0.0 2.0)!');
    end
else
    lambda = 1;
end

if isfield(para,'silenceMode') && ~isempty(para.silenceMode)
    silenceMode = para.silenceMode;
else
    silenceMode = 0;
end
[~,nv] = size(P);
if (nv ~= numel(para.phi_vec) )
    error('Size of Projection and phi_vec mismatch!');
end
RMSE = zeros(maxnumIter,1);
m = m0;
for k = 1:1
    if (~silenceMode)
        fprintf('Outer k=%d.\n',k); 
    end
    %% SART
    m_pre = m;
    datatol = 0;
    for ii = 1:length(para.phi_vec)     
        theta_i = para.phi_vec(ii)/180*pi;
        if (~silenceMode)
            fprintf('inner ii=%d.\n',ii);
        end
        %% 正投，作差
        para.phi = -1*theta_i;
        [proj,sumR] = Ax_cone_gpu(single(m(:)),para);
        proj = double(proj);
        sumR = double(sumR);
        sumR(sumR==0) = 1;
        cor0 = P(:,ii) - proj;
        datatol = datatol + (norm(cor0(:)))^2;
        cor = (cor0)./sumR;
        %% 反投
        [bProj,sumC] = Atx_cone_gpu(single(cor(:)),para);
        bProj = double(bProj);
        sumC = double(sumC);
        sumC(sumC==0) = 1;
        %% 补偿
        m = m + lambda*(bProj./ sumC);
        m((m<0)) = 0;
        %%
        clear W;
    end
    %%    
    disp(['iterNo.=' num2str(k,'%04d') ', data error='  num2str(datatol,'%1.10f')]);
    if isfield(para,'I0') && ~isempty(para.I0)
        o=m(:)-para.I0(:);
        RMSE(k) =  sqrt(sum(o(:).^2)/(para.nx*para.ny*para.nz));
        fprintf(' Iter %d, rmse = %f \n',k,RMSE(k));
    end
    relativeChange = norm(m(:)-m_pre(:),2)/numel(m);
    if relativeChange<=stopRelChg
        break;
    end
end

out.IterationNum = k;
out.dataTolerance = datatol;
out.lambda = lambda;
out.reconImg = m;
if isfield(para,'I0') && ~isempty(para.I0)
    out.RMSE = RMSE(1:k);
end
end

