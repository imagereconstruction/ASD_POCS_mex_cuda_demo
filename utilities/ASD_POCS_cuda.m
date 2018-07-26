function [ m_res, out] = ASD_POCS_cuda(P, para)
%ASD_POCS_cuda Summary of this function goes here
%   ASD_POCS_cuda reconstruction for fan/cone beam CT
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
% para.beta_red = beta_red;
%% Example: 
% (todo)
% %%
%%
if isfield(para,'m0') && ~isempty(para.m0)
    m0 = para.m0;
else
    if isfield(para,'initBySART') && ~isempty(para.initBySART)
        if para.initBySART==true
            m0 = SART_cuda_inline(P,para);
        else
            m0 = zeros(para.nx*para.ny*para.nz,1);
        end
    else
        m0 = zeros(para.nx*para.ny*para.nz,1);
    end
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

if isfield(para,'numOfgradients') && ~isempty(para.numOfgradients)
    ng = para.numOfgradients;
else
    ng = 20;
end

if isfield(para,'alpha') && ~isempty(para.alpha)
    alpha = para.alpha;
else
    alpha = 0.02;
end

if isfield(para,'alpha_red') && ~isempty(para.alpha_red)
    alpha_red = para.alpha_red;
else
    alpha_red = 0.95;
end

if isfield(para,'beta_red') && ~isempty(para.beta_red)
    beta_red = para.beta_red;
else
    beta_red = 0.995;
end

if isfield(para,'r_max') && ~isempty(para.r_max)
    r_max = para.r_max;
else
    r_max = 0.95;
end

if isfield(para,'epsilon') && ~isempty(para.epsilon)
    epsilon = para.epsilon;
else
    epsilon = eps;
end

if isfield(para,'beta') && ~isempty(para.beta)
    beta = para.beta;
    if (beta>=2.0 || beta<=0.0)
        warning('Bad value for para.lambda!');
        warning('For convergence, lambda should be in (0.0 2.0)!');
    end
else
    beta = 1;
end

if isfield(para,'silenceMode') && ~isempty(para.silenceMode)
    silenceMode = para.silenceMode;
else
    silenceMode = true;
end
[~,nv] = size(P);
if (nv ~= numel(para.phi_vec) )
    error('Size of Projection and phi_vec mismatch!');
end
RMSE = zeros(maxnumIter,1);
m = m0;
for k = 1:maxnumIter
    if (~silenceMode)
        fprintf('Outer k=%d.\n',k); 
    end
    %% SART
    m_pre = m;
    dd2 = 0;
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
        dd2 = dd2 + (norm(cor0(:)))^2;
        cor = (cor0)./sumR;
        %% 反投
        [bProj,sumC] = Atx_cone_gpu(single(cor(:)),para);
        bProj = double(bProj);
        sumC = double(sumC);
        sumC(sumC==0) = 1;
        %% 补偿
        m = m + beta*(bProj./ sumC);
        m((m<0)) = 0;
        %%
        clear W;
    end
    %%    
    disp(['iterNo.=' num2str(k,'%04d') ', data error='  num2str(dd2,'%1.10f')]);
    dd = dd2^0.5;
    m_res = m;
    if isfield(para,'I0') && ~isempty(para.I0)
        o=m(:)-para.I0(:);
        RMSE(k) =  sqrt(sum(o(:).^2)/(para.nx*para.ny*para.nz));
        fprintf(' Iter %d, rmse = %f \n',k,RMSE(k));
    end
    %
    dp = norm(m(:)-m_pre(:),2);
    relativeChange = dp/numel(m);
    if relativeChange<=stopRelChg
        break;
    end
    
    if k==1
       m_k1 = m_pre; 
    end
    
    f0 = m;
    if para.nz>=2
        m = imgTVGradMin3D((reshape(m,[para.nx,para.ny,para.nz])),...
            alpha, ng, 1, reshape(m_k1,[para.nx,para.ny,para.nz]));
        m = m(:);
    else
        m = imgTVGradMin((reshape(m,para.nx,para.ny)),...
            alpha, ng, 1, reshape(m_k1,para.nx,para.ny));
        m = m(:);
    end
    
    dg = norm(m(:)-f0(:),2);
    if dg>r_max*dp && dd>epsilon
        alpha = alpha*alpha_red;
    end  
    beta = beta*beta_red;
end
m_res = reshape(m_res,[para.nx,para.ny,para.nz]);
out.IterationNum = k;
out.dataTolerance = dd2;
out.beta = beta;
out.alpha = alpha;
out.reconImg = m_res;
if isfield(para,'I0') && ~isempty(para.I0)
    out.RMSE = RMSE(1:k);
end

end

