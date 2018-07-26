#include <mex.h>
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>


/* backward projeciton CUDA code:
   Atx_cone_mf_gpu_new      */
extern "C" void Atx_cone_mf_gpu_new(float *X,float *y,float *sc,float cos_phi,float sin_phi,float *y_det,float *z_det,
	float SO,float OD,float scale,float dy_det,float dz_det,float dz,int nx,int ny,int nz,int na,int nb);

/* forward projection CUDA code
   Ax_cone_mf_gpu_new       */
extern "C" void Ax_cone_mf_gpu_new(float *X,float *y,float *sr,float cos_phi,float sin_phi,float *y_det,float *z_det,
float SO,float OD,float scale,float dz,int nx,int ny,int nz,int na,int nb);

//void main()
void mexFunction(int nlhs, mxArray *plhs[], int nrhs,
	const mxArray *prhs[])
{
	if (nrhs>2 || nrhs <=1)
	{
		mexPrintf("Usage: [y,sr] = Ax_cone_mf_gpu(X,parameters)\n");
		mexPrintf("X: image of size nx*ny*nz\n");
		mexPrintf("parameters.nx: nx\n");
		mexPrintf("parameters.ny: ny\n");
		mexPrintf("parameters.nz: nz (=1 for 2D image)\n");
		mexPrintf("parameters.SO: sod\n");
		mexPrintf("parameters.OD: odd\n");
		mexPrintf("parameters.vxlsize: voxel size in x, y and z dimension\n");
		mexPrintf("parameters.detsize: detector bin size\n");
		mexPrintf("parameters.phi: rotation angle(scalar)\n");
		mexPrintf("parameters.na: number of bins in horizontal direction\n");
		mexPrintf("parameters.nb: number of bins in vertical direction\n");
		mexPrintf("output  y: projection data\n");
		mexPrintf("output sr: row sum of system matrix\n");
		mexErrMsgTxt("Error usage!");
		return;
	}
	
	int nx,ny,nz,na,nb;
    float *X,*y,*sr,cos_phi,sin_phi,SO,OD,scale,det,dz,phi;

    X=(float*)mxGetData(prhs[0]);

    /************************************************************************/
	if(NULL==mxGetField(prhs[1],0,"SO"))
	{
		mexPrintf("parameters.SO is not assigned! Please check!\n");
		mexErrMsgTxt("Error input argument parameter.SO!");
		return;
	}
    SO=(float)mxGetScalar(mxGetField(prhs[1],0,"SO"));
    /************************************************************************/
	if(NULL==mxGetField(prhs[1],0,"OD"))
	{
		mexPrintf("parameters.OD is not assigned! Please check!\n");
		mexErrMsgTxt("Error input argument parameter.OD!");
		return;
	}    
    OD=(float)mxGetScalar(mxGetField(prhs[1],0,"OD"));
    /************************************************************************/
	if(NULL==mxGetField(prhs[1],0,"vxlsize"))
	{
		mexPrintf("parameters.vxlsize is not assigned! Please check!\n");
		mexErrMsgTxt("Error input argument parameter.vxlsize!");
		return;
	}   
    scale=(float)mxGetScalar(mxGetField(prhs[1],0,"vxlsize"));
    /************************************************************************/
	if(NULL==mxGetField(prhs[1],0,"nx"))
	{
		mexPrintf("parameters.nx is not assigned! Please check!\n");
		mexErrMsgTxt("Error input argument parameter.nx!");
		return;
	}   
    nx=(int)mxGetScalar(mxGetField(prhs[1],0,"nx"));
    /************************************************************************/
	if(NULL==mxGetField(prhs[1],0,"ny"))
	{
		mexPrintf("parameters.ny is not assigned! Please check!\n");
		mexErrMsgTxt("Error input argument parameter.ny!");
		return;
	}  
    ny=(int)mxGetScalar(mxGetField(prhs[1],0,"ny"));
    /************************************************************************/
	if(NULL==mxGetField(prhs[1],0,"nz"))
	{
		mexPrintf("parameters.nz is not assigned! Please check!\n");
		mexErrMsgTxt("Error input argument parameter.nz!");
		return;
	}  
    nz=(int)mxGetScalar(mxGetField(prhs[1],0,"nz"));
    /************************************************************************/
	if(NULL==mxGetField(prhs[1],0,"phi"))
	{
		mexPrintf("parameters.phi is not assigned! Please check!\n");
		mexErrMsgTxt("Error input argument parameter.phi!");
		return;
	}  
    phi=(float)mxGetScalar(mxGetField(prhs[1],0,"phi"));
    /************************************************************************/
	if(NULL==mxGetField(prhs[1],0,"detsize"))
	{
		mexPrintf("parameters.detsize is not assigned! Please check!\n");
		mexErrMsgTxt("Error input argument parameter.detsize!");
		return;
	}  
    det =(float)mxGetScalar(mxGetField(prhs[1],0,"detsize"));
    /************************************************************************/
	if(NULL==mxGetField(prhs[1],0,"na"))
	{
		mexPrintf("parameters.na is not assigned! Please check!\n");
		mexErrMsgTxt("Error input argument parameter.na!");
		return;
	}  
	na=(int)mxGetScalar(mxGetField(prhs[1],0,"na"));
    /************************************************************************/
	if(NULL==mxGetField(prhs[1],0,"nb"))
	{
		mexPrintf("parameters.nb is not assigned! Please check!\n");
		mexErrMsgTxt("Error input argument parameter.nb!");
		return;
	}  
	nb=(int)mxGetScalar(mxGetField(prhs[1],0,"nb"));
    /************************************************************************/
    int numelOfInputPrj = 0;
	// 检查输入数据
	numelOfInputPrj = (int)mxGetNumberOfElements(prhs[0]);
	if((numelOfInputPrj)!=(nx*ny*nz))
	{
		mexPrintf("Size of the first input data (X) mismatch nx*ny*nz! Please check!\n");
		mexErrMsgTxt("Error size of input argument X!");
		return;
	}
    /************************************************************************/
	cos_phi = cos(phi);
	sin_phi = sin(phi);

	dz = 1.0;
	SO = SO/scale;
	OD = OD/scale;
	det = det/scale;
		
	plhs[0]=mxCreateNumericMatrix(na*nb,1,mxSINGLE_CLASS,mxREAL);
    y=(float*)mxGetData(plhs[0]);

	plhs[1]=mxCreateNumericMatrix(na*nb,1,mxSINGLE_CLASS,mxREAL);
    sr=(float*)mxGetData(plhs[1]);

	float *y_det = (float*)malloc(sizeof(float)*na);
	float *z_det = (float*)malloc(sizeof(float)*nb);
	for(int i=0;i<na;i++)
	{
		y_det[i] = (float)((((float)i) - ((float)na)/2.0 + 0.5)*det);
	}
	for(int i=0;i<nb;i++)
	{
		z_det[i] = (float)((((float)i) - ((float)nb)/2.0 + 0.5)*det);
	}
		
	Ax_cone_mf_gpu_new(X,y,sr,cos_phi,sin_phi,y_det,z_det,SO,OD,scale,dz,nx,ny,nz,na,nb);
		
}