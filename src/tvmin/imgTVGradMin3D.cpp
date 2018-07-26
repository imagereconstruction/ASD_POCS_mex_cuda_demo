// TV minimization by gradient minimization
// parameters: I (image), TV-alpha, number of TV iterations, positivity, previous-image (optional)
#include <mex.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "matrix.h"
// author: caiailong, cai.ailong@163.com
int showUsage()
{
    mexPrintf("---imgTVGradMin3D - minimize image TV by gradient minimization for 3D data---\n");
    mexPrintf("usage 1: \t I2 = imgTVGradMin3D(I, alpha, nTVIter, pos, I0)\n");
    mexPrintf("usage 2: \t I2 = imgTVGradMin3D(I, alpha, nTVIter, pos)\n");
    return 0;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    //if incorrect number of inputs, print usage
    if(nrhs!=4 && nrhs!=5)
	{
        showUsage();
        return;
    }
    if( 	mxGetClassID(prhs[0]) == mxDOUBLE_CLASS)
	{
            
        const mwSize *dim;
        dim = mxGetDimensions(prhs[0]);
		double 	*img_orig	= mxGetPr(prhs[0]);  //the original image
		double 	alpha 		= mxGetScalar(prhs[1]);            
		int 	nTVIter 	= (int)(mxGetScalar(prhs[2]));  
		int 	pos 		= (int)(mxGetScalar(prhs[3]));  
		double 	*img_old;
        const mwSize *dim0;
		double *im, *v;

		int		rowN = (mwSignedIndex)mxGetM(prhs[0]);  
		int		colN = (mwSignedIndex)mxGetN(prhs[0]);  
//         int     Nx = dim[0];
//         int     Ny = dim[1];
//                 int     Nz = dim[2];
        int     Nx = dim[0];
        int     Ny = dim[1];
        int     Nz = dim[2];
        
		int imgN = Nx * Ny * Nz;
		double *img = (double *)calloc(Nx*Ny*Nz, sizeof(double));
		memcpy(img, img_orig, imgN*sizeof(double)); //copy original image to different buffer for calculations

		if (nrhs == 5)
		{
			img_old 		= mxGetPr(prhs[4]); 
            dim0 = mxGetDimensions(prhs[4]);
			int		rowN0 = (mwSignedIndex)mxGetM(prhs[4]);  
			int		colN0 = (mwSignedIndex)mxGetN(prhs[4]);  
            int     Nx0   = dim0[0];
            int     Ny0   = dim0[1];
            int     Nz0   = dim0[2];
			//
            if (Nx != Nx0 || Ny != Ny0 || Nz != Nz0)
            {
				mexPrintf("img0 dimensions do not match with img\n");
				return;
			}
		}
		else
        {
            img_old = (double *)calloc(Nx*Ny*Nz, sizeof(double)); //initialize with zeros
        }
		im = (double *)calloc((Nx+2)*(Ny+2)*(Nz+2), sizeof(double));
		v = (double *)calloc(Nx*Ny*Nz, sizeof(double));
	
		//TV minimization
		double d = 0;
		for(int i=0; i<imgN; i++)
		{
			d += pow((img[i] - img_old[i]),2);
		}
		d = sqrt(d);
	
		for(int i = 0; i< nTVIter; i++)
        {
            //printf("IterNo. = %d\n",i);
			const double eps=1e-8;
			int impos;
			double norm = 0;
	
			//get extended-image ready
			impos = 0;
			for(int h=1;h<Nz+1;h++)
            {
				for(int r=1;r<Ny+1;r++)
                {
                    for (int c=1;c<Nx+1;c++)
                    {
// 					im[r+c*(Ny+2)+h*(Ny+2)*(Nx+2)] = img[impos];
                        //im[r+c*(Ny+2)+h*(Ny+2)*(Nx+2)] = img[r+c*(Ny+2)+h*(Ny+2)*(Nx+2)];
                        im[r+c*(Ny+2)+h*(Ny+2)*(Nx+2)] = img[impos];
                        impos++;
                    }
				}
			}///////
	
			impos=0;
            #pragma omp parallel for
			for(int h=1;h<Nz+1;h++)
            {
				for(int r=1;r<Ny+1;r++)
                {
                    for(int  c=1;c<Nx+1;c++)
                    {
                        int p000 = c*(Ny+2)+r+h*(Ny+2)*(Nx+2);//  
                        
                        int p_100 = (c-1)*(Ny+2)+r+h*(Ny+2)*(Nx+2);//
                        int p0_10 = c*(Ny+2)+(r-1)+h*(Ny+2)*(Nx+2);//
                        int p00_1 = c*(Ny+2)+r+(h-1)*(Ny+2)*(Nx+2);//
                        
                        int p001 = c*(Ny+2)+r+(h+1)*(Ny+2)*(Nx+2);//
                        int p_101 = (c-1)*(Ny+2)+r+(h+1)*(Ny+2)*(Nx+2);//
                        int p0_11 = c*(Ny+2)+(r-1)+(h+1)*(Ny+2)*(Nx+2);//
                        
                        int p010 = c*(Ny+2)+(r+1)+h*(Ny+2)*(Nx+2);//
                        int p_110 = (c-1)*(Ny+2)+(r+1)+h*(Ny+2)*(Nx+2);//
                        int p01_1 = c*(Ny+2)+(r+1)+(h-1)*(Ny+2)*(Nx+2);//
                        
                        int p100 = (c+1)*(Ny+2)+r+h*(Ny+2)*(Nx+2);//
                        int p1_10 = (c+1)*(Ny+2)+(r-1)+h*(Ny+2)*(Nx+2);//
                        int p10_1 = (c+1)*(Ny+2)+r+(h-1)*(Ny+2)*(Nx+2);//
                        
                        double vv1,vv2,vv3,vv4;
                        vv1 = (6*im[p000] - 2*(im[p_100]+im[p0_10]+im[p00_1]))/sqrt(eps+pow(im[p000]-im[p_100],2)\
                                +pow(im[p000]-im[p0_10],2)+pow(im[p000]-im[p00_1],2));
                        vv2 = (2*im[p000] - 2*im[p001])/sqrt(eps+pow(im[p001]-im[p_101],2)+pow(im[p001]-im[p0_11],2)+pow(im[p001]-im[p000],2));
                        vv3 = (2*im[p000] - 2*im[p010])/sqrt(eps+pow(im[p010]-im[p_110],2)+pow(im[p010]-im[p000],2)+pow(im[p010]-im[p01_1],2));
                        vv4 = (2*im[p000] - 2*im[p100])/sqrt(eps+pow(im[p100]-im[p000],2)+pow(im[p100]-im[p1_10],2)+pow(im[p100]-im[p10_1],2));
                        
                        int idxv = (c-1) + (r-1)*Nx + (h-1)*Nx*Ny;
                        v[idxv] = vv1 + vv2 + vv3 + vv4;
                        
//                         
//                         v[impos] = vv1 + vv2 + vv3 + vv4;
//                         norm += v[impos] * v[impos];
//                         impos++;
                    }
				}
			}
            
            for(int i2=0;i2<Nx*Ny*Nz;i2++)
            {
                norm += v[i2]*v[i2];
            }
//             
			norm = sqrt(norm);
	
			for(int i=0;i<imgN;i++)
            {
				img[i] -= alpha * d * v[i] / norm;
			}
		} //end of TV iterations
	
		if (pos == 1 && nTVIter > 0)
        { //whether positivity was enforced?
			for(int i=0; i<imgN; i++)
            {
				if(img[i]<0)
					img[i]=0; //enforce positivity constraint
			} 
		}	
		
        plhs[0] = mxCreateNumericArray(3, dim,mxDOUBLE_CLASS, mxREAL );
		memcpy(mxGetPr(plhs[0]), img, Nx*Ny*Nz*sizeof(double));

		if(im) { free(im); im = NULL; }
		if(img) { free(img); img = NULL; }
		if(v) { free(v); v = NULL; }
		if(nrhs == 4) { free(img_old); v = NULL; }

/////////////////////////////////

//////////////////////
		return;
	}
	else{
		mexPrintf("Invalid data format. Please make sure the input image data is in double format.\n");
		return;
	}
}

