// TV minimization by gradient minimization
// parameters: I (image), TV-alpha, number of TV iterations, positivity, previous-image (optional)
#include <mex.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

int showUsage()
{
    mexPrintf("---imgTVGradMin - minimize image TV by gradient minimization ---\n");
    mexPrintf("usage 1: \t I2 = imgTVGradMin(I, alpha, nTVIter, pos, I0)\n");
    mexPrintf("usage 2: \t I2 = imgTVGradMin(I, alpha, nTVIter, pos)\n");
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
            
		double 	*img_orig	= mxGetPr(prhs[0]);  //the original image
		double 	alpha 		= mxGetScalar(prhs[1]);            
		int 	nTVIter 	= (int)(mxGetScalar(prhs[2]));  
		int 	pos 		= (int)(mxGetScalar(prhs[3]));  
		double 	*img_old;
		double *im, *v;

		int		rowN = (mwSignedIndex)mxGetM(prhs[0]);  
		int		colN = (mwSignedIndex)mxGetN(prhs[0]);  
		int imgN = colN * rowN;
		double *img = (double *)calloc(rowN*colN, sizeof(double));
		memcpy(img, img_orig, imgN*sizeof(double)); //copy original image to different buffer for calculations

		if (nrhs == 5)
        {
			img_old 		= mxGetPr(prhs[4]);   
			int		rowN0 = (mwSignedIndex)mxGetM(prhs[4]);  
			int		colN0 = (mwSignedIndex)mxGetN(prhs[4]);  
			if (rowN0 != rowN || colN0 != colN)
            {
				mexPrintf("img0 dimensions do not match with img\n");
				return;
			}
		}
		else
		img_old = (double *)calloc(rowN*colN, sizeof(double)); //initialize with zeros
		im = (double *)calloc((rowN+2)*(colN+2), sizeof(double));
		v = (double *)calloc(rowN*colN, sizeof(double));

		//TV minimization
		double d = 0;
		for(int i=0; i<imgN; i++)
			d += pow((img[i] - img_old[i]),2);
		d = sqrt(d);
	
		for(int i = 0; i< nTVIter; i++)
        {
			const double eps=1e-8;
			int impos;
			double norm = 0;
	
			//get extended-image ready
			impos = 0;
			for(int c=1;c<colN+1;c++)
            {
				for(int r=1;r<rowN+1;r++)
                {
					im[r+c*(rowN+2)] = img[impos];
					impos++;
				}
			}
	
			impos=0;
			for(int c=1;c<colN+1;c++)
            {
				for(int r=1;r<rowN+1;r++)
                {
					int p00  = r + c*(rowN+2);
					int p01  = r + (c+1)*(rowN+2);
					int p0_1 = r + (c-1)*(rowN+2);
					int p10  = (r+1) + c*(rowN+2);
					int p_10 = (r-1) + c*(rowN+2);
					int p1_1 = (r+1) + (c-1)*(rowN+2);
					int p_11 = (r-1) + (c+1)*(rowN+2);                
					double v1,v2,v3;
	
					v1 = ((im[p00]-im[p_10])+(im[p00]-im[p0_1])) / sqrt(eps + pow(im[p00]-im[p_10],2) + pow(im[p00]-im[p0_1],2));
					v2 = -(im[p10]-im[p00]) / sqrt(eps + pow(im[p10]-im[p00],2) + pow(im[p10]-im[p1_1],2));
					v3 = -(im[p01]-im[p00]) / sqrt(eps + pow(im[p01]-im[p00],2) + pow(im[p01]-im[p_11],2));
	
					v[impos] = v1 + v2 + v3;
					norm += v[impos] * v[impos];
					impos++;
				}
			}
			norm = sqrt(norm);
	
			for(int i=0;i<imgN;i++)
            {
				img[i] -= alpha * d * v[i] / norm;
			}
		} //end of TV iterations
	
		if (pos == 1 && nTVIter > 0)
        { //whether posititivity was enforced?
			for(int i=0; i<imgN; i++)
            {
				if(img[i]<0)
					img[i]=0; //enforce positivity constraint
			} 
			mexPrintf(" +TV-PC ");
            mexEvalString("drawnow;");
		}	
		
	
		plhs[0] = mxCreateDoubleMatrix(rowN, colN, mxREAL);
		memcpy(mxGetPr(plhs[0]), img, rowN*colN*sizeof(double));

		if(im) { free(im); im = NULL; }
		if(img) { free(img); img = NULL; }
		if(v) { free(v); v = NULL; }
		if(nrhs == 4) { free(img_old); v = NULL; }

/////////////////////////////////

//////////////////////
		return;
	}
	else
    {
		mexPrintf("Invalid data format. Please make sure the input image data is in double format.\n");
		return;
	}
}

