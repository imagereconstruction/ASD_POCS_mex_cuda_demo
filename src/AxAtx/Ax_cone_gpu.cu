#include <math.h>
#include <malloc.h>
#define ABS(a) (a>0?a:-(a))
#define MAX(a,b) (a>b?a:b)
#define MIN(a,b) (a<b?a:b)

#define BLOCK_SIZE_x 16
#define BLOCK_SIZE_y 16

const float eps=1e-8;

extern "C" void Atx_cone_mf_gpu_new(float *X,float *y,float *sc,float cos_phi,float sin_phi,float *y_det,float *z_det,
	float SO,float OD,float scale,float dy_det,float dz_det,float dz,int nx,int ny,int nz,int na,int nb);


inline __device__ float find_l(float x1_0,float y1_0,float x2_0,float y2_0,float dx,float dy,float x,float y)
{   
	float l=0,dx2,dy2,a,b,slope,tmp,tmp2,xi[2],yi[2],x1,y1;
	int i;

	a=x2_0-x1_0;
	b=y2_0-y1_0;
	dx2=dx/2.0f;
	dy2=dy/2.0f;

	if(a==0)
	{   
		tmp=ABS(x1_0-x);
		if(tmp<=dx2){l=dy;}
	}
	else
	{   
		if(b==0)
		{   
			tmp=ABS(y1_0-y);
		if(tmp<=dy2)
		{
			l=dx;
		}
	}
	else
	{   
		x1=x1_0-x;y1=y1_0-y;
		i=0;
		if(ABS(a)>ABS(b))
		{   
			slope=b/a;
			tmp=slope*(-x1)+y1;
			tmp2=slope*dx2;
			if(ABS(tmp-tmp2)<=dy2)
			{
				xi[i]=-dx2;yi[i]=tmp-tmp2;i++;
			}
			if(ABS(tmp+tmp2)<=dy2)
			{
				xi[i]=dx2;yi[i]=tmp+tmp2;i++;
			}

			if(i<2)
			{   
				slope=a/b;
				tmp=slope*(-y1)+x1;
				tmp2=slope*dy2;
				if(ABS(tmp-tmp2)<=dx2)
				{
					yi[i]=-dy2;xi[i]=tmp-tmp2;i++;
				}
				if(i<2)
				{   
					if(ABS(tmp+tmp2)<=dx2)
					{
					yi[i]=dy2;xi[i]=tmp+tmp2;i++;
					}
				}
			}
	}
	else
	{   
		slope=a/b;
		tmp=slope*(-y1)+x1;
		tmp2=slope*dy2;
		if(ABS(tmp-tmp2)<=dx2)
		{
			yi[i]=-dy2;xi[i]=tmp-tmp2;i++;
		}
		if(ABS(tmp+tmp2)<=dx2)
		{
			yi[i]=dy2;xi[i]=tmp+tmp2;i++;
		}

		if(i<2)
		{   
			slope=b/a;
			tmp=slope*(-x1)+y1;
			tmp2=slope*dx2;
			if(ABS(tmp-tmp2)<=dy2)
			{
				xi[i]=-dx2;yi[i]=tmp-tmp2;i++;
			}
			if(i<2)
			{   
				if(ABS(tmp+tmp2)<=dy2)
				{
					xi[i]=dx2;yi[i]=tmp+tmp2;i++;
				}
			}
		}
		}

		if(i==2)
		{   tmp=xi[1]-xi[0];tmp2=yi[1]-yi[0];
		l=(float)sqrt(tmp*tmp+tmp2*tmp2);
		}
		}
	}
	return l;
}


inline __device__ float find_l_3d(float x1_0,float y1_0,float z1_0,float x2_0,float y2_0,float z2_0,float dx,float dy,float dz,float x,float y,float z)
	// assuming c~=0
	// A method for computing the intersecting length of a voxel with a infinitely-narrow beam
	// A better formula will be supplied to improve the speed.
{   
	float l=0,dx2,dy2,dz2,a,b,c,slope,tmp[2],tmp2[2],tmpx,tmpy,tmpz,xi[2],yi[2],zi[2],x1,y1,z1;
	int i;

	a=x2_0-x1_0;b=y2_0-y1_0;c=z2_0-z1_0;
	dx2=dx/2.0f;dy2=dy/2.0f;dz2=dz/2.0f;

	if(a==0)
	{l=find_l(y1_0,z1_0,y2_0,z2_0,dy,dz,y,z);}
	else
	{   if(b==0)
		{l=find_l(x1_0,z1_0,x2_0,z2_0,dx,dz,x,z);}
	else
	{   x1=x1_0-x;y1=y1_0-y;z1=z1_0-z;
	//            x2=x2_0-x;y2=y2_0-y;z2=z2_0-z;

	i=0;
	if(ABS(a)>ABS(b))
	{   slope=b/a;tmp[0]=slope*(-x1)+y1;tmp2[0]=slope*dx2;
	slope=c/a;tmp[1]=slope*(-x1)+z1;tmp2[1]=slope*dx2;
	if(ABS(tmp[0]-tmp2[0])<=dy2&&ABS(tmp[1]-tmp2[1])<=dz2)
	{xi[i]=-dx2;yi[i]=tmp[0]-tmp2[0];zi[i]=tmp[1]-tmp2[1];i++;}
	if(ABS(tmp[0]+tmp2[0])<=dy2&&ABS(tmp[1]+tmp2[1])<=dz2)
	{xi[i]=dx2;yi[i]=tmp[0]+tmp2[0];zi[i]=tmp[1]+tmp2[1];i++;}

	if(i<2)
	{   slope=a/b;tmp[0]=slope*(-y1)+x1;tmp2[0]=slope*dy2;
	slope=c/b;tmp[1]=slope*(-y1)+z1;tmp2[1]=slope*dy2;
	if(ABS(tmp[0]-tmp2[0])<=dx2&&ABS(tmp[1]-tmp2[1])<=dz2)
	{xi[i]=tmp[0]-tmp2[0];yi[i]=-dy2;zi[i]=tmp[1]-tmp2[1];i++;}
	if(i<2)
	{   if(ABS(tmp[0]+tmp2[0])<=dx2&&ABS(tmp[1]+tmp2[1])<=dz2)
	{xi[i]=tmp[0]+tmp2[0];yi[i]=dy2;zi[i]=tmp[1]+tmp2[1];i++;}
	}
	}

	if(i<2)
	{   slope=a/c;tmp[0]=slope*(-z1)+x1;tmp2[0]=slope*dz2;
	slope=b/c;tmp[1]=slope*(-z1)+y1;tmp2[1]=slope*dz2;
	if(ABS(tmp[0]-tmp2[0])<=dx2&&ABS(tmp[1]-tmp2[1])<=dy2)
	{xi[i]=tmp[0]-tmp2[0];yi[i]=tmp[1]-tmp2[1];zi[i]=-dz2;i++;}
	if(i<2)
	{   if(ABS(tmp[0]+tmp2[0])<=dx2&&ABS(tmp[1]+tmp2[1])<=dy2)
	{xi[i]=tmp[0]+tmp2[0];yi[i]=tmp[1]+tmp2[1];zi[i]=dz2;i++;}
	}
	}
	}
	else
	{   slope=a/b;tmp[0]=slope*(-y1)+x1;tmp2[0]=slope*dy2;
	slope=c/b;tmp[1]=slope*(-y1)+z1;tmp2[1]=slope*dy2;
	if(ABS(tmp[0]-tmp2[0])<=dx2&&ABS(tmp[1]-tmp2[1])<=dz2)
	{xi[i]=tmp[0]-tmp2[0];yi[i]=-dy2;zi[i]=tmp[1]-tmp2[1];i++;}
	if(ABS(tmp[0]+tmp2[0])<=dx2&&ABS(tmp[1]+tmp2[1])<=dz2)
	{xi[i]=tmp[0]+tmp2[0];yi[i]=dy2;zi[i]=tmp[1]+tmp2[1];i++;}

	if(i<2)
	{   slope=b/a;tmp[0]=slope*(-x1)+y1;tmp2[0]=slope*dx2;
	slope=c/a;tmp[1]=slope*(-x1)+z1;tmp2[1]=slope*dx2;
	if(ABS(tmp[0]-tmp2[0])<=dy2&&ABS(tmp[1]-tmp2[1])<=dz2)
	{xi[i]=-dx2;yi[i]=tmp[0]-tmp2[0];zi[i]=tmp[1]-tmp2[1];i++;}
	if(i<2)
	{   if(ABS(tmp[0]+tmp2[0])<=dy2&&ABS(tmp[1]+tmp2[1])<=dz2)
	{xi[i]=dx2;yi[i]=tmp[0]+tmp2[0];zi[i]=tmp[1]+tmp2[1];i++;}
	}
	}

	if(i<2)
	{   slope=a/c;tmp[0]=slope*(-z1)+x1;tmp2[0]=slope*dz2;
	slope=b/c;tmp[1]=slope*(-z1)+y1;tmp2[1]=slope*dz2;
	if(ABS(tmp[0]-tmp2[0])<=dx2&&ABS(tmp[1]-tmp2[1])<=dy2)
	{xi[i]=tmp[0]-tmp2[0];yi[i]=tmp[1]-tmp2[1];zi[i]=-dz2;i++;}
	if(i<2)
	{   if(ABS(tmp[0]+tmp2[0])<=dx2&&ABS(tmp[1]+tmp2[1])<=dy2)
	{xi[i]=tmp[0]+tmp2[0];yi[i]=tmp[1]+tmp2[1];zi[i]=dz2;i++;}
	}
	}
	}

	if(i==2)
	{   tmpx=xi[1]-xi[0];tmpy=yi[1]-yi[0];tmpz=zi[1]-zi[0];
	l=(float)sqrt(tmpx*tmpx+tmpy*tmpy+tmpz*tmpz);
	}
	}
	}
	return l;
}




__global__ void Atx_cone_mf_gpu_new_kernel(float *x,float *y,float *sc,float cos_phi,float sin_phi,float *y_det,float *z_det,
	float SO,float OD,float scale,float dy_det,float dz_det,float dz,int nx,int ny,int nz,int na,int nb)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int ix=bx*BLOCK_SIZE_x+tx;
	int iy2=by*BLOCK_SIZE_y+ty;

	if(ix<nx&&iy2<ny*nz)
	{   
		int nx2,ny2,nz2,na2,nb2,ia,ib,iy,iz,na_min,na_max,nb_min,nb_max,idx;
		float xc,yc,zc,xr,yr,SD,l,tmp,x1,y1,z1,x2,y2,z2,d;

		SD=SO+OD;
		na2=na/2;nb2=nb/2;
		nx2=nx/2;ny2=ny/2;nz2=nz/2;
		d=(float)sqrt((1+dz*dz)/2);

		iz=(int)floor((float)iy2/(float)ny);
		iy=iy2-iz*ny;
		idx=iz*ny*nx+iy*nx+ix;

		zc=(float)(iz+0.5-nz2)*dz;
		yc=(float)(iy+0.5-ny2);
		xc=(float)(ix+0.5-nx2);


		xr=cos_phi*xc+sin_phi*yc;
		yr=-sin_phi*xc+cos_phi*yc;

		tmp=SD/((xr+SO)*dy_det);
		na_max=(int)floor((yr+1)*tmp+na2);
		na_min=(int)floor((yr-1)*tmp+na2);

		tmp=SD/((xr+SO)*dz_det);
		nb_max=(int)floor((zc+d)*tmp+nb2);
		nb_min=(int)floor((zc-d)*tmp+nb2);

		for(ib=MAX(0,nb_min);ib<=MIN(nb_max,nb-1);ib++)
		{   
			for(ia=MAX(0,na_min);ia<=MIN(na_max,na-1);ia++)
			{   
				x1=cos_phi*(-SO);
				y1=sin_phi*(-SO);
				z1=0.0;
				x2=cos_phi*OD-sin_phi*y_det[ia];
				y2=sin_phi*OD+cos_phi*y_det[ia];
				z2=z_det[ib];
				l=find_l_3d(x1,y1,z1,x2,y2,z2,1.0,1.0,dz,xc,yc,zc);
				x[idx]+=l*y[ib*na+ia];
				sc[idx]+=l;
			}
		}
	 }
}

__global__ void set2zero(float *x,int nx,int nyz)
{	int bx=blockIdx.x;
int by=blockIdx.y;
int tx=threadIdx.x;
int ty=threadIdx.y;

int ix=bx*BLOCK_SIZE_x+tx;
int iy=by*BLOCK_SIZE_y+ty;

if(ix<nx&&iy<nyz)
{x[iy*nx+ix]=0;}
}

__global__ void scalex(float *x,int nx,int nyz,float scale)
{	int bx=blockIdx.x;
int by=blockIdx.y;
int tx=threadIdx.x;
int ty=threadIdx.y;

int ix=bx*BLOCK_SIZE_x+tx;
int iy=by*BLOCK_SIZE_y+ty;

if(ix<nx&&iy<nyz)
{x[iy*nx+ix]*=scale;}
}

extern "C" void Atx_cone_mf_gpu_new(float *X,float *y,float *sc,float cos_phi,float sin_phi,float *y_det,float *z_det,
	float SO,float OD,float scale,float dy_det,float dz_det,float dz,int nx,int ny,int nz,int na,int nb)
{   
	float *x_d,*y_d,*sc_d,*y_det_d,*z_det_d;
	int nd,N;

	N=nx*ny*nz;
	nd=na*nb;

	cudaMalloc(&y_d,nd*sizeof(float));
	cudaMalloc(&x_d,N*sizeof(float));
	cudaMalloc(&sc_d,N*sizeof(float));
	cudaMalloc(&y_det_d,na*sizeof(float));cudaMemcpy(y_det_d,y_det,na*sizeof(float),cudaMemcpyHostToDevice);
	cudaMalloc(&z_det_d,nb*sizeof(float));cudaMemcpy(z_det_d,z_det,nb*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(y_d,y,nd*sizeof(float),cudaMemcpyHostToDevice);

	dim3 dimBlock(BLOCK_SIZE_x,BLOCK_SIZE_y);
	dim3 dimGrid_t((nx+dimBlock.x-1)/dimBlock.x,(ny*nz+dimBlock.y-1)/dimBlock.y);

	set2zero<<<dimGrid_t, dimBlock>>>(x_d,nx,ny*nz);
	set2zero<<<dimGrid_t, dimBlock>>>(sc_d,nx,ny*nz);

	
	Atx_cone_mf_gpu_new_kernel<<<dimGrid_t, dimBlock>>>(x_d,y_d,sc_d,cos_phi,sin_phi,y_det_d,z_det_d,
		SO,OD,scale,dy_det,dz_det,dz,nx,ny,nz,na,nb);
	cudaThreadSynchronize();
	scalex<<<dimGrid_t, dimBlock>>>(x_d,nx,ny*nz,scale);
	scalex<<<dimGrid_t, dimBlock>>>(sc_d,nx,ny*nz,scale);
	cudaMemcpy(X,x_d,N*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(sc,sc_d,N*sizeof(float),cudaMemcpyDeviceToHost);

	cudaFree(x_d);cudaFree(y_d);cudaFree(sc_d);cudaFree(y_det_d);cudaFree(z_det_d);
}




extern "C" void Ax_cone_mf_gpu_new(float *X,float *y,float *sr,float cos_phi,float sin_phi,float *y_det,float *z_det,
float SO,float OD,float scale,float dz,int nx,int ny,int nz,int na,int nb);

__global__ void Ax_cone_mf_gpu_kernel_new(float *x,float *y,float *sr,float cos_phi,float sin_phi,float *y_det,float *z_det,
float SO,float OD,float scale,float dz,int nx,int ny,int nz,int na,int nb)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tx0=threadIdx.x;
	int ty0=threadIdx.y;

	int ia=bx*BLOCK_SIZE_x+tx0;
	int ib=by*BLOCK_SIZE_y+ty0;

    if(ia<na&&ib<nb)
    {
		int nx2,ny2,nz2,id,ix,iy,iz,cx1,cx2,cy1,cy2,cz1,cz2;
		float x1,y1,x2,y2,z1,z2,xx1,yy1,zz1,xx2,yy2,zz2,slope1,slope2,l,d,tmp,rx,ry,rz;

		nx2=nx/2;
        ny2=ny/2;
        nz2=nz/2;


		id=ib*na+ia;

        x1=cos_phi*(-SO);
        y1=sin_phi*(-SO);
        z1=0.0;
		x2=cos_phi*OD-sin_phi*y_det[ia];
        y2=sin_phi*OD+cos_phi*y_det[ia];
		z2=z_det[ib];

		y[id]=0;
		sr[id]=0;
        // assuming z1-z2 is small
        if(ABS(x1-x2)>ABS(y1-y2))
        {   slope1=(y2-y1)/(x2-x1);
            slope2=(z2-z1)/(x2-x1);
            for(ix=0;ix<nx;ix++)
            {   xx1=(float)(ix-nx2);xx2=xx1+1;
                if(slope1>=0)
                {   yy1=y1+slope1*(xx1-x1)+ny2;
                    yy2=y1+slope1*(xx2-x1)+ny2;
                }
                else
                {   yy1=y1+slope1*(xx2-x1)+ny2;
                    yy2=y1+slope1*(xx1-x1)+ny2;
                }
                cy1=(int)floor(yy1);
                cy2=(int)floor(yy2);
                if(slope2>=0)
                {   zz1=(z1+slope2*(xx1-x1))/dz+nz2;
                    zz2=(z1+slope2*(xx2-x1))/dz+nz2;
                }
                else
                {   zz1=(z1+slope2*(xx2-x1))/dz+nz2;
                    zz2=(z1+slope2*(xx1-x1))/dz+nz2;
                }
                cz1=(int)floor(zz1);
                cz2=(int)floor(zz2);

                if(cy2==cy1)
                {   if(cy1>=0&&cy1<=ny-1)
                    {   if(cz2==cz1)
                        {   if(cz1>=0&&cz1<=nz-1)// 11
                            {   d=yy2-yy1;tmp=(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2);l=(float)sqrt((d*d+1)*(tmp+(z1-z2)*(z1-z2))/tmp);
                                iy=cy1;iz=cz1;y[id]+=l*x[iz*ny*nx+iy*nx+ix];sr[id]+=l;
                            }
                        }
                        else
                        {   if(cz2>0&&cz2<nz)// 12
                            {   d=yy2-yy1;tmp=(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2);l=(float)sqrt((d*d+1)*(tmp+(z1-z2)*(z1-z2))/tmp);
                                rz=(cz2-zz1)/(zz2-zz1);
                                iy=cy1;iz=cz1;y[id]+=rz*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=rz*l;
                                iy=cy1;iz=cz2;y[id]+=(1-rz)*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=(1-rz)*l;
                            }
                            else
                            {   if(cz2==0)// 13
                                {   d=yy2-yy1;tmp=(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2);l=(float)sqrt((d*d+1)*(tmp+(z1-z2)*(z1-z2))/tmp);
                                    rz=(cz2-zz1)/(zz2-zz1);
                                    iy=cy1;iz=cz2;y[id]+=(1-rz)*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=(1-rz)*l;
                                }
                                if(cz2==nz)// 14
                                {   d=yy2-yy1;tmp=(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2);l=(float)sqrt((d*d+1)*(tmp+(z1-z2)*(z1-z2))/tmp);
                                    rz=(cz2-zz1)/(zz2-zz1);
                                    iy=cy1;iz=cz1;y[id]+=rz*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=rz*l;
                                }
                            }
                        }
                    }
                }
                else
                {   if(cy2>0&&cy2<ny)
                    {   if(cz2==cz1)
                        {   if(cz1>=0&&cz1<=nz-1)// 21
                            {   d=yy2-yy1;tmp=(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2);l=(float)sqrt((d*d+1)*(tmp+(z1-z2)*(z1-z2))/tmp);
                                ry=(cy2-yy1)/d;
                                iy=cy1;iz=cz1;y[id]+=ry*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=ry*l;
                                iy=cy2;iz=cz1;y[id]+=(1-ry)*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=(1-ry)*l;
                            }
                        }
                        else
                        {   if(cz2>0&&cz2<nz)// 22
                            {   d=yy2-yy1;tmp=(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2);l=(float)sqrt((d*d+1)*(tmp+(z1-z2)*(z1-z2))/tmp);
                                ry=(cy2-yy1)/d;rz=(cz2-zz1)/(zz2-zz1);
                                if(ry>rz)
                                {   iy=cy1;iz=cz1;y[id]+=rz*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=rz*l;
                                    iy=cy1;iz=cz2;y[id]+=(ry-rz)*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=(ry-rz)*l;
                                    iy=cy2;iz=cz2;y[id]+=(1-ry)*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=(1-ry)*l;
                                }
                                else
                                {   iy=cy1;iz=cz1;y[id]+=ry*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=ry*l;
                                    iy=cy2;iz=cz1;y[id]+=(rz-ry)*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=(rz-ry)*l;
                                    iy=cy2;iz=cz2;y[id]+=(1-rz)*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=(1-rz)*l;
                                }
                            }
                            else
                            {   if(cz2==0)// 23
                                {   d=yy2-yy1;tmp=(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2);l=(float)sqrt((d*d+1)*(tmp+(z1-z2)*(z1-z2))/tmp);
                                    ry=(cy2-yy1)/d;rz=(cz2-zz1)/(zz2-zz1);
                                    if(ry>rz)
                                    {   iy=cy1;iz=cz2;y[id]+=(ry-rz)*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=(ry-rz)*l;
                                        iy=cy2;iz=cz2;y[id]+=(1-ry)*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=(1-ry)*l;
                                    }
                                    else
                                    {   iy=cy2;iz=cz2;y[id]+=(1-rz)*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=(1-rz)*l;
                                    }
                                }
                                if(cz2==nz)// 24
                                {   d=yy2-yy1;tmp=(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2);l=(float)sqrt((d*d+1)*(tmp+(z1-z2)*(z1-z2))/tmp);
                                    ry=(cy2-yy1)/d;rz=(cz2-zz1)/(zz2-zz1);
                                    if(ry>rz)
                                    {   iy=cy1;iz=cz1;y[id]+=rz*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=rz*l;
                                    }
                                    else
                                    {   iy=cy1;iz=cz1;y[id]+=ry*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=ry*l;
                                        iy=cy2;iz=cz1;y[id]+=(rz-ry)*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=(rz-ry)*l;
                                    }
                                }
                            }
                        }
                    }
                    else
                    {   if(cy2==0)
                        {   if(cz2==cz1)
                            {   if(cz1>=0&&cz1<=nz-1)// 31
                                {   d=yy2-yy1;tmp=(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2);l=(float)sqrt((d*d+1)*(tmp+(z1-z2)*(z1-z2))/tmp);
                                    ry=(cy2-yy1)/d;
                                    iy=cy2;iz=cz1;y[id]+=(1-ry)*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=(1-ry)*l;
                                }
                            }
                            else
                            {   if(cz2>0&&cz2<nz)// 32
                                {   d=yy2-yy1;tmp=(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2);l=(float)sqrt((d*d+1)*(tmp+(z1-z2)*(z1-z2))/tmp);
                                    ry=(cy2-yy1)/d;rz=(cz2-zz1)/(zz2-zz1);
                                    if(ry>rz)
                                    {   iy=cy2;iz=cz2;y[id]+=(1-ry)*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=(1-ry)*l;
                                    }
                                    else
                                    {   iy=cy2;iz=cz1;y[id]+=(rz-ry)*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=(rz-ry)*l;
                                        iy=cy2;iz=cz2;y[id]+=(1-rz)*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=(1-rz)*l;
                                    }
                                }
                                else
                                {   if(cz2==0)// 33
                                    {   d=yy2-yy1;tmp=(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2);l=(float)sqrt((d*d+1)*(tmp+(z1-z2)*(z1-z2))/tmp);
                                        ry=(cy2-yy1)/d;rz=(cz2-zz1)/(zz2-zz1);
                                        if(ry>rz)
                                        {   iy=cy2;iz=cz2;y[id]+=(1-ry)*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=(1-ry)*l;
                                        }
                                        else
                                        {   iy=cy2;iz=cz2;y[id]+=(1-rz)*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=(1-rz)*l;
                                        }
                                    }
                                    if(cz2==nz)// 34
                                    {   d=yy2-yy1;tmp=(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2);l=(float)sqrt((d*d+1)*(tmp+(z1-z2)*(z1-z2))/tmp);
                                        ry=(cy2-yy1)/d;rz=(cz2-zz1)/(zz2-zz1);
                                        if(ry>rz)
                                        {
                                        }
                                        else
                                        {   iy=cy2;iz=cz1;y[id]+=(rz-ry)*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=(rz-ry)*l;
                                        }
                                    }
                                }
                            }
                        }

                        if(cy2==ny)
                        {   if(cz2==cz1)
                            {   if(cz1>=0&&cz1<=nz-1)// 41
                                {   d=yy2-yy1;tmp=(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2);l=(float)sqrt((d*d+1)*(tmp+(z1-z2)*(z1-z2))/tmp);
                                    ry=(cy2-yy1)/d;
                                    iy=cy1;iz=cz1;y[id]+=ry*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=ry*l;
                                }
                            }
                            else
                            {   if(cz2>0&&cz2<nz)// 42
                                {   d=yy2-yy1;tmp=(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2);l=(float)sqrt((d*d+1)*(tmp+(z1-z2)*(z1-z2))/tmp);
                                    ry=(cy2-yy1)/d;rz=(cz2-zz1)/(zz2-zz1);
                                    if(ry>rz)
                                    {   iy=cy1;iz=cz1;y[id]+=rz*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=rz*l;
                                        iy=cy1;iz=cz2;y[id]+=(ry-rz)*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=(ry-rz)*l;
                                    }
                                    else
                                    {   iy=cy1;iz=cz1;y[id]+=ry*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=ry*l;
                                    }
                                }
                                else
                                {   if(cz2==0)// 43
                                    {   d=yy2-yy1;tmp=(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2);l=(float)sqrt((d*d+1)*(tmp+(z1-z2)*(z1-z2))/tmp);
                                        ry=(cy2-yy1)/d;rz=(cz2-zz1)/(zz2-zz1);
                                        if(ry>rz)
                                        {   iy=cy1;iz=cz2;y[id]+=(ry-rz)*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=(ry-rz)*l;
                                        }
                                        else
                                        {
                                        }
                                    }
                                    if(cz2==nz)// 44
                                    {   d=yy2-yy1;tmp=(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2);l=(float)sqrt((d*d+1)*(tmp+(z1-z2)*(z1-z2))/tmp);
                                        ry=(cy2-yy1)/d;rz=(cz2-zz1)/(zz2-zz1);
                                        if(ry>rz)
                                        {   iy=cy1;iz=cz1;y[id]+=rz*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=rz*l;
                                        }
                                        else
                                        {   iy=cy1;iz=cz1;y[id]+=ry*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=ry*l;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        else
        {   slope1=(x2-x1)/(y2-y1);
            slope2=(z2-z1)/(y2-y1);
            for(iy=0;iy<ny;iy++)
            {   yy1=(float)(iy-ny2);yy2=yy1+1;
                if(slope1>=0)
                {   xx1=x1+slope1*(yy1-y1)+nx2;
                    xx2=x1+slope1*(yy2-y1)+nx2;
                }
                else
                {   xx1=x1+slope1*(yy2-y1)+nx2;
                    xx2=x1+slope1*(yy1-y1)+nx2;
                }
                cx1=(int)floor(xx1);
                cx2=(int)floor(xx2);
                if(slope2>=0)
                {   zz1=(z1+slope2*(yy1-y1))/dz+nz2;
                    zz2=(z1+slope2*(yy2-y1))/dz+nz2;
                }
                else
                {   zz1=(z1+slope2*(yy2-y1))/dz+nz2;
                    zz2=(z1+slope2*(yy1-y1))/dz+nz2;
                }
                cz1=(int)floor(zz1);
                cz2=(int)floor(zz2);

                if(cx2==cx1)
                {   if(cx1>=0&&cx1<=nx-1)
                    {   if(cz2==cz1)
                        {   if(cz1>=0&&cz1<=nz-1)// 11
                            {   d=xx2-xx1;tmp=(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2);l=(float)sqrt((d*d+1)*(tmp+(z1-z2)*(z1-z2))/tmp);
                                ix=cx1;iz=cz1;y[id]+=l*x[iz*ny*nx+iy*nx+ix];sr[id]+=l;
                            }
                        }
                        else
                        {   if(cz2>0&&cz2<nz)// 12
                            {   d=xx2-xx1;tmp=(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2);l=(float)sqrt((d*d+1)*(tmp+(z1-z2)*(z1-z2))/tmp);
                                rz=(cz2-zz1)/(zz2-zz1);
                                ix=cx1;iz=cz1;y[id]+=rz*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=rz*l;
                                ix=cx1;iz=cz2;y[id]+=(1-rz)*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=(1-rz)*l;
                            }
                            else
                            {   if(cz2==0)// 13
                                {   d=xx2-xx1;tmp=(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2);l=(float)sqrt((d*d+1)*(tmp+(z1-z2)*(z1-z2))/tmp);
                                    rz=(cz2-zz1)/(zz2-zz1);
                                    ix=cx1;iz=cz2;y[id]+=(1-rz)*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=(1-rz)*l;
                                }
                                if(cz2==nz)// 14
                                {   d=xx2-xx1;tmp=(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2);l=(float)sqrt((d*d+1)*(tmp+(z1-z2)*(z1-z2))/tmp);
                                    rz=(cz2-zz1)/(zz2-zz1);
                                    ix=cx1;iz=cz1;y[id]+=rz*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=rz*l;
                                }
                            }
                        }
                    }
                }
                else
                {   if(cx2>0&&cx2<nx)
                    {   if(cz2==cz1)
                        {   if(cz1>=0&&cz1<=nz-1)// 21
                            {   d=xx2-xx1;tmp=(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2);l=(float)sqrt((d*d+1)*(tmp+(z1-z2)*(z1-z2))/tmp);
                                rx=(cx2-xx1)/d;
                                ix=cx1;iz=cz1;y[id]+=rx*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=rx*l;
                                ix=cx2;iz=cz1;y[id]+=(1-rx)*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=(1-rx)*l;
                            }
                        }
                        else
                        {   if(cz2>0&&cz2<nz)// 22
                            {   d=xx2-xx1;tmp=(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2);l=(float)sqrt((d*d+1)*(tmp+(z1-z2)*(z1-z2))/tmp);
                                rx=(cx2-xx1)/d;rz=(cz2-zz1)/(zz2-zz1);
                                if(rx>rz)
                                {   ix=cx1;iz=cz1;y[id]+=rz*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=rz*l;
                                    ix=cx1;iz=cz2;y[id]+=(rx-rz)*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=(rx-rz)*l;
                                    ix=cx2;iz=cz2;y[id]+=(1-rx)*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=(1-rx)*l;
                                }
                                else
                                {   ix=cx1;iz=cz1;y[id]+=rx*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=rx*l;
                                    ix=cx2;iz=cz1;y[id]+=(rz-rx)*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=(rz-rx)*l;
                                    ix=cx2;iz=cz2;y[id]+=(1-rz)*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=(1-rz)*l;
                                }
                            }
                            else
                            {   if(cz2==0)// 23
                                {   d=xx2-xx1;tmp=(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2);l=(float)sqrt((d*d+1)*(tmp+(z1-z2)*(z1-z2))/tmp);
                                    rx=(cx2-xx1)/d;rz=(cz2-zz1)/(zz2-zz1);
                                    if(rx>rz)
                                    {   ix=cx1;iz=cz2;y[id]+=(rx-rz)*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=(rx-rz)*l;
                                        ix=cx2;iz=cz2;y[id]+=(1-rx)*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=(1-rx)*l;
                                    }
                                    else
                                    {   ix=cx2;iz=cz2;y[id]+=(1-rz)*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=(1-rz)*l;
                                    }
                                }
                                if(cz2==nz)// 24
                                {   d=xx2-xx1;tmp=(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2);l=(float)sqrt((d*d+1)*(tmp+(z1-z2)*(z1-z2))/tmp);
                                    rx=(cx2-xx1)/d;rz=(cz2-zz1)/(zz2-zz1);
                                    if(rx>rz)
                                    {   ix=cx1;iz=cz1;y[id]+=rz*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=rz*l;
                                    }
                                    else
                                    {   ix=cx1;iz=cz1;y[id]+=rx*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=rx*l;
                                        ix=cx2;iz=cz1;y[id]+=(rz-rx)*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=(rz-rx)*l;
                                    }
                                }
                            }
                        }
                    }
                    else
                    {   if(cx2==0)
                        {   if(cz2==cz1)
                            {   if(cz1>=0&&cz1<=nz-1)// 31
                                {   d=xx2-xx1;tmp=(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2);l=(float)sqrt((d*d+1)*(tmp+(z1-z2)*(z1-z2))/tmp);
                                    rx=(cx2-xx1)/d;
                                    ix=cx2;iz=cz1;y[id]+=(1-rx)*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=(1-rx)*l;
                                }
                            }
                            else
                            {   if(cz2>0&&cz2<nz)// 32
                                {   d=xx2-xx1;tmp=(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2);l=(float)sqrt((d*d+1)*(tmp+(z1-z2)*(z1-z2))/tmp);
                                    rx=(cx2-xx1)/d;rz=(cz2-zz1)/(zz2-zz1);
                                    if(rx>rz)
                                    {   ix=cx2;iz=cz2;y[id]+=(1-rx)*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=(1-rx)*l;
                                    }
                                    else
                                    {   ix=cx2;iz=cz1;y[id]+=(rz-rx)*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=(rz-rx)*l;
                                        ix=cx2;iz=cz2;y[id]+=(1-rz)*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=(1-rz)*l;
                                    }
                                }
                                else
                                {   if(cz2==0)// 33
                                    {   d=xx2-xx1;tmp=(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2);l=(float)sqrt((d*d+1)*(tmp+(z1-z2)*(z1-z2))/tmp);
                                        rx=(cx2-xx1)/d;rz=(cz2-zz1)/(zz2-zz1);
                                        if(rx>rz)
                                        {   ix=cx2;iz=cz2;y[id]+=(1-rx)*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=(1-rx)*l;
                                        }
                                        else
                                        {   ix=cx2;iz=cz2;y[id]+=(1-rz)*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=(1-rz)*l;
                                        }
                                    }
                                    if(cz2==nz)// 34
                                    {   d=xx2-xx1;tmp=(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2);l=(float)sqrt((d*d+1)*(tmp+(z1-z2)*(z1-z2))/tmp);
                                        rx=(cx2-xx1)/d;rz=(cz2-zz1)/(zz2-zz1);
                                        if(rx>rz)
                                        {
                                        }
                                        else
                                        {   ix=cx2;iz=cz1;y[id]+=(rz-rx)*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=(rz-rx)*l;
                                        }
                                    }
                                }
                            }
                        }

                        if(cx2==nx)
                        {   if(cz2==cz1)
                            {   if(cz1>=0&&cz1<=nz-1)// 41
                                {   d=xx2-xx1;tmp=(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2);l=(float)sqrt((d*d+1)*(tmp+(z1-z2)*(z1-z2))/tmp);
                                    rx=(cx2-xx1)/d;
                                    ix=cx1;iz=cz1;y[id]+=rx*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=rx*l;
                                }
                            }
                            else
                            {   if(cz2>0&&cz2<nz)// 42
                                {   d=xx2-xx1;tmp=(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2);l=(float)sqrt((d*d+1)*(tmp+(z1-z2)*(z1-z2))/tmp);
                                    rx=(cx2-xx1)/d;rz=(cz2-zz1)/(zz2-zz1);
                                    if(rx>rz)
                                    {   ix=cx1;iz=cz1;y[id]+=rz*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=rz*l;
                                        ix=cx1;iz=cz2;y[id]+=(rx-rz)*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=(rx-rz)*l;
                                    }
                                    else
                                    {   ix=cx1;iz=cz1;y[id]+=rx*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=rx*l;
                                    }
                                }
                                else
                                {   if(cz2==0)// 43
                                    {   d=xx2-xx1;tmp=(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2);l=(float)sqrt((d*d+1)*(tmp+(z1-z2)*(z1-z2))/tmp);
                                        rx=(cx2-xx1)/d;rz=(cz2-zz1)/(zz2-zz1);
                                        if(rx>rz)
                                        {   ix=cx1;iz=cz2;y[id]+=(rx-rz)*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=(rx-rz)*l;
                                        }
                                        else
                                        {
                                        }
                                    }
                                    if(cz2==nz)// 44
                                    {   d=xx2-xx1;tmp=(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2);l=(float)sqrt((d*d+1)*(tmp+(z1-z2)*(z1-z2))/tmp);
                                        rx=(cx2-xx1)/d;rz=(cz2-zz1)/(zz2-zz1);
                                        if(rx>rz)
                                        {   ix=cx1;iz=cz1;y[id]+=rz*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=rz*l;
                                        }
                                        else
                                        {   ix=cx1;iz=cz1;y[id]+=rx*l*x[iz*ny*nx+iy*nx+ix];sr[id]+=rx*l;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        y[id]*=scale;sr[id]*=scale;
    }
}

extern "C" void Ax_cone_mf_gpu_new(float *X,float *y,float *sr,float cos_phi,float sin_phi,float *y_det,float *z_det,
float SO,float OD,float scale,float dz,int nx,int ny,int nz,int na,int nb)
{   
	float *y_d,*x_d,*sr_d,*y_det_d,*z_det_d;
	int nd,N;

	N=nx*ny*nz;
	nd=na*nb;

	cudaMalloc((void**)&y_d,nd*sizeof(float));
	cudaMalloc((void**)&x_d,N*sizeof(float));cudaMemcpy(x_d,X,N*sizeof(float),cudaMemcpyHostToDevice);
	cudaMalloc((void**)&sr_d,nd*sizeof(float));
	cudaMalloc((void**)&y_det_d,na*sizeof(float));cudaMemcpy(y_det_d,y_det,na*sizeof(float),cudaMemcpyHostToDevice);
	cudaMalloc((void**)&z_det_d,nb*sizeof(float));cudaMemcpy(z_det_d,z_det,nb*sizeof(float),cudaMemcpyHostToDevice);
	
	//
	dim3 dimBlock(BLOCK_SIZE_x,BLOCK_SIZE_y);
	dim3 dimGrid_t((na+dimBlock.x-1)/dimBlock.x,(nb+dimBlock.y-1)/dimBlock.y);
	Ax_cone_mf_gpu_kernel_new<<<dimGrid_t, dimBlock>>>(x_d,y_d,sr_d,cos_phi,sin_phi,y_det_d,z_det_d,
		SO,OD,scale,dz,nx,ny,nz,na,nb);
	cudaThreadSynchronize();
	cudaMemcpy(y,y_d,na*nb*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(sr,sr_d,na*nb*sizeof(float),cudaMemcpyDeviceToHost);

	
	//
    cudaFree(y_d);cudaFree(x_d);cudaFree(y_det_d);cudaFree(z_det_d);cudaFree(sr_d);
}



