
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>
#include <conio.h>
#include <fstream> 
#include <string>
#include <limits.h>
#include <algorithm>
#include <ctime>
#include <stdio.h>
#include <GL\glut.h>
#include <GL\GL.h>


#define pi 3.14159265
#define pifact 0.01745329
#define lld long long int

using namespace std;
class complex
{
public:
	double real, img;


	void set(double r, double i)
	{
		real = r;
		img = i;

	}

	void mul(double a)
	{
		real *= a;
		img *= a;
	}


};


//host

complex P_theta[1205][2050], S_theta[1205][2050];
complex Q_theta[1205][2050];
double   F[512][512], P[512][512];
double T, Pmin, Pmax, Imax, Imin,D;
int N, N2, lln;

//device
complex *d_fft1, *d_fft2;
complex  *d_Qtheta;
double *d_F;




__host__ __device__ complex operator + (complex A, complex B)
{
	complex temp;
	temp.real = A.real + B.real;
	temp.img = A.img + B.img;
	return temp;
}

__host__ __device__ complex operator * (complex A, complex B)
{
	complex temp;
	temp.real = (A.real*B.real) - (A.img*B.img);
	temp.img = (A.real*B.img) + (A.img*B.real);
	return temp;

}

__host__ __device__ complex conj(complex A)
{
	complex temp;
	temp.real = A.real;
	temp.img = -1 * A.img;
	return temp;
}

__host__ __device__ complex expo(double a, double N)
{
	complex temp;
	//double co, si;
	/*if (a < 0.1)
	{
	temp.set(1, 0);
	return temp;
	}*/
	temp.real = cos(2 * pi*a / N);
	temp.img = sin(2 * pi*a / N);

	return temp;
}
__host__ __device__ double mod(double a)
{
	if (a < 0)
		return -1 * a;

	return a;
}


__host__ __device__ double modls(complex A)
{
	//	double r;
	//	r = (A.real*A.real) +(A.img*A.img);
	//	r = sqrt(r);
	return A.real;
}

/*complex compute_Qi(double t, lld k)
{
lld f;
double A, B, res1, res2;
complex temp;
if (mod(t)>double(N2))
{
temp.set(0, 0);
return temp;
}

f = floor(t);
A = Q_theta[k][f + N2].real;
B = Q_theta[k][f + 1 + N2].real;
res1 = A + (t - double(f))*(B - A);

A = Q_theta[k][f + N2].img;
B = Q_theta[k][f + 1 + N2].img;
res2 = A + (t - double(f))*(B - A);
temp.set(res1, res2);
return temp;

}

*/

__global__ void FFTKernel(complex *dfft1, complex *dfft2, int i, int N1)
{
	int j = (blockIdx.x)*(blockDim.x) + threadIdx.x;   //blockId   threadIdx.x;
	int f, p;


	if (j < N1)
	{
		f = j%i;
		p = (j*N1 / i) % N1;
		if (f<(i / 2))
		{
			dfft2[j] = dfft1[j] + (expo(p, N1)*dfft1[j + (i / 2)]);
		}
		else
		{
			dfft2[j] = dfft1[j - (i / 2)] + (expo(p, N1)*dfft1[j]);

		}

	}

}

__global__ void IMKernel(complex *Qtheta, double *d_F, double c, double s, int k, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	double x, y, t;
	double A, B, res1;
	int f;

	if (j < 512 && i<512)
	{
		x = double(i - 256)*4;
		y = double(j - 256)*4;
		t = x*c + y*s;
		if (mod(t)<double(N/2))
		{
			res1 = 0;
		
			//Linear Interpolation
			f = floor(t);
			A = Qtheta[k * 2050 + f + N / 2].real;
			B = Qtheta[k * 2050 + f + 1 + N / 2].real;
			res1 = A + (t - double(f))*(B - A);


			d_F[i * 512 + j] += res1;
		}


	}
	//c[i] = a[i] + b[i];
}

__global__ void minKernel(double *d_F, int k, int N)
{
	int j = (blockIdx.x)*(blockDim.x) + threadIdx.x;   //blockId   threadIdx.x;
	int f, p;
	lld x, y;
	x = (j / 512) - 256;
	y = (j % 512) - 256;
	x = x*x + y*y;
	double a, b;
	if (j < N / k )
	{


		a = d_F[j*k];
		b = d_F[j*k + (k / 2)];
		d_F[j*k] = min(a, b);

		/*	a = d_F[(j+1) * k -1];
		b = d_F[(j + 1) * k - 1 - (k / 2)];
		d_F[(j + 1) * k - 1] = max(a, b);*/
	}
}
__global__ void maxKernel(double *d_F, int k, int N)
{
	int j = (blockIdx.x)*(blockDim.x) + threadIdx.x;   //blockId   threadIdx.x;
	int f, p;
	lld x, y;
	double a, b;
	x = (j / 512)-256;
	y = (j % 512)-256;
	x = x*x + y*y;

	if (j < N / k )
	{


		a = d_F[j*k];
		b = d_F[j*k + (k / 2)];
		d_F[j*k] = max(a, b);

		/*		a = d_F[(j + 1) * k - 1];
		b = d_F[(j + 1) * k - 1 - (k / 2)];
		d_F[(j + 1) * k - 1] = max(a, b);*/
	}
}

__global__ void PixKernel(double *d_F, double Pmin, double temp)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (j < 512 && i < 512)
	{
		d_F[i * 512 + j] = (d_F[i * 512 + j] - Pmin) / temp;
	}
}



bool load_P_theta()
{
	ifstream fin;
	char name[30], c,str[100];
	lld i, j, a;
	double d;
	for (i = 1; i <= 1200; i++)
	{
		sprintf_s(name, "slhpdata\\mdata%d.dlj", i);
		//	cout << i << '\n';
		fin.open(name, ios::in);
		if (!fin)
		{
			cout << "Error opening file";
			return 0;
		}
		else
		{
			fin.getline(str, 100);
			for (j = 0; j < N-1; j++)
			{
				fin >> d >> a;
				//	cout << a << '\t' << d<<'\n';
				P_theta[i][j+1].set(d, 0);
			}
			P_theta[i][0].set(0, 0);
			fin.close();
		}
	}

	return 1;

}


void dealloc()
{

	cudaError_t cudaStatus;

	cudaFree(d_fft1);
	cudaFree(d_fft2);
	cudaFree(d_F);
	cudaFree(d_Qtheta);
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");

	}

}


void InitCuda()
{
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		dealloc();
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&d_fft1, N * sizeof(complex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "1cudaMalloc failed!");
		dealloc();
	}

	cudaStatus = cudaMalloc((void**)&d_fft2, N * sizeof(complex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "2cudaMalloc failed!");
		dealloc();
	}
	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&d_Qtheta, 1205 * 2050 * sizeof(complex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "1cudaMalloc failed!");
		dealloc();
	}

	cudaStatus = cudaMalloc((void**)&d_F, 512 * 512 * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "2cudaMalloc failed!");
		dealloc();
	}

}




void computeS_theta()
{
	lld i, j, k, p, iN, i2, f, a;
	complex sum, temp, fft1[2050], fft2[2050];
	cudaError_t cudaStatus;

	for (k = 1; k <= 1200; k++)
	{
		/*	for (i = 0; i < N2; i++)
		{
		fft2[i] = P_theta[k][i + N2];
		fft2[i + N2] = P_theta[k][i];
		}*/

		//	fft2[i] = P_theta[k][i];

		for (i = 0; i<N; i++)
		{
			j = i;
			f = 0;
			a = lln;
			while (a--)
			{
				p = (j & 1);
				f = 2 * f + p;
				j = j / 2;
			}
			//     cout<<f<<'\n';
			if (f < N2)
				fft1[i] = P_theta[k][f + N2];
			else
				fft1[i] = P_theta[k][f - N2];
		}


		cudaStatus = cudaMemcpy(d_fft1, fft1, N * sizeof(complex), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "computeS_theta3cudaMemcpy failed!");
			dealloc();
		}

		for (i = 2; i <= N; i = i * 2)
		{


			FFTKernel << <21, 98 >> >(d_fft1, d_fft2, i, N);

			// Check for any errors launching the kernel
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
				dealloc();
			}
			// cudaDeviceSynchronize waits for the kernel to finish, and returns
			// any errors encountered during the launch.
			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
				dealloc();
			}

			// Copy output vector from GPU buffer to device memory. fft1:=fft2          
			cudaStatus = cudaMemcpy(d_fft1, d_fft2, N * sizeof(complex), cudaMemcpyDeviceToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "5cudaMemcpy failed!");
				dealloc();
			}




		}
		// Copy output vector from GPU buffer to host memory.
		cudaStatus = cudaMemcpy(fft1, d_fft1, N * sizeof(complex), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "5cudaMemcpy failed!");
			dealloc();
		}

		/*	for (i = 0; i < N; i++)
		{
		S_theta[k][i] = fft1[i];
		memcpy(S_theta[k], fft1, N*sizeof(complex));
		}*/
		memcpy(S_theta[k], fft1, N*sizeof(complex));
	}

}


void compute_Qtheta()
{
	lld i, m, k, j, p, iN, i2, a, fc;
	complex sum, temp, fft1[2050], fft2[2050];
	double f;
	cudaError_t cudaStatus;
	f = 1 / (N*T);

	for (k = 1; k <= 1200; k++)
	{
		for (i = 0; i < N; i++)
		{
			a = mod(((i + N2) % N) - N2);


			fft2[i] = conj(S_theta[k][i]);

			fft2[i].mul(a*f);

		}

		for (i = 0; i<N; i++)
		{
			j = i;
			fc = 0;
			a = lln;
			while (a--)
			{
				p = (j & 1);
				fc = 2 * fc + p;
				j = j / 2;
			}

			fft1[i] = fft2[fc];
		}

		cudaStatus = cudaMemcpy(d_fft1, fft1, N * sizeof(complex), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "computeS_theta3cudaMemcpy failed!");
			dealloc();
		}

		for (i = 2; i <= N; i = i * 2)
		{


			FFTKernel << <21, 98 >> >(d_fft1, d_fft2, i, N);

			// Check for any errors launching the kernel
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
				dealloc();
			}
			// cudaDeviceSynchronize waits for the kernel to finish, and returns
			// any errors encountered during the launch.
			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
				dealloc();
			}

			// Copy output vector from GPU buffer to device memory. fft1:=fft2          
			cudaStatus = cudaMemcpy(d_fft1, d_fft2, N * sizeof(complex), cudaMemcpyDeviceToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "5cudaMemcpy failed!");
				dealloc();
			}




		}

		// Copy output vector from GPU buffer to host memory.
		cudaStatus = cudaMemcpy(fft1, d_fft1, N * sizeof(complex), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "5cudaMemcpy failed!");
			dealloc();
		}


		for (i = 0; i < N2; i++)
		{
			Q_theta[k][i + N2] = fft1[i];
			Q_theta[k][i] = fft1[i + N2];
		}


	}
}


void computeimage()
{
	lld i, j, k;
	double temp, t, r, x, y, c, s;
	complex ctemp, tr;
	cudaError_t cudaStatus;
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(512 / threadsPerBlock.x, 512 / threadsPerBlock.y);

	
	/*for (i = -256; i < 256; i++)
	{
	x = double(i) / (128 * T);
	for (j = -256; j < 256; j++)
	{
	y = double(j) / (128 * T);
	r = i*i + j*j;
	//	r = r / double(N2*N2);
	if (r < N2*N2)
	{
	ctemp.set(0, 0);
	for (k = 1; k <= 181; k++)
	{
	t = double(x)*cos(double(k - 1)*pifact) + double(y)*sin(double(k - 1)*pifact);

	tr = compute_Qi(t, k);
	if (k == 1)
	tr.mul(0.5);
	ctemp = ctemp + tr;
	}
	F[i + N2][j + N2] = modls(ctemp)*pifact;
	//	Imax = max(ctemp.img, Imax);
	//	Imin = min(ctemp.img, Imin);
	Pmin = min(F[i + N2][j + N2], Pmin);
	Pmax = max(F[i + N2][j + N2], Pmax);
	}
	else
	F[i + N2][j + N2] = 0;
	}
	}                                                                              d_F = 0 ??
	*/
	cudaStatus = cudaMemset(d_F, 0, 512 * 512 * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "3cudaMemset failed!");
		dealloc();
	}


	cudaStatus = cudaMemcpy(d_Qtheta, Q_theta, 1205 * 2050 * sizeof(complex), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "3cudaMemcpy failed!");
		dealloc();
	}


	for (k = 1; k <= 1200; k++)
	{
		c = cos(double(k - 1)*pifact*0.15);
		s = sin(double(k - 1)*pifact*0.15);

		IMKernel << <numBlocks, threadsPerBlock >> >(d_Qtheta, d_F, c, s, k, N);

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "1addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			dealloc();
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			dealloc();
		}



	}

	cudaStatus = cudaMemcpy(P, d_F, 512 * 512 * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "5cudaMemcpy failed!");
		dealloc();
	}

	//cout << "P" << P[3][123] << '\n';

	/*fstream fout;
	fout.open("Zdata.txt", ios::out);
	if (!fout)
	{
	cout << "error";

	}
	else
	{*/



	/*	for (i = -256; i < 256; i++)
	{
	for (j = -256; j < 256; j++)
	{
	r = i*i + j*j;
	//r = r / double(N2*N2);

	if (r <= N2*N2)
	{
	//k = (F[i + N2][j + N2] - Pmin) / temp;
	P[i + N2][j + N2] = (F[i + N2][j + N2] - Pmin) / temp;

	//					if (P[i + N2][j + N2] >0.9)
	//					fout << '(' << i << ',' << j << ")\n";
	}
	else
	P[i + N2][j + N2] = 0;



	}
	//	fout << '\n';
	}*/



	for (i = 2; i <= 262144; i *= 2)
	{
		if (265144 / (i * 256) == 0)
			minKernel << <1, 262144 / i >> >(d_F, i, 262144);
		else
			minKernel << <262144 / (i * 256), 256 >> >(d_F, i, 262144);


		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "2addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			dealloc();
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			dealloc();
		}

	}


	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(F, d_F, 512 * 512 * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "d_F5cudaMemcpy failed!\n");
		dealloc();
	}

	Pmin = F[0][0];

	for (i = 2; i <= 262144; i *= 2)
	{
		if (265144 / (i * 256) == 0)
			maxKernel << <1, 262144 / i >> >(d_F, i, 262144);
		else
			maxKernel << <262144 / (i * 256), 256 >> >(d_F, i, 262144);


		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "2addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			dealloc();
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			dealloc();
		}

	}


	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(F, d_F, 512 * 512 * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "d_F5cudaMemcpy failed!\n");
		dealloc();
	}

	Pmax = F[0][0];
	//Pmax = F[511][511];
	temp = Pmax - Pmin;



	cudaStatus = cudaMemcpy(d_F, P, 512 * 512 * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "3d_FcudaMemcpy failed!\n");
		dealloc();
	}


	PixKernel << <numBlocks, threadsPerBlock >> >(d_F, Pmin, temp);


	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "3addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		dealloc();
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		dealloc();
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(P, d_F, 512 * 512 * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "5cudaMemcpy failed!");
		dealloc();
	}




	//	fout.close();

	//	}


}

void Display(void)
{
	int i, j;

	glClear(GL_COLOR_BUFFER_BIT);
	cout << "on\n";

	for (i = -256; i < 256; i++)
	{
		for (j = -256; j < 256; j++)
		{


			glColor3f(P[i + 256][j + 256], P[i + 256][j + 256], P[i + 256][j + 256]);
			//	glColor3f(1.0, 0.2, 0.7);
			glBegin(GL_POINTS);

			glVertex2f(float(i) / (256), float(j) / (256));
			glEnd();



		}
	}
	glFlush();
}

int main(int argc, char **argv)
{
	//double D;
	int i, j;
	clock_t start = clock();
	//InitCuda();
	cout << "Enter value of N : ";
	cin >> N;
	cout << "Enter value of D : ";
	cin >> D;
	T = D / (N - 2);
	//N = 512;
	N2 = N / 2;
	lln = log2(N);
	//	N15 = 1.5*N2;
	//	compute_exp();
	if (load_P_theta() == 0)
	{
		cout << "error loading from files\n";
	}
	else
	{
		InitCuda();
		cout << P_theta[5][123].real << ' ' << P_theta[5][321].img << '\n';
		clock_t end1 = clock();
		cout << (end1 - start) / CLOCKS_PER_SEC << "go1\n";
		computeS_theta();
		cout << S_theta[5][123].real << ' ' << S_theta[5][321].img << '\n';
		clock_t end2 = clock();
		cout << (end2 - start) / CLOCKS_PER_SEC << "go2\n";


		compute_Qtheta();
		cout << Q_theta[5][123].real << ' ' << Q_theta[5][321].img << '\n';

		clock_t end3 = clock();
		cout << (end3 - start) / CLOCKS_PER_SEC << "go3\n";
		computeimage();
		cout << P[5][123] << ' ' << P[5][321] << '\n';
		dealloc();
		cout << Pmax << ' ' << Pmin;

		/*for (i = -N2; i < N2; i++)
		{
		for (j = -N2; j < N2; j++)
		{


		cout << P[i + N2][j + N2]<<' ';


		}
		cout << '\n';
		}*/
		//	cout << P[10][10] << ' ' <<  P[100][100];
		glutInit(&argc, argv);

		glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
		glutInitWindowSize(512, 512);
		clock_t end4 = clock();
		cout << (end4 - start) / CLOCKS_PER_SEC << "go6\n";

		glutCreateWindow("Output");
		clock_t end6 = clock();
		cout << (end6 - start) / CLOCKS_PER_SEC << "go5\n";


		//	glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
		glutDisplayFunc(Display);

		glutMainLoop();
	}
	//cout << expo_p[N2][N2].real << ' ' << expo_p[N2-4][N2 + 4].real << ' ' << expo_p[N2 + 1][N2 + 1].real;
	//cout << cos(pi);

	char c = _getch();

	return 0;
}
