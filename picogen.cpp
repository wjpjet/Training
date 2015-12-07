/*
 *	Copyright (c) 2013, Nenad Markus
 *	All rights reserved.
 *
 *	This is an implementation of the algorithm described in the following paper:
 *		N. Markus, M. Frljak, I. S. Pandzic, J. Ahlberg and R. Forchheimer,
 *		Object Detection with Pixel Intensity Comparisons Organized in Decision Trees,
 *		http://arxiv.org/abs/1305.4537
 *
 *	Redistribution and use of this program as source code or in binary form, with or without modifications, are permitted provided that the following conditions are met:
 *		1. Redistributions may not be sold, nor may they be used in a commercial product or activity without prior permission from the copyright holder (contact him at nenad.markus@fer.hr).
 *		2. Redistributions may not be used for military purposes.
 *		3. Any published work which utilizes this program shall include the reference to the paper available at http://arxiv.org/abs/1305.4537
 *		4. Redistributions must retain the above copyright notice and the reference to the algorithm on which the implementation is based on, this list of conditions and the following disclaimer.
 *
 *	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 *	IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <string>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <stdint.h>

#define MAX(a, b) ((a)>(b)?(a):(b))
#define ABS(x) ((x)>0?(x):(-(x)))

float tsr, tsc;
int tdepth;
int ntrees=0;

int32_t tcodes[4096][1024];
float luts[4096][1024];

float thresholds[4096];

bool load_cascade(const char* path, double threshold_shift)
{
	FILE* file = fopen(path, "rb");
	if (!file)
		return false;

	fread(&tsr, sizeof(float), 1, file);
	fread(&tsc, sizeof(float), 1, file);
	fread(&tdepth, sizeof(int), 1, file);
	fread(&ntrees, sizeof(int), 1, file);

	for (int i = 0; i < ntrees; ++i)
	{
		fread(&tcodes[i][0], sizeof(int32_t), (1<<tdepth)-1, file);
		fread(&luts[i][0], sizeof(float), 1<<tdepth, file);
		fread(&thresholds[i], sizeof(float), 1, file);
		if (threshold_shift)
			thresholds[i] -= fabs(thresholds[i]) * threshold_shift;
	}

	fclose(file);
	return true;
}

int save_cascade(const char* path)
{
	FILE* file = fopen(path, "wb");
	if (!file)
		return 0;

	fwrite(&tsr, sizeof(float), 1, file);
	fwrite(&tsc, sizeof(float), 1, file);
	fwrite(&tdepth, sizeof(int), 1, file);
	fwrite(&ntrees, sizeof(int), 1, file);

	for (int i = 0; i < ntrees; ++i)
	{
		fwrite(&tcodes[i][0], sizeof(int32_t), (1<<tdepth)-1, file);
		fwrite(&luts[i][0], sizeof(float), 1<<tdepth, file);
		fwrite(&thresholds[i], sizeof(float), 1, file);
	}

	fclose(file);
	return 1;
}

void print_func_name_cuda(const char *name)
{
	printf("__global__ void %s_cuda(float* response, unsigned char *result, "
		"int s, const unsigned char* pixels, "
		"int nrows, int ncols, int ldim, "
		"float dr, float dc, int res_cols)\n", name);
}

void print_func_name_c(const char *name)
{
	printf("int %s(float* o, int r, int c, int s, const uint8_t* pixels, "
		   "int nrows, int ncols, int ldim)\n", name);
}

void print_c_code(const char* name, double rotation, bool cuda)
{
	static int16_t rtcodes[4096][1024][4];

	// generate rotated binary tests
	int q = (1<<16);

	int qsin = (int)( q*sin(rotation) );
	int qcos = (int)( q*cos(rotation) );

	int maxr = 0;
	int maxc = 0;

	for (int i = 0; i < ntrees; ++i)
	{
		for (int j = 0; j < (1<<tdepth) - 1; ++j)
		{
			int8_t* p = (int8_t*)&tcodes[i][j];

			rtcodes[i][j][0] = (p[0]*qcos - p[1]*qsin)/q;
			rtcodes[i][j][1] = (p[0]*qsin + p[1]*qcos)/q;

			rtcodes[i][j][2] = (p[2]*qcos - p[3]*qsin)/q;
			rtcodes[i][j][3] = (p[2]*qsin + p[3]*qcos)/q;

			maxr = MAX(maxr, MAX(ABS(rtcodes[i][j][0]), ABS(rtcodes[i][j][2])));
			maxc = MAX(maxc, MAX(ABS(rtcodes[i][j][1]), ABS(rtcodes[i][j][3])));
		}
	}

	if (!cuda)
	{
		print_func_name_c(name);
		printf("{\n");
	}

	if (cuda)
		printf("	__device__ short tcodes[%d][%d][4] =\n", ntrees, 1<<tdepth);
	else
		printf("	static int16_t tcodes[%d][%d][4] =\n", ntrees, 1<<tdepth);

	printf("	{\n");
	for (int i = 0; i < ntrees; ++i)
	{
		printf("		{{0, 0, 0, 0}");
		for (int j = 0; j < (1<<tdepth) - 1; ++j)
			printf(", {%d, %d, %d, %d}", rtcodes[i][j][0], rtcodes[i][j][1], rtcodes[i][j][2], rtcodes[i][j][3]);
		printf("},\n");
	}
	printf("	};\n");

	printf("\n");
	if (cuda)
		printf("	__device__ float lut[%d][%d] =\n", ntrees, 1<<tdepth);
	else
		printf("	static float lut[%d][%d] =\n", ntrees, 1<<tdepth);
	printf("	{\n");
	for (int i = 0; i < ntrees; ++i)
	{
		printf("		{");
		for (int j = 0; j < (1<<tdepth) - 1; ++j)
			printf("%ff, ", luts[i][j]);
		printf("%ff},\n", luts[i][(1<<tdepth)-1]);
	}
	printf("	};\n");

	printf("\n");
	if (cuda)
		printf("	__device__ float thresholds[%d] =\n", ntrees);
	else
		printf("	static float thresholds[%d] =\n", ntrees);
	printf("	{\n\t\t");
	for (int i = 0; i < ntrees - 1; ++i)
		printf("%ff, ", thresholds[i]);
	printf("%ff\n", thresholds[ntrees-1]);
	printf("	};\n");

	if (cuda)
	{
		print_func_name_cuda(name);
		printf("{\n");
		printf("	int i, idx;\n");
		printf("\n");
	}

	printf("\n");
	printf("	int sr = (int)(%ff*s);\n", tsr);
	printf("	int sc = (int)(%ff*s);\n", tsc);

	printf("\n");
	if (cuda)
	{
		printf("	int px = blockIdx.x * blockDim.x + threadIdx.x;\n");
		printf("	int py = blockIdx.y * blockDim.y + threadIdx.y;\n");
		printf("	int r = int(s / 2 + 1 + py * dr) * 256;\n");
		printf("	int c = int(s / 2 + 1 + px * dc) * 256;\n");
		printf("	int res_stride = py * res_cols + px;\n");
	}
	else
	{
		printf("	r *= 256;\n");
		printf("	c *= 256;\n");
	}

	// check image boundaries
	printf("\n");
	printf("	if( (r+%d*sr)/256>=nrows || (r-%d*sr)/256<0 || "
		   "(c+%d*sc)/256>=ncols || (c-%d*sc)/256<0 )\n",
		   maxr, maxr, maxc, maxc);
	if (cuda)
	{
		printf("	{\n");
		printf("		result[res_stride] = 0;\n");
		printf("		return;\n");
		printf("	}\n");
	}
	else
		printf("		return -1;\n");

	printf("\n");
	if (cuda)
		printf("	float *o = response + res_stride;\n\n");

	printf("	*o = 0.0f;\n\n");
	// printf("	pixels = &pixels[r*ldim+c];\n");
	printf("	for (int i = 0; i < %d; ++i)\n", ntrees);
	printf("	{\n");
	printf("		int idx = 1;\n");
	for (int i = 0; i < tdepth; ++i)
	{
		printf("		idx = 2*idx + (pixels[(r+tcodes[i][idx][0]*sr)/256*ldim + (c+tcodes[i][idx][1]*sc)/256]<=pixels[(r+tcodes[i][idx][2]*sr)/256*ldim + (c+tcodes[i][idx][3]*sc)/256]);\n");
		///printf("		idx = 2*idx + (pixels[tcodes[i][idx][0]*sr/256*ldim + tcodes[i][idx][1]*sc/256]<=pixels[tcodes[i][idx][2]*sr/256*ldim + tcodes[i][idx][3]*sc/256]);\n");
	}
	printf("\n		*o += lut[i][idx-%d];\n\n", 1<<tdepth);
	printf("		if (*o <= thresholds[i])\n");
	if (cuda)
	{
		printf("		{\n");
		printf("			result[res_stride] = 0;\n");
		printf("			return;\n");
		printf("		}\n");
	}
	else
		printf("			return -1;\n");
	printf("	}\n");

	printf("\n	*o -= thresholds[%d];\n", ntrees - 1);
	printf("\n");
	if (cuda)
		printf("	result[res_stride] = 1;\n");
	else
		printf("	return 1;\n");

	printf("}\n");
}

void usage(const char *prog_name)
{
	printf("Usage:\n");
	printf("%s [-r rotation_angle] [-s threshold_shift] "
		   "[-sr scale_row] [-sc scale_col] [--cuda] "
		   "<cascade>  <detection function name>\n", prog_name);
}

int main(int argc, char* argv[])
{
	std::string cascade_name;
	std::string func_name;
	double th_shift = 0;
	double rotation = 0;
	double scale_row = 1.0;
	double scale_col = 1.0;
	bool use_cuda = false;
	int opt_count = 1;
	while (opt_count < argc)
	{
		if (std::string(argv[opt_count]) == "-h")
		{
			usage(argv[0]);
			return 0;
		}
		else if (std::string(argv[opt_count]) == "-r")
		{
			++opt_count;
			if (opt_count < argc)
				rotation = atof(argv[opt_count]);
		}
		else if (std::string(argv[opt_count]) == "-s")
		{
			++opt_count;
			if (opt_count < argc)
				th_shift = atof(argv[opt_count]);
		}
		else if (std::string(argv[opt_count]) == "--cuda")
		{
			use_cuda = true;
		}
		else if (std::string(argv[opt_count]) == "-sr")
		{
			++opt_count;
			if (opt_count < argc)
				scale_row = atof(argv[opt_count]);
		}
		else if (std::string(argv[opt_count]) == "-sc")
		{
			++opt_count;
			if (opt_count < argc)
				scale_col = atof(argv[opt_count]);
		}
		else if (argv[opt_count][0] == '-')
		{
			printf("unknown parameter %s\n", argv[opt_count]);
		}
		else if (cascade_name.empty())
		{
			cascade_name = argv[opt_count];
		}
		else if (func_name.empty())
		{
			func_name = argv[opt_count];
		}
		else
		{
			printf("unknown parameter %s\n", argv[opt_count]);
		}
		++opt_count;
	}

	if (cascade_name.empty() || func_name.empty())
	{
		usage(argv[0]);
		return -1;
	}

	if (!load_cascade(cascade_name.c_str(), th_shift))
	{
		printf("ERROR: can't load cascade %s\n", cascade_name.c_str());
		return -2;
	}

	tsr *= scale_row;
	tsc *= scale_col;

	print_c_code(func_name.c_str(), rotation, use_cuda);
	return 0;
}
