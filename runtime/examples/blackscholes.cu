


#define N(x)       (erf((x)/sqrt(2.0f))/2+0.5f)

float BlackScholes(float * call, float * S, float * X, float * T, float * r, float * sigma, int len)
{

	int ii = threadIdx.x + blockDim.x * blockIdx.x;
	if (ii > len) {
		return ;
	}
	float d1=(log(S[ii]/X[ii])+(r[ii]+sigma[ii]*sigma[ii]/2)*T)/(sigma[ii]*sqrt(T[ii]));
	float d2=d1-sigma[ii]*sqrt(T[ii]);

	call[ii] = S[ii] *N(d1) - X[ii] * exp(-r[ii]*T[ii])*N(d2);
}

