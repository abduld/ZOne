

#include "curand_kernel.h"

__global __ void randChoice(float *out, int outLength, float *values, int *inx,
                            float *q, int aliasLength, int *seeds) {
  curandState rngState;
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  curand_init(seeds[index], index, 0, &rngState);
  if (index < outLength) {
    float u = curand_uniform(&rngState);
    int j = curand(&rngState) % aliasLength;
    out[index] = u < q[j] ? values[j] : values[inx[j]];
  }
}

#if 0
initializeAlias[dist_, n_] :=
 Block[{$MinPrecision = $MachinePrecision, $MaxPrecision = \
$MachinePrecision},
  Block[{p = Table[PDF[dist, k], {k, n}], q, long = {}, short = {}, 
    inx, k = 0, j = 0},
   q = (p n)/Total[p];
   Do[If[q[[ii]] < 1, AppendTo[short, ii], AppendTo[long, ii]], {ii, 
     n}];
   inx = Join[long, short];
   While[short != {} && long != {},
    k = Last[long]; j = Last[short]; q[[k]] -= 1 - q[[j]]; 
    inx[[j]] = k;
    If[q[[k]] < 1, short[[-1]] = k; long = Most[long], 
     short = Most[short]];
    ];
   {inx - 1, q}
   ]
  ]
#endif