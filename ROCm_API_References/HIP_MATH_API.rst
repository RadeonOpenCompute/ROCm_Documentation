###########
HIP MATH APIs Documentation 
###########

HIP supports most of the device functions supported by CUDA. Way to find the unsupported one is to search for the function and check its description



For Developers 
##############

If you add or fixed a device function, make sure to add a signature of the function and definition later.
For example, if you want to add `__device__ float __dotf(float4, float4)`, which does a dot product on 4 float vector components 

The way to add to the header is, 

.. code-block:: 
__device__ static float __dotf(float4, float4); 
/*Way down in the file....*/
__device__ static inline float __dotf(float4 x, float4 y)

*implementation*



This helps python script to add the device function newly declared into markdown documentation (as it looks at functions with `;` at the end and `__device__` at the beginning)

The next step would be to add Description to  `deviceFuncDesc` dictionary in python script.
From the above example, it can be writtern as,
`deviceFuncDesc['__dotf'] = 'This functions takes 2 4 component float vector and outputs dot product across them'`

**acosf
.. code-block:: 
__device__ float acosf(float x);


*Description:  This function returns floating point of arc cosine from a floating point input


**acoshf
.. code-block:: 
__device__ float acoshf(float x);


*Description:   Supported


**asinf
.. code-block::  
__device__ float asinf(float x);


*Description:  Supported


**asinhf
.. code-block::  
__device__ float asinhf(float x);


*Description:* Supported


**atan2f
.. code-block::  
__device__ float atan2f(float y, float x);


**Description:  Supported


**atanf
.. code-block::  
__device__ float atanf(float x);


*Description:  Supported


**atanhf
.. code-block::  
__device__ float atanhf(float x);


*Description:  Supported


**cbrtf
.. code-block::  
__device__ float cbrtf(float x);


*Description:* Supported


**ceilf
.. code-block::  
__device__ float ceilf(float x);


*Description:* Supported


**copysignf
.. code-block::  
__device__ float copysignf(float x, float y);


*Description:* Supported


**cosf
.. code-block::  
__device__ float cosf(float x);


*Description:* Supported


**coshf
.. code-block::  
__device__ float coshf(float x);


*Description:* Supported


**cospif
.. code-block::  
__device__ __host__ float cospif(float x);


*Description:* Supported


**cyl_bessel_i0f
.. code-block::  
//__device__ float cyl_bessel_i0f(float x);


*Description:* **NOT Supported**


**cyl_bessel_i1f
.. code-block::  
//__device__ float cyl_bessel_i1f(float x);


*Description:* **NOT Supported**


**erfcf
.. code-block::  
__device__ float erfcf(float x);


*Description:* Supported


**erfcinvf
.. code-block::  
__device__  float erfcinvf(float y);


*Description:* Supported


**erfcxf
.. code-block::  
__device__ float erfcxf(float x);


*Description:* Supported


**erff
.. code-block::  
__device__ float erff(float x);


*Description:* Supported


**erfinvf
.. code-block::  
__device__ float erfinvf(float y);


*Description:* Supported


**exp10f
.. code-block::  
__device__ float exp10f(float x);


*Description:* Supported


**exp2f
.. code-block::  
__device__ float exp2f(float x);


*Description:* Supported


**expf
.. code-block::  
__device__ float expf(float x);


*Description:* Supported


**expm1f
.. code-block::  
__device__ float expm1f(float x);


*Description:* Supported


**fabsf
.. code-block::  
__device__ float fabsf(float x);


*Description:* Supported


**fdimf
.. code-block::  
__device__ float fdimf(float x, float y);


*Description:* Supported


**fdividef
.. code-block::  
__device__ __host__ float fdividef(float x, float y);


*Description:* Supported


**floorf
.. code-block::  
__device__ float floorf(float x);


*Description:* Supported


**fmaf
.. code-block::  
__device__ float fmaf(float x, float y, float z);


*Description:* Supported


**fmaxf
.. code-block::  
__device__ float fmaxf(float x, float y);


*Description:* Supported


**fminf
.. code-block::  
__device__ float fminf(float x, float y);


*Description:* Supported


**fmodf
.. code-block::  
__device__ float fmodf(float x, float y);


*Description:* Supported


**frexpf
.. code-block::  
//__device__ float frexpf(float x, int*nptr);


*Description:* **NOT Supported**


**hypotf
.. code-block::  
__device__ float hypotf(float x, float y);


*Description:* Supported


**ilogbf
.. code-block::  
__device__ float ilogbf(float x);


*Description:* Supported


**isfinite
.. code-block::  
__device__ __host__ int isfinite(float a);


*Description:* Supported


**isinf
.. code-block::  
__device__ unsigned isinf(float a);


*Description:* Supported


**isnan
.. code-block::  
__device__ unsigned isnan(float a);


*Description:* Supported


**j0f
.. code-block::  
__device__ float j0f(float x);


*Description:* Supported


**j1f
.. code-block::  
__device__ float j1f(float x);


*Description:* Supported


**jnf
.. code-block::  
__device__ float jnf(int n, float x);


*Description:* Supported


**ldexpf
.. code-block::  
__device__ float ldexpf(float x, int exp);


*Description:* Supported


**lgammaf
.. code-block::  
//__device__ float lgammaf(float x);


*Description:* **NOT Supported**


**llrintf
.. code-block::  
__device__ long long int llrintf(float x);


*Description:* Supported


**llroundf
.. code-block::  
__device__ long long int llroundf(float x);


*Description:* Supported


**log10f
.. code-block::  
__device__ float log10f(float x);


*Description:* Supported


**log1pf
.. code-block::  
__device__ float log1pf(float x);


*Description:* Supported


**logbf
.. code-block::  
__device__ float logbf(float x);


*Description:* Supported


**lrintf
.. code-block::  
__device__ long int lrintf(float x);


*Description:* Supported


**lroundf
.. code-block::  
__device__ long int lroundf(float x);


*Description:* Supported


**modff
.. code-block::  
//__device__ float modff(float x, float *iptr);


*Description:* **NOT Supported**


**nanf
.. code-block::  
__device__ float nanf(const char*tagp);


*Description:* Supported


**nearbyintf
.. code-block::  
__device__ float nearbyintf(float x);


*Description:* Supported


**nextafterf
.. code-block::  
//__device__ float nextafterf(float x, float y);


*Description:* **NOT Supported**


**norm3df
.. code-block::  
__device__ float norm3df(float a, float b, float c);


*Description:* Supported


**norm4df
.. code-block::  
__device__ float norm4df(float a, float b, float c, float d);


*Description:* Supported


**normcdff
.. code-block::  
__device__ float normcdff(float y);


*Description:* Supported


**normcdfinvf
.. code-block::  
__device__ float normcdfinvf(float y);


*Description:* Supported


**normf
.. code-block::  
__device__ float normf(int dim, const float *a);


*Description:* Supported


**powf
.. code-block::  
__device__ float powf(float x, float y);


*Description:* Supported


**rcbrtf
.. code-block::  
__device__ float rcbrtf(float x);


*Description:* Supported


**remainderf
.. code-block::  
__device__ float remainderf(float x, float y);


*Description:* Supported


**remquof
.. code-block::  
__device__ float remquof(float x, float y, int *quo);


*Description:* Supported


**rhypotf
.. code-block::  
__device__ float rhypotf(float x, float y);


*Description:* Supported


**rintf
.. code-block::  
__device__ float rintf(float x);


*Description:* Supported


**rnorm3df
.. code-block::  
__device__ float rnorm3df(float a, float b, float c);


*Description:* Supported


**rnorm4df
.. code-block::  
__device__ float rnorm4df(float a, float b, float c, float d);


*Description:* Supported


**rnormf
.. code-block::  
__device__ float rnormf(int dim, const float*a);


*Description:* Supported


**roundf
.. code-block::  
__device__ float roundf(float x);


*Description:* Supported


**rsqrtf
.. code-block::  
__device__ __host__ float rsqrtf(float x);


*Description:* Supported


**scalblnf
.. code-block::  
__device__ float scalblnf(float x, long int n);


*Description:* Supported


**scalbnf
.. code-block::  
__device__ float scalbnf(float x, int n);


*Description:* Supported


**signbit
.. code-block::  
__device__ __host__ unsigned signbit(float a);


*Description:* Supported


**sincosf
.. code-block::  
__device__ void sincosf(float x, float *sptr, float *cptr);


*Description:* Supported


**sincospif
.. code-block::  
__device__ void sincospif(float x, float *sptr, float *cptr);


*Description:* Supported


**sinf
.. code-block::  
__device__ float sinf(float x);


*Description:* Supported


**sinhf
.. code-block::  
__device__ float sinhf(float x);


*Description:* Supported


**sinpif
.. code-block::  
__device__ __host__ float sinpif(float x);


*Description:* Supported


**sqrtf
.. code-block::  
__device__ float sqrtf(float x);


*Description:* Supported


**tanf
.. code-block::  
__device__ float tanf(float x);


*Description:* Supported


**tanhf
.. code-block::  
__device__ float tanhf(float x);


*Description:* Supported


**tgammaf
.. code-block::  
__device__ float tgammaf(float x);


*Description:* Supported


**truncf
.. code-block::  
__device__ float truncf(float x);


*Description:* Supported


**y0f
.. code-block::  
__device__ float y0f(float x);


*Description:* Supported


**y1f
.. code-block::  
__device__ float y1f(float x);


*Description:* Supported


**ynf
.. code-block::  
__device__ float ynf(int n, float x);


*Description:* Supported


**acos
.. code-block::  
__device__ double acos(double x);


*Description:* Supported


**acosh
.. code-block::  
__device__ double acosh(double x);


*Description:* Supported


**asin
.. code-block::  
__device__ double asin(double x);


*Description:* Supported


**asinh
.. code-block::  
__device__ double asinh(double x);


*Description:* Supported


**atan
.. code-block::  
__device__ double atan(double x);


*Description:* Supported


**atan2
.. code-block::  
__device__ double atan2(double y, double x);


*Description:* Supported


**atanh
.. code-block::  
__device__ double atanh(double x);


*Description:* Supported


**cbrt
.. code-block::  
__device__ double cbrt(double x);


*Description:* Supported


**ceil
.. code-block::  
__device__ double ceil(double x);


*Description:* Supported


**copysign
.. code-block::  
__device__ double copysign(double x, double y);


*Description:* Supported


**cos
.. code-block::  
__device__ double cos(double x);


*Description:* Supported


**cosh
.. code-block::  
__device__ double cosh(double x);


*Description:* Supported


**cospi
.. code-block::  
__device__ __host__ double cospi(double x);


*Description:* Supported


**cyl_bessel_i0
.. code-block::  
//__device__ double cyl_bessel_i0(double x);


*Description:* **NOT Supported**


**cyl_bessel_i1
.. code-block::  
//__device__ double cyl_bessel_i1(double x);


*Description:* **NOT Supported**


**erf
.. code-block::  
__device__ double erf(double x);


*Description:* Supported


**erfc
.. code-block::  
__device__ double erfc(double x);


*Description:* Supported


**erfcinv
.. code-block::  
__device__ double erfcinv(double y);


*Description:* Supported


**erfcx
.. code-block::  
__device__ double erfcx(double x);


*Description:* Supported


**erfinv
.. code-block::  
__device__ double erfinv(double x);


*Description:* Supported


**exp
.. code-block::  
__device__ double exp(double x);


*Description:* Supported


**exp10
.. code-block::  
__device__ double exp10(double x);


*Description:* Supported


**exp2
.. code-block::  
__device__ double exp2(double x);


*Description:* Supported


**expm1
.. code-block::  
__device__ double expm1(double x);


*Description:* Supported


**fabs
.. code-block::  
__device__ double fabs(double x);


*Description:* Supported


**fdim
.. code-block::  
__device__ double fdim(double x, double y);


*Description:* Supported


**floor
.. code-block::  
__device__ double floor(double x);


*Description:* Supported


**fma
.. code-block::  
__device__ double fma(double x, double y, double z);


*Description:* Supported


**fmax
.. code-block::  
__device__ double fmax(double x, double y);


*Description:* Supported


**fmin
.. code-block::  
__device__ double fmin(double x, double y);


*Description:* Supported


**fmod
.. code-block::  
__device__ double fmod(double x, double y);


*Description:* Supported


**frexp
.. code-block::  
//__device__ double frexp(double x, int *nptr);


*Description:* **NOT Supported**


**hypot
.. code-block::  
__device__ double hypot(double x, double y);


*Description:* Supported


**ilogb
.. code-block::  
__device__ double ilogb(double x);


*Description:* Supported


**isfinite
.. code-block::  
__device__ __host__ unsigned isfinite(double x);


*Description:* Supported


**isinf
.. code-block::  
__device__ unsigned isinf(double x);


*Description:* Supported


**isnan
.. code-block::  
__device__ unsigned isnan(double x);


*Description:* Supported


**j0
.. code-block::  
__device__ double j0(double x);


*Description:* Supported


**j1
.. code-block::  
__device__ double j1(double x);


*Description:* Supported


**jn
.. code-block::  
__device__ double jn(int n, double x);


*Description:* Supported


**ldexp
.. code-block::  
__device__ double ldexp(double x, int exp);


*Description:* Supported


**lgamma
.. code-block::  
__device__ double lgamma(double x);


*Description:* Supported


**llrint
.. code-block::  
__device__ long long llrint(double x);


*Description:* Supported


**llround
.. code-block::  
__device__ long long llround(double x);


*Description:* Supported


**log
.. code-block::  
__device__ double log(double x);


*Description:* Supported


**log10
.. code-block::  
__device__ double log10(double x);


*Description:* Supported


**log1p
.. code-block::  
__device__ double log1p(double x);


*Description:* Supported


**log2
.. code-block::  
__device__ double log2(double x);


*Description:* Supported


**logb
.. code-block::  
__device__ double logb(double x);


*Description:* Supported


**lrint
.. code-block::  
__device__ long int lrint(double x);


*Description:* Supported


**lround
.. code-block::  
__device__ long int lround(double x);


*Description:* Supported


**modf
.. code-block::  
//__device__ double modf(double x, double *iptr);


*Description:* **NOT Supported**


**nan
.. code-block::  
__device__ double nan(const char*tagp);


*Description:* Supported


**nearbyint
.. code-block::  
__device__ double nearbyint(double x);


*Description:* Supported


**nextafter
.. code-block::  
__device__ double nextafter(double x, double y);


*Description:* Supported


**norm
.. code-block::  
__device__ double norm(int dim, const double*t);


*Description:* Supported


**norm3d
.. code-block::  
__device__ double norm3d(double a, double b, double c);


*Description:* Supported


**norm4d
.. code-block::  
__device__ double norm4d(double a, double b, double c, double d);


*Description:* Supported


**normcdf
.. code-block::  
__device__ double normcdf(double y);


*Description:* Supported


**normcdfinv
.. code-block::  
__device__ double normcdfinv(double y);


*Description:* Supported


**pow
.. code-block::  
__device__ double pow(double x, double y);


*Description:* Supported


**rcbrt
.. code-block::  
__device__ double rcbrt(double x);


*Description:* Supported


**remainder
.. code-block::  
__device__ double remainder(double x, double y);


*Description:* Supported


**remquo
.. code-block::  
//__device__ double remquo(double x, double y, int *quo);


*Description:* **NOT Supported**


**rhypot
.. code-block::  
__device__ double rhypot(double x, double y);


*Description:* Supported


**rint
.. code-block::  
__device__ double rint(double x);


*Description:* Supported


**rnorm
.. code-block::  
__device__ double rnorm(int dim, const double*t);


*Description:* Supported


**rnorm3d
.. code-block::  
__device__ double rnorm3d(double a, double b, double c);


*Description:* Supported


**rnorm4d
.. code-block::  
__device__ double rnorm4d(double a, double b, double c, double d);


*Description:* Supported


**round
.. code-block::  
__device__ double round(double x);


*Description:* Supported


**rsqrt
.. code-block::  
__device__ __host__ double rsqrt(double x);


*Description:* Supported


**scalbln
.. code-block::  
__device__ double scalbln(double x, long int n);


*Description:* Supported


**scalbn
.. code-block::  
__device__ double scalbn(double x, int n);


*Description:* Supported


**signbit
.. code-block::  
__device__ __host__ unsigned signbit(double a);


*Description:* Supported


**sin
.. code-block::  
__device__ double sin(double a);


*Description:* Supported


**sincos
.. code-block::  
__device__ void sincos(double x, double *sptr, double *cptr);


*Description:* Supported


**sincospi
.. code-block::  
__device__ void sincospi(double x, double *sptr, double *cptr);


*Description:* Supported


**sinh
.. code-block::  
__device__ double sinh(double x);


*Description:* Supported


**sinpi
.. code-block::  
__device__ __host__ double sinpi(double x);


*Description:* Supported


**sqrt
.. code-block::  
__device__ double sqrt(double x);


*Description:* Supported


**tan
.. code-block::  
__device__ double tan(double x);


*Description:* Supported


**tanh
.. code-block::  
__device__ double tanh(double x);


*Description:* Supported


**tgamma
.. code-block::  
__device__ double tgamma(double x);


*Description:* Supported


**trunc
.. code-block::  
__device__ double trunc(double x);


*Description:* Supported


**y0
.. code-block::  
__device__ double y0(double x);


*Description:* Supported


**y1
.. code-block::  
__device__ double y1(double y);


*Description:* Supported


**yn
.. code-block::  
__device__ double yn(int n, double x);


*Description:* Supported


**__cosf
.. code-block::  
__device__  float __cosf(float x);


*Description:* Supported


**__exp10f
.. code-block::  
__device__  float __exp10f(float x);


*Description:* Supported


**__expf
.. code-block::  
__device__  float __expf(float x);


*Description:* Supported


**__fadd_rd
.. code-block::  
__device__ static  float __fadd_rd(float x, float y);


*Description:* Supported


**__fadd_rn
.. code-block::  
__device__ static  float __fadd_rn(float x, float y);


*Description:* Supported


**__fadd_ru
.. code-block::  
__device__ static  float __fadd_ru(float x, float y);


*Description:* Supported


**__fadd_rz
.. code-block::  
__device__ static  float __fadd_rz(float x, float y);


*Description:* Supported


**__fdiv_rd
.. code-block::  
__device__ static  float __fdiv_rd(float x, float y);


*Description:* Supported


**__fdiv_rn
.. code-block::  
__device__ static  float __fdiv_rn(float x, float y);


*Description:* Supported


**__fdiv_ru
.. code-block::  
__device__ static  float __fdiv_ru(float x, float y);


*Description:* Supported


**__fdiv_rz
.. code-block::  
__device__ static  float __fdiv_rz(float x, float y);


*Description:* Supported


**__fdividef
.. code-block::  
__device__ static  float __fdividef(float x, float y);


*Description:* Supported


**__fmaf_rd
.. code-block::  
__device__  float __fmaf_rd(float x, float y, float z);


*Description:* Supported


**__fmaf_rn
.. code-block::  
__device__  float __fmaf_rn(float x, float y, float z);


*Description:* Supported


**__fmaf_ru
.. code-block::  
__device__  float __fmaf_ru(float x, float y, float z);


*Description:* Supported


**__fmaf_rz
.. code-block::  
__device__  float __fmaf_rz(float x, float y, float z);


*Description:* Supported


**__fmul_rd
.. code-block::  
__device__ static  float __fmul_rd(float x, float y);


*Description:* Supported


**__fmul_rn
.. code-block::  
__device__ static  float __fmul_rn(float x, float y);


*Description:* Supported


**__fmul_ru
.. code-block::  
__device__ static  float __fmul_ru(float x, float y);


*Description:* Supported


**__fmul_rz
.. code-block::  
__device__ static  float __fmul_rz(float x, float y);


*Description:* Supported


**__frcp_rd
.. code-block::  
__device__  float __frcp_rd(float x);


*Description:* Supported


**__frcp_rn
.. code-block::  
__device__  float __frcp_rn(float x);


*Description:* Supported


**__frcp_ru
.. code-block::  
__device__  float __frcp_ru(float x);


*Description:* Supported


**__frcp_rz
.. code-block::  
__device__  float __frcp_rz(float x);


*Description:* Supported


**__frsqrt_rn
.. code-block::  
__device__  float __frsqrt_rn(float x);


*Description:* Supported


**__fsqrt_rd
.. code-block::  
__device__  float __fsqrt_rd(float x);


*Description:* Supported


**__fsqrt_rn
.. code-block::  
__device__  float __fsqrt_rn(float x);


*Description:* Supported


**__fsqrt_ru
.. code-block::  
__device__  float __fsqrt_ru(float x);


*Description:* Supported


**__fsqrt_rz
.. code-block::  
__device__  float __fsqrt_rz(float x);


*Description:* Supported


**__fsub_rd
.. code-block::  
__device__ static  float __fsub_rd(float x, float y);


*Description:* Supported


**__fsub_rn
.. code-block::  
__device__ static  float __fsub_rn(float x, float y);


*Description:* Supported


**__fsub_ru
.. code-block::  
__device__ static  float __fsub_ru(float x, float y);


*Description:* Supported


**__log10f
.. code-block::  
__device__  float __log10f(float x);


*Description:* Supported


**__log2f
.. code-block::  
__device__  float __log2f(float x);


*Description:* Supported


**__logf
.. code-block::  
__device__  float __logf(float x);


*Description:* Supported


**__powf
.. code-block::  
__device__  float __powf(float base, float exponent);


*Description:* Supported


**__saturatef
.. code-block::  
__device__ static  float __saturatef(float x);


*Description:* Supported


**__sincosf
.. code-block::  
__device__  void __sincosf(float x, float *s, float *c);


*Description:* Supported


**__sinf
.. code-block::  
__device__  float __sinf(float x);


*Description:* Supported


**__tanf
.. code-block::  
__device__  float __tanf(float x);


*Description:* Supported


**__dadd_rd
.. code-block::  
__device__ static  double __dadd_rd(double x, double y);


*Description:* Supported


**__dadd_rn
.. code-block::  
__device__ static  double __dadd_rn(double x, double y);


*Description:* Supported


**__dadd_ru
.. code-block::  
__device__ static  double __dadd_ru(double x, double y);


*Description:* Supported


**__dadd_rz
.. code-block::  
__device__ static  double __dadd_rz(double x, double y);


*Description:* Supported


**__ddiv_rd
.. code-block::  
__device__ static  double __ddiv_rd(double x, double y);


*Description:* Supported


**__ddiv_rn
.. code-block::  
__device__ static  double __ddiv_rn(double x, double y);


*Description:* Supported


**__ddiv_ru
.. code-block::  
__device__ static  double __ddiv_ru(double x, double y);


*Description:* Supported


**__ddiv_rz
.. code-block::  
__device__ static  double __ddiv_rz(double x, double y);


*Description:* Supported


**__dmul_rd
.. code-block::  
__device__ static  double __dmul_rd(double x, double y);


*Description:* Supported


**__dmul_rn
.. code-block::  
__device__ static  double __dmul_rn(double x, double y);


*Description:* Supported


**__dmul_ru
.. code-block::  
__device__ static  double __dmul_ru(double x, double y);


*Description:* Supported


**__dmul_rz
.. code-block::  
__device__ static  double __dmul_rz(double x, double y);


*Description:* Supported


**__drcp_rd
.. code-block::  
__device__  double __drcp_rd(double x);


*Description:* Supported


**__drcp_rn
.. code-block::  
__device__  double __drcp_rn(double x);


*Description:* Supported


**__drcp_ru
.. code-block::  
__device__  double __drcp_ru(double x);


*Description:* Supported


**__drcp_rz
.. code-block::  
__device__  double __drcp_rz(double x);


*Description:* Supported


**__dsqrt_rd
.. code-block::  
__device__  double __dsqrt_rd(double x);


*Description:* Supported


**__dsqrt_rn
.. code-block::  
__device__  double __dsqrt_rn(double x);


*Description:* Supported


**__dsqrt_ru
.. code-block::  
__device__  double __dsqrt_ru(double x);


*Description:* Supported


**__dsqrt_rz
.. code-block::  
__device__  double __dsqrt_rz(double x);


*Description:* Supported


**__dsub_rd
.. code-block::  
__device__ static  double __dsub_rd(double x, double y);


*Description:* Supported


**__dsub_rn
.. code-block::  
__device__ static  double __dsub_rn(double x, double y);


*Description:* Supported


**__dsub_ru
.. code-block::  
__device__ static  double __dsub_ru(double x, double y);


*Description:* Supported


**__dsub_rz
.. code-block::  
__device__ static  double __dsub_rz(double x, double y);


*Description:* Supported


**__fma_rd
.. code-block::  
__device__  double __fma_rd(double x, double y, double z);


*Description:* Supported


**__fma_rn
.. code-block::  
__device__  double __fma_rn(double x, double y, double z);


*Description:* Supported


**__fma_ru
.. code-block::  
__device__  double __fma_ru(double x, double y, double z);


*Description:* Supported


**__fma_rz
.. code-block::  
__device__  double __fma_rz(double x, double y, double z);


*Description:* Supported


**__brev
.. code-block::  
__device__ unsigned int __brev( unsigned int x);


*Description:* Supported


**__brevll
.. code-block::  
__device__ unsigned long long int __brevll( unsigned long long int x);


*Description:* Supported


**__byte_perm
.. code-block::  
__device__ unsigned int __byte_perm(unsigned int x, unsigned int y, unsigned int s);


*Description:* Supported


**__clz
.. code-block::  
__device__ unsigned int __clz(int x);


*Description:* Supported


**__clzll
.. code-block::  
__device__ unsigned int __clzll(long long int x);


*Description:* Supported


**__ffs
.. code-block::  
__device__ unsigned int __ffs(int x);


*Description:* Supported


**__ffsll
.. code-block::  
__device__ unsigned int __ffsll(long long int x);


*Description:* Supported


**__hadd
.. code-block::  
__device__ static unsigned int __hadd(int x, int y);


*Description:* Supported


**__mul24
.. code-block::  
__device__ static int __mul24(int x, int y);


*Description:* Supported


**__mul64hi
.. code-block::  
__device__ long long int __mul64hi(long long int x, long long int y);


*Description:* Supported


**__mulhi
.. code-block::  
__device__ static int __mulhi(int x, int y);


*Description:* Supported


**__popc
.. code-block::  
__device__ unsigned int __popc(unsigned int x);


*Description:* Supported


**__popcll
.. code-block::  
__device__ unsigned int __popcll(unsigned long long int x);


*Description:* Supported


**__rhadd
.. code-block::  
__device__ static int __rhadd(int x, int y);


*Description:* Supported


**__sad
.. code-block::  
__device__ static unsigned int __sad(int x, int y, int z);


*Description:* Supported


**__uhadd
.. code-block::  
__device__ static unsigned int __uhadd(unsigned int x, unsigned int y);


*Description:* Supported


**__umul24
.. code-block::  
__device__ static int __umul24(unsigned int x, unsigned int y);


*Description:* Supported


**__umul64hi
.. code-block::  
__device__ unsigned long long int __umul64hi(unsigned long long int x, unsigned long long int y);


*Description:* Supported


**__umulhi
.. code-block::  
__device__ static unsigned int __umulhi(unsigned int x, unsigned int y);


*Description:* Supported


**__urhadd
.. code-block::  
__device__ static unsigned int __urhadd(unsigned int x, unsigned int y);


*Description:* Supported


**__usad
.. code-block::  
__device__ static unsigned int __usad(unsigned int x, unsigned int y, unsigned int z);


*Description:* Supported


**__double2float_rd
.. code-block::  
__device__ float __double2float_rd(double x);


*Description:* Supported


**__double2float_rn
.. code-block::  
__device__ float __double2float_rn(double x);


*Description:* Supported


**__double2float_ru
.. code-block::  
__device__ float __double2float_ru(double x);


*Description:* Supported


**__double2float_rz
.. code-block::  
__device__ float __double2float_rz(double x);


*Description:* Supported


**__double2hiint
.. code-block::  
__device__ int __double2hiint(double x);


*Description:* Supported


**__double2int_rd
.. code-block::  
__device__ int __double2int_rd(double x);


*Description:* Supported


**__double2int_rn
.. code-block::  
__device__ int __double2int_rn(double x);


*Description:* Supported


**__double2int_ru
.. code-block::  
__device__ int __double2int_ru(double x);


*Description:* Supported


**__double2int_rz
.. code-block::  
__device__ int __double2int_rz(double x);


*Description:* Supported


**__double2ll_rd
.. code-block::  
__device__ long long int __double2ll_rd(double x);


*Description:* Supported


**__double2ll_rn
.. code-block::  
__device__ long long int __double2ll_rn(double x);


*Description:* Supported


**__double2ll_ru
.. code-block::  
__device__ long long int __double2ll_ru(double x);


*Description:* Supported


**__double2ll_rz
.. code-block::  
__device__ long long int __double2ll_rz(double x);


*Description:* Supported


**__double2loint
.. code-block::  
__device__ int __double2loint(double x);


*Description:* Supported


**__double2uint_rd
.. code-block::  
__device__ unsigned int __double2uint_rd(double x);


*Description:* Supported


**__double2uint_rn
.. code-block::  
__device__ unsigned int __double2uint_rn(double x);


*Description:* Supported


**__double2uint_ru
.. code-block::  
__device__ unsigned int __double2uint_ru(double x);


*Description:* Supported


**__double2uint_rz
.. code-block::  
__device__ unsigned int __double2uint_rz(double x);


*Description:* Supported


**__double2ull_rd
.. code-block::  
__device__ unsigned long long int __double2ull_rd(double x);


*Description:* Supported


**__double2ull_rn
.. code-block::  
__device__ unsigned long long int __double2ull_rn(double x);


*Description:* Supported


**__double2ull_ru
.. code-block::  
__device__ unsigned long long int __double2ull_ru(double x);


*Description:* Supported


**__double2ull_rz
.. code-block::  
__device__ unsigned long long int __double2ull_rz(double x);


*Description:* Supported


**__double_as_longlong
.. code-block::  
__device__ long long int __double_as_longlong(double x);


*Description:* Supported


**__float2half_rn
.. code-block::  
__device__ unsigned short __float2half_rn(float x);


*Description:* Supported


**__half2float
.. code-block::  
__device__ float __half2float(unsigned short);


*Description:* Supported


**__float2half_rn
.. code-block::  
__device__ __half __float2half_rn(float x);


*Description:* Supported


**__half2float
.. code-block::  
__device__ float __half2float(__half);


*Description:* Supported


**__float2int_rd
.. code-block::  
__device__ int __float2int_rd(float x);


*Description:* Supported


**__float2int_rn
.. code-block::  
__device__ int __float2int_rn(float x);


*Description:* Supported


**__float2int_ru
.. code-block::  
__device__ int __float2int_ru(float x);


*Description:* Supported


**__float2int_rz
.. code-block::  
__device__ int __float2int_rz(float x);


*Description:* Supported


**__float2ll_rd
.. code-block::  
__device__ long long int __float2ll_rd(float x);


*Description:* Supported


**__float2ll_rn
.. code-block::  
__device__ long long int __float2ll_rn(float x);


*Description:* Supported


**__float2ll_ru
.. code-block::  
__device__ long long int __float2ll_ru(float x);


*Description:* Supported


**__float2ll_rz
.. code-block::  
__device__ long long int __float2ll_rz(float x);


*Description:* Supported


**__float2uint_rd
.. code-block::  
__device__ unsigned int __float2uint_rd(float x);


*Description:* Supported


**__float2uint_rn
.. code-block::  
__device__ unsigned int __float2uint_rn(float x);


*Description:* Supported


**__float2uint_ru
.. code-block::  
__device__ unsigned int __float2uint_ru(float x);


*Description:* Supported


**__float2uint_rz
.. code-block::  
__device__ unsigned int __float2uint_rz(float x);


*Description:* Supported


**__float2ull_rd
.. code-block::  
__device__ unsigned long long int __float2ull_rd(float x);


*Description:* Supported


**__float2ull_rn
.. code-block::  
__device__ unsigned long long int __float2ull_rn(float x);


*Description:* Supported


**__float2ull_ru
.. code-block::  
__device__ unsigned long long int __float2ull_ru(float x);


*Description:* Supported


**__float2ull_rz
.. code-block::  
__device__ unsigned long long int __float2ull_rz(float x);


*Description:* Supported


**__float_as_int
.. code-block::  
__device__ int __float_as_int(float x);


*Description:* Supported


**__float_as_uint
.. code-block::  
__device__ unsigned int __float_as_uint(float x);


*Description:* Supported


**__hiloint2double
.. code-block::  
__device__ double __hiloint2double(int hi, int lo);


*Description:* Supported


**__int2double_rn
.. code-block::  
__device__ double __int2double_rn(int x);


*Description:* Supported


**__int2float_rd
.. code-block::  
__device__ float __int2float_rd(int x);


*Description:* Supported


**__int2float_rn
.. code-block::  
__device__ float __int2float_rn(int x);


*Description:* Supported


**__int2float_ru
.. code-block::  
__device__ float __int2float_ru(int x);


*Description:* Supported


**__int2float_rz
.. code-block::  
__device__ float __int2float_rz(int x);


*Description:* Supported


**__int_as_float
.. code-block::  
__device__ float __int_as_float(int x);


*Description:* Supported


**__ll2double_rd
.. code-block::  
__device__ double __ll2double_rd(long long int x);


*Description:* Supported


**__ll2double_rn
.. code-block::  
__device__ double __ll2double_rn(long long int x);


*Description:* Supported


**__ll2double_ru
.. code-block::  
__device__ double __ll2double_ru(long long int x);


*Description:* Supported


**__ll2double_rz
.. code-block::  
__device__ double __ll2double_rz(long long int x);


*Description:* Supported


**__ll2float_rd
.. code-block::  
__device__ float __ll2float_rd(long long int x);


*Description:* Supported


**__ll2float_rn
.. code-block::  
__device__ float __ll2float_rn(long long int x);


*Description:* Supported


**__ll2float_ru
.. code-block::  
__device__ float __ll2float_ru(long long int x);


*Description:* Supported


**__ll2float_rz
.. code-block::  
__device__ float __ll2float_rz(long long int x);


*Description:* Supported


**__longlong_as_double
.. code-block::  
__device__ double __longlong_as_double(long long int x);


*Description:* Supported


**__uint2double_rn
.. code-block::  
__device__ double __uint2double_rn(int x);


*Description:* Supported


**__uint2float_rd
.. code-block::  
__device__ float __uint2float_rd(unsigned int x);


*Description:* Supported


**__uint2float_rn
.. code-block::  
__device__ float __uint2float_rn(unsigned int x);


*Description:* Supported


**__uint2float_ru
.. code-block::  
__device__ float __uint2float_ru(unsigned int x);


*Description:* Supported


**__uint2float_rz
.. code-block::  
__device__ float __uint2float_rz(unsigned int x);


*Description:* Supported


**__uint_as_float
.. code-block::  
__device__ float __uint_as_float(unsigned int x);


*Description:* Supported


**__ull2double_rd
.. code-block::  
__device__ double __ull2double_rd(unsigned long long int x);


*Description:* Supported


**__ull2double_rn
.. code-block::  
__device__ double __ull2double_rn(unsigned long long int x);


*Description:* Supported


**__ull2double_ru
.. code-block::  
__device__ double __ull2double_ru(unsigned long long int x);


*Description:* Supported


**__ull2double_rz
.. code-block::  
__device__ double __ull2double_rz(unsigned long long int x);


*Description:* Supported


**__ull2float_rd
.. code-block::  
__device__ float __ull2float_rd(unsigned long long int x);


*Description:* Supported


**__ull2float_rn
.. code-block::  
__device__ float __ull2float_rn(unsigned long long int x);


*Description:* Supported


**__ull2float_ru
.. code-block::  
__device__ float __ull2float_ru(unsigned long long int x);


*Description:* Supported


**__ull2float_rz
.. code-block::  
__device__ float __ull2float_rz(unsigned long long int x);


*Description:* Supported


**__hadd
.. code-block::  
__device__ static __half __hadd(const __half a, const __half b);


*Description:* Supported


**__hadd_sat
.. code-block::  
__device__ static __half __hadd_sat(__half a, __half b);


*Description:* Supported


**__hfma
.. code-block::  
__device__ static __half __hfma(__half a, __half b, __half c);


*Description:* Supported


**__hfma_sat
.. code-block::  
__device__ static __half __hfma_sat(__half a, __half b, __half c);


*Description:* Supported


**__hmul
.. code-block::  
__device__ static __half __hmul(__half a, __half b);


*Description:* Supported


**__hmul_sat
.. code-block::  
__device__ static __half __hmul_sat(__half a, __half b);


*Description:* Supported


**__hneg
.. code-block::  
__device__ static __half __hneg(__half a);


*Description:* Supported


**__hsub
.. code-block::  
__device__ static __half __hsub(__half a, __half b);


*Description:* Supported


**__hsub_sat
.. code-block::  
__device__ static __half __hsub_sat(__half a, __half b);


*Description:* Supported


**hdiv
.. code-block::  
__device__ static __half hdiv(__half a, __half b);


*Description:* Supported


**__hadd2
.. code-block::  
__device__ static __half2 __hadd2(__half2 a, __half2 b);


*Description:* Supported


**__hadd2_sat
.. code-block::  
__device__ static __half2 __hadd2_sat(__half2 a, __half2 b);


*Description:* Supported


**__hfma2
.. code-block::  
__device__ static __half2 __hfma2(__half2 a, __half2 b, __half2 c);


*Description:* Supported


**__hfma2_sat
.. code-block::  
__device__ static __half2 __hfma2_sat(__half2 a, __half2 b, __half2 c);


*Description:* Supported


**__hmul2
.. code-block::  
__device__ static __half2 __hmul2(__half2 a, __half2 b);


*Description:* Supported


**__hmul2_sat
.. code-block::  
__device__ static __half2 __hmul2_sat(__half2 a, __half2 b);


*Description:* Supported


**__hsub2
.. code-block::  
__device__ static __half2 __hsub2(__half2 a, __half2 b);


*Description:* Supported


**__hneg2
.. code-block::  
__device__ static __half2 __hneg2(__half2 a);


*Description:* Supported


**__hsub2_sat
.. code-block::  
__device__ static __half2 __hsub2_sat(__half2 a, __half2 b);


*Description:* Supported


**h2div
.. code-block::  
__device__ static __half2 h2div(__half2 a, __half2 b);


*Description:* Supported


**__heq
.. code-block::  
__device__  bool __heq(__half a, __half b);


*Description:* Supported


**__hge
.. code-block::  
__device__  bool __hge(__half a, __half b);


*Description:* Supported


**__hgt
.. code-block::  
__device__  bool __hgt(__half a, __half b);


*Description:* Supported


**__hisinf
.. code-block::  
__device__  bool __hisinf(__half a);


*Description:* Supported


**__hisnan
.. code-block::  
__device__  bool __hisnan(__half a);


*Description:* Supported


**__hle
.. code-block::  
__device__  bool __hle(__half a, __half b);


*Description:* Supported


**__hlt
.. code-block::  
__device__  bool __hlt(__half a, __half b);


*Description:* Supported


**__hne
.. code-block::  
__device__  bool __hne(__half a, __half b);


*Description:* Supported


**__hbeq2
.. code-block::  
__device__  bool __hbeq2(__half2 a, __half2 b);


*Description:* Supported


**__hbge2
.. code-block::  
__device__  bool __hbge2(__half2 a, __half2 b);


*Description:* Supported


**__hbgt2
.. code-block::  
__device__  bool __hbgt2(__half2 a, __half2 b);


*Description:* Supported


**__hble2
.. code-block::  
__device__  bool __hble2(__half2 a, __half2 b);


*Description:* Supported


**__hblt2
.. code-block::  
__device__  bool __hblt2(__half2 a, __half2 b);


*Description:* Supported


**__hbne2
.. code-block::  
__device__  bool __hbne2(__half2 a, __half2 b);


*Description:* Supported


**__heq2
.. code-block::  
__device__  __half2 __heq2(__half2 a, __half2 b);


*Description:* Supported


**__hge2
.. code-block::  
__device__  __half2 __hge2(__half2 a, __half2 b);


*Description:* Supported


**__hgt2
.. code-block::  
__device__  __half2 __hgt2(__half2 a, __half2 b);


*Description:* Supported


**__hisnan2
.. code-block::  
__device__  __half2 __hisnan2(__half2 a);


*Description:* Supported


**__hle2
.. code-block::  
__device__  __half2 __hle2(__half2 a, __half2 b);


*Description:* Supported


**__hlt2
.. code-block::  
__device__  __half2 __hlt2(__half2 a, __half2 b);


*Description:* Supported


**__hne2
.. code-block::  
__device__  __half2 __hne2(__half2 a, __half2 b);


*Description:* Supported


**hceil
.. code-block::  
__device__ static __half hceil(const __half h);


*Description:* Supported


**hcos
.. code-block::  
__device__ static __half hcos(const __half h);


*Description:* Supported


**hexp
.. code-block::  
__device__ static __half hexp(const __half h);


*Description:* Supported


**hexp10
.. code-block::  
__device__ static __half hexp10(const __half h);


*Description:* Supported


**hexp2
.. code-block::  
__device__ static __half hexp2(const __half h);


*Description:* Supported


**hfloor
.. code-block::  
__device__ static __half hfloor(const __half h);


*Description:* Supported


**hlog
.. code-block::  
__device__ static __half hlog(const __half h);


*Description:* Supported


**hlog10
.. code-block::  
__device__ static __half hlog10(const __half h);


*Description:* Supported


**hlog2
.. code-block::  
__device__ static __half hlog2(const __half h);


*Description:* Supported


**hrcp
.. code-block::  
//__device__ static __half hrcp(const __half h);


*Description:* **NOT Supported**


**hrint
.. code-block::  
__device__ static __half hrint(const __half h);


*Description:* Supported


**hsin
.. code-block::  
__device__ static __half hsin(const __half h);


*Description:* Supported


**hsqrt
.. code-block::  
__device__ static __half hsqrt(const __half a);


*Description:* Supported


**htrunc
.. code-block::  
__device__ static __half htrunc(const __half a);


*Description:* Supported


**h2ceil
.. code-block::  
__device__ static __half2 h2ceil(const __half2 h);


*Description:* Supported


**h2exp
.. code-block::  
__device__ static __half2 h2exp(const __half2 h);


*Description:* Supported


**h2exp10
.. code-block::  
__device__ static __half2 h2exp10(const __half2 h);


*Description:* Supported


**h2exp2
.. code-block::  
__device__ static __half2 h2exp2(const __half2 h);


*Description:* Supported


**h2floor
.. code-block::  
__device__ static __half2 h2floor(const __half2 h);


*Description:* Supported


**h2log
.. code-block::  
__device__ static __half2 h2log(const __half2 h);


*Description:* Supported


**h2log10
.. code-block::  
__device__ static __half2 h2log10(const __half2 h);


*Description:* Supported


**h2log2
.. code-block::  
__device__ static __half2 h2log2(const __half2 h);


*Description:* Supported


**h2rcp
.. code-block::  
__device__ static __half2 h2rcp(const __half2 h);


*Description:* Supported


**h2rsqrt
.. code-block::  
__device__ static __half2 h2rsqrt(const __half2 h);


*Description:* Supported


**h2sin
.. code-block::  
__device__ static __half2 h2sin(const __half2 h);


*Description:* Supported


**h2sqrt
.. code-block::  
__device__ static __half2 h2sqrt(const __half2 h);


*Description:* Supported


**__float22half2_rn
.. code-block::  
__device__  __half2 __float22half2_rn(const float2 a);


*Description:* Supported


**__float2half
.. code-block::  
__device__  __half __float2half(const float a);


*Description:* Supported


**__float2half2_rn
.. code-block::  
__device__  __half2 __float2half2_rn(const float a);


*Description:* Supported


**__float2half_rd
.. code-block::  
__device__  __half __float2half_rd(const float a);


*Description:* Supported


**__float2half_rn
.. code-block::  
__device__  __half __float2half_rn(const float a);


*Description:* Supported


**__float2half_ru
.. code-block::  
__device__  __half __float2half_ru(const float a);


*Description:* Supported


**__float2half_rz
.. code-block::  
__device__  __half __float2half_rz(const float a);


*Description:* Supported


**__floats2half2_rn
.. code-block::  
__device__  __half2 __floats2half2_rn(const float a, const float b);


*Description:* Supported


**__half22float2
.. code-block::  
__device__  float2 __half22float2(const __half2 a);


*Description:* Supported


**__half2float
.. code-block::  
__device__  float __half2float(const __half a);


*Description:* Supported


**half2half2
.. code-block::  
__device__  __half2 half2half2(const __half a);


*Description:* Supported


**__half2int_rd
.. code-block::  
__device__  int __half2int_rd(__half h);


*Description:* Supported


**__half2int_rn
.. code-block::  
__device__  int __half2int_rn(__half h);


*Description:* Supported


**__half2int_ru
.. code-block::  
__device__  int __half2int_ru(__half h);


*Description:* Supported


**__half2int_rz
.. code-block::  
__device__  int __half2int_rz(__half h);


*Description:* Supported


**__half2ll_rd
.. code-block::  
__device__  long long int __half2ll_rd(__half h);


*Description:* Supported


**__half2ll_rn
.. code-block::  
__device__  long long int __half2ll_rn(__half h);


*Description:* Supported


**__half2ll_ru
.. code-block::  
__device__  long long int __half2ll_ru(__half h);


*Description:* Supported


**__half2ll_rz
.. code-block::  
__device__  long long int __half2ll_rz(__half h);


*Description:* Supported


**__half2short_rd
.. code-block::  
__device__  short __half2short_rd(__half h);


*Description:* Supported


**__half2short_rn
.. code-block::  
__device__  short __half2short_rn(__half h);


*Description:* Supported


**__half2short_ru
.. code-block::  
__device__  short __half2short_ru(__half h);


*Description:* Supported


**__half2short_rz
.. code-block::  
__device__  short __half2short_rz(__half h);


*Description:* Supported


**__half2uint_rd
.. code-block::  
__device__  unsigned int __half2uint_rd(__half h);


*Description:* Supported


**__half2uint_rn
.. code-block::  
__device__  unsigned int __half2uint_rn(__half h);


*Description:* Supported


**__half2uint_ru
.. code-block::  
__device__  unsigned int __half2uint_ru(__half h);


*Description:* Supported


**__half2uint_rz
.. code-block::  
__device__  unsigned int __half2uint_rz(__half h);


*Description:* Supported


**__half2ull_rd
.. code-block::  
__device__  unsigned long long int __half2ull_rd(__half h);


*Description:* Supported


**__half2ull_rn
.. code-block::  
__device__  unsigned long long int __half2ull_rn(__half h);


*Description:* Supported


**__half2ull_ru
.. code-block::  
__device__  unsigned long long int __half2ull_ru(__half h);


*Description:* Supported


**__half2ull_rz
.. code-block::  
__device__  unsigned long long int __half2ull_rz(__half h);


*Description:* Supported


**__half2ushort_rd
.. code-block::  
__device__  unsigned short int __half2ushort_rd(__half h);


*Description:* Supported


**__half2ushort_rn
.. code-block::  
__device__  unsigned short int __half2ushort_rn(__half h);


*Description:* Supported


**__half2ushort_ru
.. code-block::  
__device__  unsigned short int __half2ushort_ru(__half h);


*Description:* Supported


**__half2ushort_rz
.. code-block::  
__device__  unsigned short int __half2ushort_rz(__half h);


*Description:* Supported


**__half_as_short
.. code-block::  
__device__  short int __half_as_short(const __half h);


*Description:* Supported


**__half_as_ushort
.. code-block::  
__device__  unsigned short int __half_as_ushort(const __half h);


*Description:* Supported


**__halves2half2
.. code-block::  
__device__  __half2 __halves2half2(const __half a, const __half b);


*Description:* Supported


**__high2float
.. code-block::  
__device__  float __high2float(const __half2 a);


*Description:* Supported


**__high2half
.. code-block::  
__device__  __half __high2half(const __half2 a);


*Description:* Supported


**__high2half2
.. code-block::  
__device__  __half2 __high2half2(const __half2 a);


*Description:* Supported


**__highs2half2
.. code-block::  
__device__  __half2 __highs2half2(const __half2 a, const __half2 b);


*Description:* Supported


**__int2half_rd
.. code-block::  
__device__  __half __int2half_rd(int i);


*Description:* Supported


**__int2half_rn
.. code-block::  
__device__  __half __int2half_rn(int i);


*Description:* Supported


**__int2half_ru
.. code-block::  
__device__  __half __int2half_ru(int i);


*Description:* Supported


**__int2half_rz
.. code-block::  
__device__  __half __int2half_rz(int i);


*Description:* Supported


**__ll2half_rd
.. code-block::  
__device__  __half __ll2half_rd(long long int i);


*Description:* Supported


**__ll2half_rn
.. code-block::  
__device__  __half __ll2half_rn(long long int i);


*Description:* Supported


**__ll2half_ru
.. code-block::  
__device__  __half __ll2half_ru(long long int i);


*Description:* Supported


**__ll2half_rz
.. code-block::  
__device__  __half __ll2half_rz(long long int i);


*Description:* Supported


**__low2float
.. code-block::  
__device__  float __low2float(const __half2 a);


*Description:* Supported


**__low2half
.. code-block::  
__device__ __half __low2half(const __half2 a);


*Description:* Supported


**__low2half2
.. code-block::  
__device__ __half2 __low2half2(const __half2 a, const __half2 b);


*Description:* Supported


**__low2half2
.. code-block::  
__device__ __half2 __low2half2(const __half2 a);


*Description:* Supported


**__lowhigh2highlow
.. code-block::  
__device__ __half2 __lowhigh2highlow(const __half2 a);


*Description:* Supported


**__lows2half2
.. code-block::  
__device__ __half2 __lows2half2(const __half2 a, const __half2 b);


*Description:* Supported


**__short2half_rd
.. code-block::  
__device__  __half __short2half_rd(short int i);


*Description:* Supported


**__short2half_rn
.. code-block::  
__device__  __half __short2half_rn(short int i);


*Description:* Supported


**__short2half_ru
.. code-block::  
__device__  __half __short2half_ru(short int i);


*Description:* Supported


**__short2half_rz
.. code-block::  
__device__  __half __short2half_rz(short int i);


*Description:* Supported


**__uint2half_rd
.. code-block::  
__device__  __half __uint2half_rd(unsigned int i);


*Description:* Supported


**__uint2half_rn
.. code-block::  
__device__  __half __uint2half_rn(unsigned int i);


*Description:* Supported


**__uint2half_ru
.. code-block::  
__device__  __half __uint2half_ru(unsigned int i);


*Description:* Supported


**__uint2half_rz
.. code-block::  
__device__  __half __uint2half_rz(unsigned int i);


*Description:* Supported


**__ull2half_rd
.. code-block::  
__device__  __half __ull2half_rd(unsigned long long int i);


*Description:* Supported


**__ull2half_rn
.. code-block::  
__device__  __half __ull2half_rn(unsigned long long int i);


*Description:* Supported


**__ull2half_ru
.. code-block::  
__device__  __half __ull2half_ru(unsigned long long int i);


*Description:* Supported


**__ull2half_rz
.. code-block::  
__device__  __half __ull2half_rz(unsigned long long int i);


*Description:* Supported


**__ushort2half_rd
.. code-block::  
__device__  __half __ushort2half_rd(unsigned short int i);


*Description:* Supported


**__ushort2half_rn
.. code-block::  
__device__  __half __ushort2half_rn(unsigned short int i);


*Description:* Supported


**__ushort2half_ru
.. code-block::  
__device__  __half __ushort2half_ru(unsigned short int i);


*Description:* Supported


**__ushort2half_rz
.. code-block::  
__device__  __half __ushort2half_rz(unsigned short int i);


*Description:* Supported


**__ushort_as_half
.. code-block::  
__device__  __half __ushort_as_half(const unsigned short int i);


*Description:* Supported

