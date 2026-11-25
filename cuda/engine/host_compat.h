// =============================================================================
// host_compat.h
//
// Lets the rules layer (movegen, make/unmake, attacks, zobrist) compile with
// a host C++ compiler when ENGINE_HOST_ONLY is defined. GPU builds include
// the real CUDA runtime instead and leave the compiler's own attributes.
// =============================================================================
#ifndef ENGINE_HOST_COMPAT_H
#define ENGINE_HOST_COMPAT_H

#ifdef ENGINE_HOST_ONLY

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __global__
#define __global__
#endif
#ifndef __constant__
#define __constant__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif

#else

#include <cuda_runtime.h>

#endif // ENGINE_HOST_ONLY

#endif // ENGINE_HOST_COMPAT_H
