#define USE_EXCEPTIONS 0

#if USE_EXCEPTIONS
#include <cstdio>
#define LOG(error) (printf("%s\n", error))
#else 
#define LOG(error)
#endif

