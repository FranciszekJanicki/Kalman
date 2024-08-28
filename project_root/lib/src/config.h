#define ENV_DEBUG

#ifdef ENV_DEBUG
#include <iostream>
#define LOG(error) (std::cerr << error \n;)
#else
#define LOG(info) (std::cout << info \n;)
#endif
