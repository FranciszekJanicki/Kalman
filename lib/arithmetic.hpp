#ifndef ARITHMETIC_HPP
#define ARITHMETIC_HPP

#include <concepts>
#include <type_traits>

namespace Linalg {

    template <typename Type>
    concept Arithmetic = std::is_arithmetic_v<Type>;

}; // namespace Linalg

#endif // ARITHMETIC_HPP