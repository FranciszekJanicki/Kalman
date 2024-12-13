#ifndef REGULATOR_HPP
#define REGULATOR_HPP

#include "arithmetic.hpp"
#include "matrix.hpp"
#include "regulator.hpp"
#include <cassert>
#include <fmt/core.h>
#include <functional>
#include <memory>
#include <utility>
#include <variant>

namespace Regulator {

#define REGULATOR_VARIANT

    enum struct Algorithm {
        PID,
        LQR,
        ADRC,
        BINARY,
        TERNARY,
    };

#if defined(REGULATOR_VARIANT)

    template <Linalg::Arithmetic Value>
    using Regulator = std::variant<LQR<Value>>;

    template <Linalg::Arithmetic Value>
    [[nodiscard]] constexpr Regulator<Value> make_regulator()
    {}

#elif defined(REGULATOR_PTR)

    template <Linalg::Arithmetic Value>
    using Regulator = std::unique_ptr<LQR<Value>>;

    template <Linalg::Arithmetic Value>
    [[nodiscard]] constexpr Regulator<Value> make_regulator()
    {}

#elif defined(REGULATOR_LAMBDA)

    template <Linalg::Arithmetic Value>
    using Regulator = std::function<Value(Value)>;

    template <Linalg::Arithmetic Value>
    [[nodiscard]] constexpr auto make_regulator()
    {}

#endif

}; // namespace Regulator

#endif // REGULATOR_HPP