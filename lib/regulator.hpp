#ifndef REGULATOR_HPP
#define REGULATOR_HPP

#include "adrc.hpp"
#include "binary.hpp"
#include "lqr.hpp"
#include "pid.hpp"
#include "ternary.hpp"
#include <functional>
#include <memory>
#include <utility>
#include <variant>

// i have chosen std::variant for polymorphic regulator, as its sigma container
#define REGULATOR_VARIANT

namespace Regulator {

    enum struct Algorithm {
        PID,
        LQR,
        ADRC,
        BINARY,
        TERNARY,
    };

    template <typename First, typename... Rest>
    struct FirstType {
        using Type = First;
    };

/* VARIANT SOLUTION (COMPILE TIME POLYMORPHISM) */
#ifdef REGULATOR_VARIANT

    template <Linalg::Arithmetic Value>
    using Regulator = std::variant<LQR<Value>, PID<Value>, ADRC<Value>, Binary<Value>, Ternary<Value>>;

    template <Algorithm algorithm, typename... Args>
    [[nodiscard]] auto make_regulator(Args... args) noexcept
    {
        using Value = typename FirstType<Args...>::Type;

        if constexpr (algorithm == Algorithm::PID) {
            return Regulator<Value>{std::in_place_type<PID<Value>>, args...};
        }
        if constexpr (algorithm == Algorithm::LQR) {
            return Regulator<Value>{std::in_place_type<LQR<Value>>, args...};
        }
        if constexpr (algorithm == Algorithm::ADRC) {
            return Regulator<Value>{std::in_place_type<ADRC<Value>>, args...};
        }
        if constexpr (algorithm == Algorithm::BINARY) {
            return Regulator<Value>{std::in_place_type<Binary<Value>>, args...};
        }
        if constexpr (algorithm == Algorithm::TERNARY) {
            return Regulator<Value>{std::in_place_type<Ternary<Value>>, args...};
        }
    }

#endif // REGULATOR_VARIANT

/* LAMBDA SOLUTION (TYPE ERASURE, although you will need std::function to containerize state-full lambda) */
#ifdef REGULATOR_LAMBDA

    template <Linalg::Arithmetic Value>
    using Regulator = std::function<Value(Value, Value)>;

    template <Algorithm algorithm, Linalg::Arithmetic... Args>
    [[nodiscard]] auto make_regulator(Args... args) noexcept
    {
        using Value = typename FirstType<Args...>::Type;

        if constexpr (algorithm == Algorithm::PID) {
            return [pid = PID<Value>{args...}](const Value error, const Value dt) mutable { return pid(error, dt); };
        }
        if constexpr (algorithm == Algorithm::LQR) {
            return [lqr = LQR<Value>{args...}](const Value error, const Value dt) mutable { return lqr(error, dt); };
        }
        if constexpr (algorithm == Algorithm::ADRC) {
            return [adrc = ADRC<Value>{args...}](const Value error, const Value dt) mutable { return adrc(error, dt); };
        }
        if constexpr (algorithm == Algorithm::BINARY) {
            return [binary = Binary<Value>{args...}](const Value error) mutable { return binary(error); };
        }
        if constexpr (algorithm == Algorithm::TERNARY) {
            return [ternary = Ternary<Value>{args...}](const Value error) mutable { return ternary(error); };
        }
    }
}
#endif // REGULATOR_LAMBDA

/* UNIQUE_PTR SOLUTION (RUNTIME POLYMORPHISM)*/
#ifdef REGULATOR_PTR

/* REGULATOR BASE */
template <Linalg::Arithmetic Value>
struct Base {
    virtual ~Base() noexcept = 0;
};

template <Linalg::Arithmetic Value>
using Regulator = std::unique_ptr<Regulator<Value>>;

template <Algorithm algorithm, Linalg::Arithmetic... Args>
[[nodiscard]] Regulator<Value> make_regulator(Args... args)
{
    using Value = typename FirstType<Args...>::Type;

    if constexpr (algorithm == Algorithm::PID) {
        return std::make_unique<PID<Value>>(args...);
    }
    if constexpr (algorithm == Algorithm::LQR) {
        return std::make_unique<LQR<Value>>(args...);
    }
    if constexpr (algorithm == Algorithm::ADRC) {
        return std::make_unique<ADRC<Value>>(args...);
    }
    if constexpr (algorithm == Algorithm::BINARY) {
        return std::make_unique<Binary<Value>>(args...);
    }
    if constexpr (algorithm == Algorithm::TERNARY) {
        return std::make_unique<Ternary<Value>>(args...);
    }
}

#endif // REGULATOR_PTR
}
; // namespace Regulator

#endif // REGULATOR_HPP