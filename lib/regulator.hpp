#ifndef REGULATOR_HPP
#define REGULATOR_HPP

#include "arithmetic.hpp"
#include "matrix.hpp"
#include <algorithm>
#include <cassert>
#include <concepts>
#include <functional>
#include <memory>
#include <ranges>
#include <utility>
#include <variant>
#include <vector>

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

    template <Linalg::Arithmetic Value>
    struct PID
#ifdef REGULATOR_PTR
        : public Base<Value>
#endif
    {
        Value operator()(this PID& self, const Value error, const Value dt) noexcept
        {
            self.sum += (error + self.previous_error) / 2 * dt;
            self.sum = std::clamp(self.sum, -self.windup, self.windup);
            return self.kp * error + self.kd * (error - std::exchange(self.previous_error, error)) / dt +
                   self.ki * self.sum;
        }

        Value kp{};
        Value ki{};
        Value kd{};
        Value windup{};

        Value sum{0};
        Value previous_error{0};
    };

    template <Linalg::Arithmetic Value>
    struct LQR
#ifdef REGULATOR_PTR
        : public Base<Value>
#endif
    {
    public:
        using Matrix = Linalg::Matrix<Value>;
        using RicattiSolutions = std::vector<Matrix>;

        constexpr LQR(Matrix const& state_transition,
                      Matrix const& input_transition,
                      Matrix const& state_cost,
                      Matrix const& input_cost,
                      Matrix const& end_condition,
                      std::uint64_t const samples) :
            state_transition_{state_transition},
            input_transition_{input_transition},
            state_cost_{state_cost},
            input_cost_{input_cost},
            ricatti_solutions_{get_ricatti_solutions(state_transition,
                                                     input_transition,
                                                     state_cost,
                                                     input_cost,
                                                     end_condition,
                                                     samples)}
        {}

        constexpr LQR(Matrix&& state_transition,
                      Matrix&& input_transition,
                      Matrix&& state_cost,
                      Matrix&& input_cost,
                      Matrix const& end_condition,
                      std::uint64_t const samples) noexcept :
            state_transition_{std::forward<Matrix>(state_transition)},
            input_transition_{std::forward<Matrix>(input_transition)},
            state_cost_{std::forward<Matrix>(state_cost)},
            input_cost_{std::forward<Matrix>(input_cost)},
            ricatti_solutions_{get_ricatti_solutions(state_transition,
                                                     input_transition,
                                                     state_cost,
                                                     input_cost,
                                                     end_condition,
                                                     samples)}
        {}

        Value operator()(this LQR& self, const Value error, const Value) noexcept
        {
            // implement lqr algorithm here
            return error;
        }

        [[nodiscard]] constexpr Matrix
        operator()(this LQR& self, std::uint64_t const sample, Matrix const& input, Matrix const& measurement)
        {
            auto error{input - measurement};
            return input - (get_optimal_gain(sample,
                                             self.ricatti_solutions_[sample],
                                             self.input_transition_,
                                             self.input_cost_) *
                            error);
        }

    private:
        static constexpr Matrix get_optimal_gain(std::uint64_t const sample,
                                                 Matrix const& ricatti,
                                                 Matrix const& input_transition,
                                                 Matrix const& input_cost)
        {
            return Matrix::inverse(input_cost).value() * Matrix::transpose(input_transition) * ricatti;
        }

        static constexpr RicattiSolutions get_ricatti_solutions(Matrix const& state_transition,
                                                                Matrix const& input_transition,
                                                                Matrix const& state_cost,
                                                                Matrix const& input_cost,
                                                                Matrix const& end_condition,
                                                                std::uint64_t const samples)
        {
            RicattiSolutions solutions{};
            solutions.reserve(samples);
            solutions.push_back(end_condition);
            for (std::uint64_t i{}; i < samples; ++i) {
                solutions.push_back(
                    get_ricatti_solution(state_transition, input_transition, state_cost, input_cost, solutions.back()));
            }
            std::ranges::reverse(solutions);
            return solutions;
        }

        static constexpr Matrix get_ricatti_solution(Matrix const& state_transition,
                                                     Matrix const& input_transition,
                                                     Matrix const& state_cost,
                                                     Matrix const& input_cost,
                                                     Matrix const& prev_solution)
        {
            return -1 * (prev_solution * state_transition -
                         prev_solution * input_transition * Matrix::inverse(input_cost).value() *
                             Matrix::transpose(input_transition) * prev_solution +
                         Matrix::transpose(state_transition) * prev_solution + state_cost);
        }

        Matrix state_transition_{};
        Matrix input_transition_{};
        Matrix state_cost_{};
        Matrix input_cost_{};

        RicattiSolutions ricatti_solutions_{};
    };

    template <Linalg::Arithmetic Value>
    struct ADRC
#ifdef REGULATOR_PTR
        : public Base<Value>
#endif
    {
        Value operator()(this ADRC& self, const Value error, const Value) noexcept
        {
            // implement adrc algorithm here
            return error;
        }
    };

    template <Linalg::Arithmetic Value>
    struct Binary
#ifdef REGULATOR_PTR
        : public Base<Value>
#endif
    {
        enum struct State {
            POSITIVE,
            ZERO,
        };

        Value operator()(this Binary& self, const Value error, const Value dt) noexcept
        {
            return error;
        }

        State operator()(this Binary& self, const Value error) noexcept
        {
            switch (self.state) {
                case State::POSITIVE:
                    if (error < self.hysteresis_down) {
                        self.state = State::ZERO;
                    } else {
                        self.state = State::POSITIVE;
                    }
                    break;
                case State::ZERO:
                    if (error > self.hysteresis_up) {
                        self.state = State::POSITIVE;
                    } else {
                        self.state = State::ZERO;
                    }
                    break;
                default:
                    break;
            }
            return self.state;
        }

        Value hysteresis_up{};
        Value hysteresis_down{};

        State state{State::ZERO};
    };

    template <Linalg::Arithmetic Value>
    struct Ternary
#ifdef REGULATOR_PTR
        : public Base<Value>
#endif
    {
        enum struct State {
            POSITIVE,
            NEGATIVE,
            ZERO,
        };

        Value operator()(this Ternary& self, const Value error, const Value dt) noexcept
        {
            return error;
        }

        State operator()(this Ternary& self, const Value error) noexcept
        {
            switch (self.state) {
                case State::POSITIVE:
                    if (error < self.hysteresis_down) {
                        self.state = State::ZERO;
                    }
                    break;
                case State::NEGATIVE:
                    if (error > self.hysteresis_up) {
                        self.state = State::ZERO;
                    }
                    break;
                case State::ZERO:
                    if (error > self.hysteresis_up) {
                        self.state = State::POSITIVE;
                    } else if (error < self.hysteresis_down) {
                        self.state = State::NEGATIVE;
                    }
                    break;
                default:
                    break;
            }
            return self.state;
        }

        Value hysteresis_up{};
        Value hysteresis_down{};

        State state{State::ZERO};
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
}

#endif // REGULATOR_PTR
}
; // namespace Regulator

#endif // REGULATOR_HPP