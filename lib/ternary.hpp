#ifndef TERNARY_HPP
#define TERNARY_HPP

#include "arithmetic.hpp"

namespace Regulators {

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

}; // namespace Regulators

#endif // TERNARY_HPP