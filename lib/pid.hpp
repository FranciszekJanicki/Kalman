#ifndef PID_HPP
#define PID_HPP

#include "arithmetic.hpp"

namespace Regulator {

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
            return self.proportional_gain * error +
                   self.derivative_gain * (error - std::exchange(self.previous_error, error)) / dt +
                   self.integral_gain * self.sum;
        }

        Value proportional_gain{};
        Value integral_gain{};
        Value derivative_gain{};
        Value windup{};

        Value sum{0};
        Value previous_error{0};
    };

}; // namespace Regulator

#endif // PID_HPP
