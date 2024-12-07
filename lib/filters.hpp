#ifndef FILTERS_HPP
#define FILTERS_HPP

#include "arithmetic.hpp"
#include "kalman.hpp"
#include <functional>
#include <queue>
#include <utility>

namespace Filter {
    template <typename Filtered>
    using Filter = std::function<Filtered(Filtered)>;

    template <typename Filtered, Linalg::Arithmetic Sample = Filtered>
    [[nodiscard]] constexpr auto make_recursive_average(Filtered const start_condition = 0) noexcept
    {
        return [estimate = Filtered{}, prev_estimate = start_condition, samples = Sample{1}](
                   Filtered const measurement) mutable {
            estimate = prev_estimate * Sample{samples - 1} / samples + measurement / samples;
            prev_estimate = estimate;
            samples += Sample{1};
            return estimate;
        };
    }

    template <typename Filtered, Linalg::Arithmetic Sample = Filtered>
    [[nodiscard]] constexpr auto make_moving_average(Filtered const start_condition = 0, Sample const last_samples = 10)
    {
        assert(last_samples > 0);
        return [estimate = Filtered{},
                prev_estimate = start_condition,
                measurements = std::queue<Filtered>{start_condition},
                last_samples](Filtered const measurement) mutable {
            estimate = prev_estimate + (measurement - measurements.front()) / last_samples;
            measurements.pop();
            measurements.push(measurement);
            prev_estimate = estimate;
            return estimate;
        };
    }

    template <typename Filtered, Linalg::Arithmetic Sample = Filtered, Linalg::Arithmetic Alpha = Filtered>
    [[nodiscard]] constexpr auto make_low_pass(Filtered const start_condition = 0, Alpha const alpha = 0.5) noexcept
    {
        assert(alpha >= 0 && alpha <= 1);
        return [estimate = Filtered{}, prev_estimate = start_condition, alpha](Filtered const measurement) {
            estimate = prev_estimate * alpha + measurement * Alpha{1 - alpha};
            prev_estimate = estimate;
            return estimate;
        };
    }

}; // namespace Filter

#endif // FILTERS_HPP