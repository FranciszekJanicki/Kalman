#ifndef FILTERS_HPP
#define FILTERS_HPP

#include "arithmetic.hpp"
#include <queue>

template <typename Filtered, typename Samples = Filtered>
[[nodiscard]] constexpr auto make_recursive_average(const Filtered start_condition = {}) noexcept
{
    return [estimate = Filtered{}, prev_estimate = start_condition, samples = Samples{1}](
               const Filtered measurement) mutable {
        estimate = prev_estimate * Samples{samples - 1} / samples + measurement / samples;
        prev_estimate = estimate;
        samples += Samples{1};
        return estimate;
    };
}

template <typename Filtered, typename Samples = Filtered>
[[nodiscard]] constexpr auto make_moving_average(const Filtered start_condition = {}, const Samples last_samples = 10)
{
    assert(last_samples > 0);
    return [estimate = Filtered{},
            prev_estimate = start_condition,
            measurements = std::queue<Filtered>{start_condition},
            last_samples](const Filtered measurement) mutable {
        estimate = prev_estimate + (measurement - measurements.front()) / last_samples;
        measurements.pop();
        measurements.push(measurement);
        prev_estimate = estimate;
        return estimate;
    };
}

template <typename Filtered, typename Samples = Filtered, typename Alpha = Filtered>
[[nodiscard]] constexpr auto make_low_pass(const Filtered start_condition = {}, const Alpha alpha = 1) noexcept
{
    assert(alpha >= 0 && alpha <= 1);
    return [estimate = Filtered{}, prev_estimate = start_condition, alpha](const Filtered measurement) {
        estimate = prev_estimate * alpha + measurement * Alpha{1 - alpha};
        prev_estimate = estimate;
        return estimate;
    };
}

#endif // FILTERS_HPP