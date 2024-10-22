#ifndef FILTERS_HPP
#define FILTERS_HPP

#include "arithmetic.hpp"
#include <queue>

template <typename Filtered, typename Sample = Filtered>
[[nodiscard]] constexpr auto make_recursive_average(const Filtered start_condition = {}) noexcept
{
    return [estimate = Filtered{}, prev_estimate = start_condition, samples = Sample{0}](
               const Filtered measurement) mutable {
        estimate = prev_estimate * (samples - Sample{1}) / samples + measurement / samples;
        prev_estimate = estimate;
        return estimate;
    };
}

template <typename Filtered, typename Sample = Filtered>
[[nodiscard]] constexpr auto make_moving_average(const Filtered start_condition = {}, const Sample last_samples = 10)
{
    assert(samples > 0);
    std::queue<Filtered> queue{};
    queue.push(start_condition);
    return [estimate = Filtered{}, prev_estimate = start_condition, measurements = std::move(queue), last_samples](
               const Filtered measurement) mutable {
        estimate = prev_estimate + (measurement - measurements.front()) / last_samples;
        measurements.pop();
        measurements.push(measurement);
        prev_estimate = estimate;
        return estimate;
    };
}

template <typename Filtered, typename Sample = Filtered, typename Aplha = Filtered>
[[nodiscard]] constexpr auto make_low_pass(const Filtered start_condition = {}, const Aplha alpha = 1) noexcept
{
    assert(alpha >= 0 && alpha <= 1);
    return [estimate = Filtered{}, prev_estimate = start_condition, alpha](const Filtered measurement) {
        estimate = prev_estimate * alpha + measurement * (Filtered{1} - alpha);
        prev_estimate = estimate;
        return estimate;
    };
}

#endif // FILTERS_HPP