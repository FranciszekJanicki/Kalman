#include "kalman.hpp"
#include "matrix_wrapper.hpp"

int main([[maybe_unused]] int argc, [[maybe_unused]] char const* argv[])
{
    const std::size_t states{2};
    const std::size_t measurements{2};
    const std::size_t inputs{2};
    kalman kalman_filter{filter_model{
                             states,
                             inputs,
                             matrix_wrapper<float>::zeros(states, inputs),
                             matrix_wrapper<float>::zeros(states, 1),
                             matrix_wrapper<float>::zeros(states, inputs),
                             matrix_wrapper<float>::zeros(inputs, 1),
                             matrix_wrapper<float>::zeros(states, states),
                             matrix_wrapper<float>::zeros(inputs, inputs),
                         },
                         measure_model{
                             states,
                             measurements,
                             matrix_wrapper<float>::zeros(measurements, states),
                             matrix_wrapper<float>::zeros(measurements, 1),
                             matrix_wrapper<float>::zeros(measurements, measurements),
                             matrix_wrapper<float>::zeros(measurements, 1),
                             matrix_wrapper<float>::zeros(measurements, measurements),
                             matrix_wrapper<float>::zeros(states, measurements),
                         }};

    const auto iterations{100};
    auto i{0};
    while (i++ < iterations) {
        kalman_filter.predict();
        kalman_filter.print_predicted();
        kalman_filter.update();
        kalman_filter.print_state();
    }

    return 0;
}
