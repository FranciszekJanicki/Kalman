#include "kalman.hpp"
#include "matrix.hpp"

int main([[maybe_unused]] int argc, [[maybe_unused]] char const* argv[])
{
    const std::size_t states{2};
    const std::size_t measurements{2};
    const std::size_t inputs{2};
    kalman kalman_filter{filter_model{
                             states,
                             inputs,
                             matrix<float>::ones(states, inputs),
                             matrix<float>::ones(states, 1),
                             matrix<float>::ones(states, inputs),
                             matrix<float>::ones(inputs, 1),
                             matrix<float>::ones(states, states),
                             matrix<float>::ones(inputs, inputs),
                         },
                         measure_model{
                             states,
                             measurements,
                             matrix<float>::ones(measurements, states),
                             matrix<float>::ones(measurements, 1),
                             matrix<float>::ones(measurements, measurements),
                             matrix<float>::ones(measurements, 1),
                             matrix<float>::ones(measurements, measurements),
                             matrix<float>::ones(states, measurements),
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
