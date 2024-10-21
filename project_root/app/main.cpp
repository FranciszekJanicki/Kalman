#include "kalman.hpp"
#include "matrix.hpp"
#include "vector3d.hpp"
#include "vector6d.hpp"
#include "quaternion3d.hpp"

using Matrix = Linalg::Matrix;

int main([[maybe_unused]] int argc, [[maybe_unused]] char const* argv[])
{
    const std::size_t states{2};
    const std::size_t measurements{2};
    const std::size_t inputs{2};
    Kalman kalman{filter_model{states,
                               inputs,
                                Matrix::ones(states, inputs),
                                Matrix::ones(states, 1),
                                Matrix::ones(states, inputs),
                                Matrix::ones(inputs, 1),
                                Matrix::ones(states, states),
                                Matrix::ones(inputs, inputs)},
                    measure_model{states,
                                measurements,
                                Matrix::ones(measurements, states),
                                Matrix::ones(measurements, 1),
                                Matrix::ones(measurements, measurements),
                                Matrix::ones(measurements, 1),
                                Matrix::ones(measurements, measurements),
                                Matrix::ones(states, measurements)}};

    const auto iterations{100};
    auto i{0};
    while (i++ < iterations) {
        kalman_filter.predict();
        kalman_filter.print_predicted();
        kalman_filter.update();
        kalman_filter.print_state();
    }

    Vector3D dupa{1, 2, 3};
    Vector6D dupa1{1, 2, 3, 4, 5, 6};
    Quaternion3D dupa2{0, 1, 2, 3};

    return 0;
}
