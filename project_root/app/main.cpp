#include "dijkstra.hpp"
#include "kalman.hpp"
#include "matrix.hpp"
#include "quaternion3d.hpp"
#include "rotation3d.hpp"
#include "vector3d.hpp"
#include "vector6d.hpp"

int main([[maybe_unused]] int argc, [[maybe_unused]] char const* argv[])
{
    const std::size_t states{2};
    const std::size_t measurements{2};
    const std::size_t inputs{2};
    Kalman kalman{FilterModel{states,
                              inputs,
                              Matrix::ones(states, inputs),
                              Matrix::ones(states, 1),
                              Matrix::ones(states, inputs),
                              Matrix::ones(inputs, 1),
                              Matrix::ones(states, states),
                              Matrix::ones(inputs, inputs)},
                  MeasureModel{states,
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
        kalman.predict();
        kalman.print_predicted();
        kalman.update();
        kalman.print_state();
    }

    return 0;
}
