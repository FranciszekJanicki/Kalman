#include "filters.hpp"
#include "matrix.hpp"
#include "rotation3d.hpp"
#include "vector.hpp"
#include "vector3d.hpp"
#include "vector6d.hpp"
#include <fmt/core.h>
#include <functional>

int main([[maybe_unused]] const int argc, [[maybe_unused]] char const* argv[])
{
    using Matrix = Linalg::Matrix<float>;
    using Kalman = Filter::Kalman<float>;

    const auto states{1};
    const auto measurements{1};
    const auto inputs{1};

    Kalman kalman{Matrix::make_ones(states, inputs),
                  Matrix::make_ones(states, states),
                  Matrix::make_ones(states, inputs),
                  Matrix::make_ones(inputs, inputs),
                  Matrix::make_ones(measurements, states),
                  Matrix::make_ones(measurements, measurements)};

    auto i{0};
    while (i++ < 100) {
        std::invoke(kalman, Matrix::make_ones(states, 1), Matrix::make_ones(states, 1));
        kalman.print_state();
    }

    return 0;
}
