#include "filters.hpp"
#include "matrix.hpp"
#include "rotation3d.hpp"
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

    Kalman kalman{Matrix::ones(states, inputs),
                  Matrix::ones(states, states),
                  Matrix::ones(states, inputs),
                  Matrix::ones(inputs, inputs),
                  Matrix::ones(measurements, states),
                  Matrix::ones(measurements, measurements)};

    auto i{0};
    while (i++ < 100) {
        std::invoke(kalman, Matrix::ones(states, 1), Matrix::ones(states, 1));
    }

    return 0;
}
