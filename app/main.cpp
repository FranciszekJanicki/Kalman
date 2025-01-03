#include "quaternion3d.hpp"
#include "stack_kalman.hpp"
#include "stack_matrix.hpp"
#include "stack_vector.hpp"
#include "vector3d.hpp"
#include <fmt/core.h>
#include <functional>

int main([[maybe_unused]] int const argc, [[maybe_unused]] char const* argv[])
{
    using namespace Linalg::Stack;
    using Kalman = Kalman<double, 2>;
    using Matrix2x2 = Kalman::Matrix<2, 2>;
    using Matrix1x2 = Kalman::Matrix<1, 2>;
    using Matrix2x1 = Kalman::Matrix<2, 1>;
    using Matrix1x1 = Kalman::Matrix<1, 1>;

    auto const dt{1.0};
    auto const sigma_perc{1.0};
    auto const sigma_move{1.0};

    Kalman kalman{Matrix2x1{{1.0}, {0.0}},
                  Matrix2x2{{1.0, dt}, {0.0, 1.0}},
                  Matrix2x2{{1.0, 0.0}, {0.0, 10.0}},
                  Matrix2x1{{1.0}, {1.0}},
                  Matrix1x1{{1.0}},
                  Matrix1x2{{1.0, 0.0}},
                  Matrix1x1{{std::pow(sigma_perc, 2)}},
                  Matrix2x2{{0.25 * std::pow(dt, 4), 0.5 * std::pow(dt, 3)}, {0.5 * std::pow(dt, 3), std::pow(dt, 2)}} *
                      std::pow(sigma_move, 2)};

    auto i{0};
    while (i++ < 100) {
        try {
            auto const kalman_state = std::invoke(kalman, Matrix1x1{{0.0}}, Matrix1x1{{0.0}});
            kalman_state.print();
        } catch (std::runtime_error const& error) {
            fmt::print("error: {}", error.what());
        }
    }

    return 0;
}
