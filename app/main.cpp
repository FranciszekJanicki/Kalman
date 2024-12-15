#include "kalman.hpp"
#include <fmt/core.h>
#include <functional>

int main([[maybe_unused]] const int argc, [[maybe_unused]] char const* argv[])
{
    using Matrix = Linalg::Matrix<double>;
    using Kalman = Filter::Kalman<double>;

    auto const dt{1.0};
    auto const sigma_perc{1.0};
    auto const sigma_move{1.0};

    Kalman kalman{Matrix({{1.0}, {10.0}}),
                  Matrix({{1.0, dt}, {0.0, 1.0}}),
                  Matrix({{1.0, 0.0}, {0.0, 10.0}}),
                  Matrix({{1.0}, {1.0}}),
                  Matrix({{1.0}}),
                  Matrix({{1.0, 0.0}}),
                  Matrix({{std::pow(sigma_perc, 2)}}),
                  Matrix({{0.25 * std::pow(dt, 4), 0.5 * std::pow(dt, 3)}, {0.5 * std::pow(dt, 3), std::pow(dt, 2)}}) *
                      std::pow(sigma_move, 2)};

    auto i{0};
    while (i++ < 100) {
        try {
            std::invoke(kalman, Matrix({{1.0}, {0.0}}), Matrix({{1.0}, {0.0}}));
        } catch (std::runtime_error const& error) {
            fmt::print("{}", error.what());
        }

        kalman.print_state();
        kalman.print_covariance();
    }

    return 0;
}
