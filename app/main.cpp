#include "heap_matrix.hpp"
#include "heap_vector.hpp"
#include "kalman.hpp"
#include "quaternion3d.hpp"
#include "stack_matrix.hpp"
#include "stack_vector.hpp"
#include "vector3d.hpp"
#include <fmt/core.h>
#include <functional>

int main([[maybe_unused]] int const argc, [[maybe_unused]] char const* argv[])
{
    using Matrix = Linalg::Heap::Matrix<double>;
    using Kalman = Filters::Kalman<double>;

    auto const dt{1.0};
    auto const sigma_perc{1.0};
    auto const sigma_move{1.0};

    Kalman kalman{Matrix{{1.0}, {10.0}},
                  Matrix{{1.0, dt}, {0.0, 1.0}},
                  Matrix{{1.0, 0.0}, {0.0, 10.0}},
                  Matrix{{1.0}, {1.0}},
                  Matrix{{1.0}},
                  Matrix{{1.0, 0.0}},
                  Matrix{{std::pow(sigma_perc, 2)}},
                  Matrix{{0.25 * std::pow(dt, 4), 0.5 * std::pow(dt, 3)}, {0.5 * std::pow(dt, 3), std::pow(dt, 2)}} *
                      std::pow(sigma_move, 2)};

    auto i{0};
    while (i++ < 100) {
        try {
            std::invoke(kalman, Matrix{{1.0}, {0.0}}, Matrix{{1.0}, {0.0}});
        } catch (std::runtime_error const& error) {
            fmt::print("{}", error.what());
        }
    }

    return 0;
}
