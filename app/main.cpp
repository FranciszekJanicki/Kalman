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

    Kalman kalman{Matrix::make_eye(1),
                  Matrix::make_eye(1),
                  Matrix::make_eye(1),
                  Matrix::make_eye(1),
                  Matrix::make_eye(1),
                  Matrix::make_eye(1)};

    auto i{0};
    while (i++ < 100) {
        std::invoke(kalman, Matrix::make_column(1), Matrix::make_column(1));
        kalman.print_state();
    }

    return 0;
}
