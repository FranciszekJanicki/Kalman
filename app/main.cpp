#include "filters.hpp"
#include "matrix.hpp"
#include "rotation3d.hpp"
#include "vector3d.hpp"
#include "vector6d.hpp"
#include <fmt/core.h>

int main([[maybe_unused]] const int argc, [[maybe_unused]] char const* argv[])
{
    auto make_kalman{&Filter::make_kalman<float>};
    auto make_measure_model{&Filter::Kalman<float>::make_measure_model};
    auto make_filter_model{&Filter::Kalman<float>::make_filter_model};
    auto ones{&Linalg::Matrix<float>::ones};

    const auto states{2U};
    const auto measurements{2U};
    const auto inputs{2U};

    auto kalman = make_kalman(
        make_filter_model(ones(states, inputs), ones(states, inputs), ones(states, states), ones(inputs, inputs)),
        make_measure_model(ones(measurements, states),
                           ones(measurements, measurements),
                           ones(measurements, 1U),
                           ones(measurements, measurements),
                           ones(states, measurements)));

    for (auto i{0U}; i < 100U; ++i) {
        kalman(ones(measurements, 1U));
        fmt::print("dupa");
    }

    return 0;
}
