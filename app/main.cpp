#include "filters.hpp"
#include "matrix.hpp"
#include "rotation3d.hpp"
#include "vector3d.hpp"
#include "vector6d.hpp"

int main([[maybe_unused]] const int argc, [[maybe_unused]] char const* argv[])
{
    using Kalman = Filter::Kalman<float>;
    using FilterModel = Kalman::FilterModel;
    using MeasureModel = Kalman::MeasureModel;
    using Matrix = Linalg::Matrix<float>;

    const std::size_t states{2};
    const std::size_t measurements{2};
    const std::size_t inputs{2};
    auto kalman = Filter::make_kalman<float>(FilterModel{states,
                                                         inputs,
                                                         Matrix::ones(states, inputs),
                                                         Matrix::ones(states, inputs),
                                                         Matrix::ones(states, states),
                                                         Matrix::ones(inputs, inputs)},
                                             MeasureModel{states,
                                                          measurements,
                                                          Matrix::ones(measurements, states),
                                                          Matrix::ones(measurements, measurements),
                                                          Matrix::ones(measurements, 1),
                                                          Matrix::ones(measurements, measurements),
                                                          Matrix::ones(states, measurements)});

    const auto iterations{100};
    auto i{0};
    while (i++ < iterations) {
        kalman(Matrix::ones(measurements, 1));
    }

    return 0;
}
