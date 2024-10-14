#include "kalman.hpp"
#include <cassert>
#include <print>
#include <utility>

namespace lib {

    void kalman::initialize() noexcept
    {
        std::print("Checking correct dimensions");
        assert(A_.rows() == states_);
        assert(A_.columns() == inputs_);
        assert(B_.rows() == states_);
        assert(B_.columns() == inputs_);
        assert(u_.rows() == inputs_);
        assert(u_.columns() == 1);
        assert(P_.rows() == states_);
        assert(P_.columns() == states_);
        assert(Q_.rows() == inputs_);
        assert(Q_.columns() == inputs_);
        assert(x_.rows() == states_);
        assert(x_.columns() == 1);
        assert(H_.rows() == measurements_);
        assert(H_.columns() == states_);
        assert(z_.rows() == measurements_);
        assert(z_.columns() == 1);
        assert(R_.rows() == measurements_);
        assert(R_.columns() == measurements_);
        assert(y_.rows() == measurements_);
        assert(y_.columns() == 1);
        assert(S_.rows() == measurements_);
        assert(S_.columns() == measurements_);
        assert(K_.rows() == states_);
        assert(K_.columns() == measurements_);
        std::print("Filter succesfully initialized");
        is_initialized_ = true;
    }

    kalman::kalman(const filter_model& filter, const measure_model& measure) :

        states_{filter.inputs},
        inputs_{filter.inputs},
        measurements_{measure.measurements},
        A_(filter.A),
        B_(filter.B),
        u_(filter.u),
        P_(filter.P),
        Q_(filter.Q),
        x_(filter.x),
        H_(measure.H),
        z_(measure.z),
        R_(measure.R),
        y_(measure.z),
        S_(measure.S),
        K_(measure.K)
    {
        initialize();
    }

    kalman::kalman(filter_model&& filter, measure_model&& measure) noexcept :
        states_{filter.inputs},
        inputs_{filter.inputs},
        measurements_{measure.measurements},
        A_(std::move(filter.A)),
        B_(std::move(filter.B)),
        u_(std::move(filter.u)),
        P_(std::move(filter.P)),
        Q_(std::move(filter.Q)),
        x_(std::move(filter.x)),
        H_(std::move(measure.H)),
        z_(std::move(measure.z)),
        R_(std::move(measure.R)),
        y_(std::move(measure.z)),
        S_(std::move(measure.S)),
        K_(std::move(measure.K))
    {
        initialize();
    }

    matrix<float>&& kalman::state() && noexcept
    {
        return std::move(x_);
    }

    const matrix<float>& kalman::state() const& noexcept
    {
        return x_;
    }

    void kalman::inputs(const matrix<float>& inputs)
    {
        u_ = inputs;
    }

    void kalman::inputs(matrix<float>&& inputs) noexcept
    {
        u_ = std::forward<matrix<float>>(inputs);
    }

    void kalman::predict()
    {
        if (!is_initialized_) {
            std::print("Filter unitialized!");
            return;
        }

        // using temp variables to store result of one operation and use it to perform next

        /* predict state        */
        // xP = A*x

        xP_ = A_ * x_;

        /************************/

        /* predict covariance   */
        // P = A*P*A + B*Q*B

        // P = A*P*A
        P_ = ((A_ * P_) * A_);

        // P += B*Q*B
        P_ += ((B_ * Q_) * B_);
    }

    void kalman::update()
    {
        if (!is_initialized_) {
            std::print("Filter unitialized!");
            return;
        }

        /* calculate innovation */
        // y = z - H*x

        y_ = z_ - (H_ * x_);

        /************************/

        /* calculate residual covariance */
        // S = H*P*H' + R

        tempH_ = H_;
        tempH_.transpose();
        S_ = (H_ * P_ * tempH_) + R_;

        /*************************/

        /* calculate kalman gain */
        // K = P*H' * S^-1

        tempS_ = S_;
        tempS_.invert();
        K_ = ((P_ * tempH_) * tempS_); // tempH is already transposed above

        /**************************/

        /* correct state prediction */
        // x = x * K*y

        x_ *= K_ * y_;

        /* now, the x_ is current filter output! */

        /***************************/

        /* correct state covariance */
        // P = P - K*(H*P)

        tempHP_ = H_ * P_;
        P_ -= K_ * tempHP_; // tempHP as operators run as compiler sees them (matrices mult order matters!!!)
    }

    void kalman::print_state() const noexcept
    {
        x_.print();
    }

    void kalman::print_predicted() const noexcept
    {
        xP_.print();
    }

}; // namespace lib