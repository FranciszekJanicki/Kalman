#ifndef KALMAN_H
#define KALMAN_H

#include <vector>

#include "matrix.hpp"
// #include "Eigen/Dense"


// unit type
typedef mtx::Matrix<float> mtx::Matrix;


enum class MatrixID {
    kMatrixA,
    kMatrixB,
    kMatrixP,
    kMatrixQ,
    kVectorX,
    kVectorU,
    kMatrixH,
    kMatrixR,
    kMatrixS,
    kMatrixK,
    kVectorZ,
    kVectorY
};

struct FilterModel {
    const int states_ {}; // number of filter units
    const int inputs_ {}; // number of filter inputs
    const mtx::Matrix A_ {}; // state transition matrix (numStates x numInputs)
    const mtx::Matrix x_ {}; // state vector (numStates x 1)
    const mtx::Matrix B_ {}; // input transition matrix (numStates x numInputs)
    const mtx::Matrix u_ {}; // input vector (numInputs x 1)
    const mtx::Matrix P_ {}; // state covariance matrix (numStates x numStates)
    const mtx::Matrix Q_ {}; // input covariance matrix (numInputs x numInputs)
};

struct MeasureModel {
    const int states_ {}; // number of filter units
    const int measurements_ {}; // number of meauserements performed 
    mtx::Matrix H_ {}; // measurement transformation matrix (measurements_ x states_)
    mtx::Matrix z_ {}; // measurement vector (numMeasures x  1)
    mtx::Matrix R_ {}; // process noise (measurement uncertainty) (measurements_ x measurements_)
    mtx::Matrix y_ {}; // innovation (measurements_ x 1)
    mtx::Matrix S_ {}; // residual covariance (measurements_ x measurements_)
    mtx::Matrix K_ {}; // kalman gain (states_ x measurements_)
};


class Kalman {
    public:
        Kalman(const FilterModel &filter, const MeasureModel &measure);

        inline Kalman() = default;
        inline Kalman(const Kalman &other) = default;
        inline Kalman(Kalman &&other) = default;
        inline ~Kalman() = default;

        inline Kalman &operator=(const Kalman &other) = default;
        inline Kalman &operator=(Kalman &&other) = default;

        inline const mtx::Matrix &getState() const;
        inline double getTime() const;
        inline void setInputs(const mtx::Matrix &inputs);
        inline void init();

    private:
        void update();
        void predict();        

        bool isInitialized_ {false};

        float stepTime_ {1.0f};
        float currentTime_ {0.0f};
        const float startTime_ {0.0f};

        // common model
        const int states_ {1}; // number of filter outputs

        // filter model
        const int inputs_ {1}; // number of filter inputs
        const mtx::Matrix A_ {}; // state transition matrix (states_ x inputs_)
        const mtx::Matrix x_ {}; // state vector (states_ x 1)
        const mtx::Matrix B_ {}; // input transition matrix (states_ x inputs_)
        const mtx::Matrix u_ {}; // input vector (inputs_ x 1)
        const mtx::Matrix P_ {}; // state covariance matrix (states_ x states_)
        const mtx::Matrix Q_ {}; // input covariance matrix (inputs_ x inputs_)
        const mtx::Matrix predX_ {}; // predicted state vector (states_ x 1)

        // measurement model
        const int measurements_ {1}; // number of meauserements performed 
        mtx::Matrix H_ {}; // measurement transformation matrix (measurements_ x states_)
        mtx::Matrix z_ {}; // measurement vector (numMeasures x  1)
        mtx::Matrix R_ {}; // process noise (measurement uncertainty) (measurements_ x measurements_)
        mtx::Matrix y_ {}; // innovation (measurements_ x 1)
        mtx::Matrix S_ {}; // residual covariance (measurements_ x measurements_)
        mtx::Matrix K_ {}; // kalman gain (states_ x measurements_)
};

                    
#endif // KALMAN_H