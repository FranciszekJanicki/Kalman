#ifndef KALMAN_H
#define KALMAN_H

#include <vector>

#include "../src/matrix.hpp"
// #include "Eigen/Dense"


// unit type


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
    const size_t states_ {}; // number of filter units
    const size_t inputs_ {}; // number of filter inputs
    const mtx::Matrix<float> A_ {}; // state transition matrix (numStates x numInputs)
    const mtx::Matrix<float> x_ {}; // state vector (numStates x 1)
    const mtx::Matrix<float> B_ {}; // input transition matrix (numStates x numInputs)
    const mtx::Matrix<float> u_ {}; // input vector (numInputs x 1)
    const mtx::Matrix<float> P_ {}; // state covariance matrix (numStates x numStates)
    const mtx::Matrix<float> Q_ {}; // input covariance matrix (numInputs x numInputs)
};

struct MeasureModel {
    const size_t states_ {}; // number of filter units
    const size_t measurements_ {}; // number of meauserements performed 
    mtx::Matrix<float> H_ {}; // measurement transformation matrix (measurements_ x states_)
    mtx::Matrix<float> z_ {}; // measurement vector (numMeasures x  1)
    mtx::Matrix<float> R_ {}; // process noise (measurement uncertainty) (measurements_ x measurements_)
    mtx::Matrix<float> y_ {}; // innovation (measurements_ x 1)
    mtx::Matrix<float> S_ {}; // residual covariance (measurements_ x measurements_)
    mtx::Matrix<float> K_ {}; // kalman gain (states_ x measurements_)
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

        inline const mtx::Matrix<float> &getState() const;
        inline double getTime() const;
        inline bool getIsInitialized() const;

        inline void setInputs(const mtx::Matrix<float> &inputs);


    private:
        void predict();     
        void update();
    

        bool isInitialized_ {false};

        float stepTime_ {1.0f};
        float currentTime_ {0.0f};
        const float startTime_ {0.0f};

        // common model
        const size_t states_ {1};                                   // number of filter outputs

        // filter model
        const size_t inputs_ {1};                                   // number of filter inputs
        mtx::Matrix<float> A_ {states_, inputs_};             // state transition matrix (states_ x inputs_)
        mtx::Matrix<float> B_ {states_, inputs_};             // input transition matrix (states_ x inputs_)
        mtx::Matrix<float> u_ {inputs_, 1};                         // input vector (inputs_ x 1)
        mtx::Matrix<float> P_ {states_, states_};             // state covariance matrix (states_ x states_)
        mtx::Matrix<float> Q_ {inputs_, inputs_};             // input covariance matrix (inputs_ x inputs_)
        mtx::Matrix<float> x_ {states_, 1};                         // state vector (states_ x 1)
        mtx::Matrix<float> pX_ {states_, 1};                        // predicted state vector (states_ x 1)
        
        // measurement model
        const size_t measurements_ {1};                             // number of measerements performed 
        mtx::Matrix<float> H_ {measurements_, states_};       // measurement transformation matrix (measurements_ x states_)
        mtx::Matrix<float> z_ {measurements_, 1};             // measurement vector (measurements_ x  1)
        mtx::Matrix<float> R_ {measurements_, measurements_}; // process noise (measurement uncertainty) (measurements_ x measurements_)
        mtx::Matrix<float> y_ {measurements_, 1};             // innovation vector (measurements_ x 1)
        mtx::Matrix<float> S_ {measurements_, measurements_}; // residual covariance (measurements_ x measurements_)
        mtx::Matrix<float> K_ {states_, measurements_};             // kalman gain (states_ x measurements_)
};

                    
#endif // KALMAN_H