#include "apt/neuromorphic.hpp"

#include <cmath>

namespace apt::neuro {

double lif_rate(double leak, double gain, double theta, double u) {
    if (u <= 0 || gain <= 0 || theta <= 0) return 0.0;
    if (leak <= 0) return gain * u / theta;
    const double v_inf = gain * u / leak;
    if (v_inf <= theta) return 0.0;
    return leak / -std::log(1.0 - theta / v_inf);
}

double ungated_fraction(double rate_hz, double isi_ms) {
    if (rate_hz <= 0) return 1.0;
    return 1000.0 / rate_hz >= isi_ms ? 1.0 : 0.0;
}

}  // namespace apt::neuro
