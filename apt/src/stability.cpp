#include "apt/stability.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "apt/afferent.hpp"
#include "apt/numerics.hpp"

namespace apt {

bool delayed_loop_stable(double k, double tau_ms, const AdmittanceDesign& p, double tmax,
                         double kick) {
    // Operation order mirrors 05_closed_loop_control.admittance_stable() so that the integer
    // step counts (n, d) and every intermediate double are the same IEEE values.
    const double dt = p.dt_s, m = p.mass_kg, b = p.damping_Ns_m;
    const int n = static_cast<int>(tmax / dt);
    const int d = std::max(static_cast<int>((tau_ms / 1000.0) / dt), 1);
    std::vector<double> x(n, 0.0), v(n, 0.0);
    x[0] = kick;
    const double blow = 50.0 * std::abs(kick);
    for (int i = 1; i < n; ++i) {
        const double xd = i - d >= 0 ? x[i - d] : x[0];
        const double vd = i - d < 0 ? v[i - 1] : v[std::max(i - d, 0)];
        const double f = -k * xd - b * vd;
        const double a = f / m;
        v[i] = v[i - 1] + dt * a;
        x[i] = x[i - 1] + dt * v[i];
        if (!std::isfinite(x[i]) || std::abs(x[i]) > blow) return false;
    }
    const int third = n / 3;
    double late = 0.0, mid = 0.0;
    for (int i = n - third; i < n; ++i) late = std::max(late, std::abs(x[i]));
    for (int i = third; i < 2 * third; ++i) mid = std::max(mid, std::abs(x[i]));
    return late <= mid + 1e-9;
}

double max_stable_stiffness(double tau_ms, const AdmittanceDesign& d) {
    static const auto grid = num::geomspace(0.5, 20000.0, 80);
    double k_hi = 0.0;
    for (double k : grid) {
        if (!delayed_loop_stable(k, tau_ms, d)) break;
        k_hi = k;
    }
    return k_hi;
}

std::vector<EnvelopeRow> stability_envelope(const std::vector<double>& lat,
                                            const AdmittanceDesign& d, double bw) {
    std::vector<EnvelopeRow> out;
    out.reserve(lat.size());
    for (double tau : lat)
        out.push_back({tau, max_stable_stiffness(tau, d), AfferentChannel::usable_fraction(tau, bw)});
    return out;
}

double closure_latency_ms(const AdmittanceDesign& d, double tau_hi) {
    if (max_stable_stiffness(tau_hi, d) > 0) return std::numeric_limits<double>::infinity();
    double lo = 1.0, hi = tau_hi;
    if (max_stable_stiffness(lo, d) <= 0) return lo;
    // Delays are realised in whole samples, so bisect to one sample period.
    while (hi - lo > d.dt_s * 1000.0) {
        const double mid = 0.5 * (lo + hi);
        (max_stable_stiffness(mid, d) > 0 ? lo : hi) = mid;
    }
    return hi;
}

}  // namespace apt
