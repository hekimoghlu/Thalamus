// closed_loop_model.cpp
// Build: g++ -O2 -std=c++17 closed_loop_model.cpp -o closed_loop_model

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

// Fraction of zero-delay learning rate retained at loop latency tau_ms.
// a(tau) = a0 / (1 + tau / tau_half), with tau_half = 152 ms from the manuscript.
double retained(double tau_ms, double tau_half_ms = 152.0) {
    return 1.0 / (1.0 + tau_ms / tau_half_ms);
}

struct Params {
    double dt_ms = 1.0;   // integration step
    double m     = 0.25;  // rendered virtual mass, kg (design choice, not sourced)
    double b     = 6.0;   // rendered virtual damping, N s/m (design choice)
    double T_ms  = 5000;  // simulation length
};

// Delayed loop: m x'' = -K x(t - L) - b x'.
// The device renders stiffness from a position that is L ms old.
// "Stable" here means the envelope decays: late peak < 0.5 * early peak.
bool stable(double K, double L_ms, const Params& p) {
    const double dt = p.dt_ms / 1000.0;
    const int d = (int)std::lround(L_ms / p.dt_ms);
    const int N = (int)std::lround(p.T_ms / p.dt_ms);

    std::vector<double> xs(N, 0.0);
    double x = 1e-3, v = 0.0;               // small initial displacement, metres

    for (int n = 0; n < N; ++n) {
        xs[n] = x;
        double xd = (n - d >= 0) ? xs[n - d] : 0.0;   // stale position
        double f  = -K * xd - p.b * v;
        v += (f / p.m) * dt;                           // semi-implicit Euler
        x += v * dt;
        if (!std::isfinite(x) || std::fabs(x) > 1e3) return false;
    }

    auto peak = [&](int a, int z) {
        double m = 0.0;
        for (int i = a; i < z; ++i) m = std::max(m, std::fabs(xs[i]));
        return m;
    };
    double early = peak(0, N / 5);
    double late  = peak(4 * N / 5, N);
    return late < 0.5 * early;
}

// Bisection on log K for the largest stable stiffness at a given latency.
double max_stable_K(double L_ms, const Params& p) {
    double lo = 1.0, hi = 1e5;
    if (!stable(lo, L_ms, p)) return 0.0;
    if (stable(hi, L_ms, p))  return hi;
    for (int i = 0; i < 40; ++i) {
        double mid = std::sqrt(lo * hi);
        if (stable(mid, L_ms, p)) lo = mid; else hi = mid;
    }
    return lo;
}

int main() {
    std::printf("Retained learning rate (tau_half = 152 ms)\n");
    for (double L : {1.0, 10.0, 30.0, 110.0, 220.0})
        std::printf("  %6.1f ms  %5.1f%%\n", L, 100.0 * retained(L));

    Params p;
    std::printf("\nMax stable virtual stiffness (m = %.2f kg, b = %.1f N s/m)\n", p.m, p.b);
    for (double L : {1.0, 10.0, 20.0, 50.0, 80.0, 110.0})
        std::printf("  %6.1f ms  %10.1f N/m\n", L, max_stable_K(L, p));
    return 0;
}
