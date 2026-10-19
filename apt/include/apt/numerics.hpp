// apt/numerics.hpp - the small numerical toolkit the suite needs and nothing more: a seeded
// normal generator, Nelder-Mead, bisection, probabilists' Gauss-Hermite quadrature, numpy-
// compatible grids, and the few statistics the reports quote.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <numeric>
#include <random>
#include <span>
#include <utility>
#include <vector>

namespace apt::num {

inline constexpr std::uint64_t kSeed = 20260912;  // the framework's SEED (kcore.py)

class Rng {
public:
    explicit Rng(std::uint64_t seed) : eng_(seed) {}
    double normal(double mu = 0.0, double sd = 1.0) { return mu + sd * nd_(eng_); }
    double uniform(double lo = 0.0, double hi = 1.0) {
        return lo + (hi - lo) * std::uniform_real_distribution<double>(0.0, 1.0)(eng_);
    }
    std::vector<double> normals(std::size_t n, double mu = 0.0, double sd = 1.0) {
        std::vector<double> v(n);
        for (auto& x : v) x = normal(mu, sd);
        return v;
    }
    std::mt19937_64& engine() { return eng_; }

private:
    std::mt19937_64 eng_;
    std::normal_distribution<double> nd_{0.0, 1.0};
};

inline double mean(std::span<const double> v) {
    return v.empty() ? NAN : std::accumulate(v.begin(), v.end(), 0.0) / double(v.size());
}

inline double sd(std::span<const double> v) {
    const double m = mean(v);
    double s = 0.0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / double(v.size() - 1));
}

inline double quantile(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    const double pos = q * double(v.size() - 1);
    const auto i = static_cast<std::size_t>(pos);
    const double f = pos - double(i);
    return i + 1 < v.size() ? v[i] * (1 - f) + v[i + 1] * f : v.back();
}

// numpy.linspace (endpoint=True), reproduced operation-for-operation.
inline std::vector<double> linspace(double a, double b, std::size_t n) {
    std::vector<double> y(n);
    if (n == 1) return {a};
    const double step = (b - a) / double(n - 1);
    for (std::size_t i = 0; i < n; ++i) y[i] = double(i) * step + a;
    y.back() = b;
    return y;
}

// numpy.geomspace for positive endpoints: logspace over log10 with exact endpoints.
inline std::vector<double> geomspace(double a, double b, std::size_t n) {
    auto y = linspace(std::log10(a), std::log10(b), n);
    for (auto& v : y) v = std::pow(10.0, v);
    y.front() = a;
    y.back() = b;
    return y;
}

// Bisection on a monotone-increasing f for f(x) = target. `geometric` bisects in log space.
inline double bisect(const std::function<double(double)>& f, double target, double lo, double hi,
                     int iters, bool geometric = false) {
    for (int i = 0; i < iters; ++i) {
        const double mid = geometric ? std::sqrt(lo * hi) : 0.5 * (lo + hi);
        (f(mid) > target ? hi : lo) = mid;
    }
    return geometric ? std::sqrt(lo * hi) : 0.5 * (lo + hi);
}

struct NMResult {
    std::vector<double> x;
    double fx;
    int iterations;
};

// Nelder-Mead with the standard coefficients and scipy's initial simplex (5% perturbation,
// 0.00025 for zero coordinates) so fits match scipy.optimize.minimize to optimiser tolerance.
inline NMResult nelder_mead(const std::function<double(const std::vector<double>&)>& f,
                            std::vector<double> x0, int max_iter = 4000, double xatol = 1e-6,
                            double fatol = 1e-9) {
    const std::size_t n = x0.size();
    std::vector<std::vector<double>> s(n + 1, x0);
    for (std::size_t i = 0; i < n; ++i) s[i + 1][i] = x0[i] != 0 ? 1.05 * x0[i] : 0.00025;
    std::vector<double> fs(n + 1);
    for (std::size_t i = 0; i <= n; ++i) fs[i] = f(s[i]);
    int it = 0;
    std::vector<std::size_t> idx(n + 1);
    for (; it < max_iter; ++it) {
        std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(idx.begin(), idx.end(), [&](auto a, auto b) { return fs[a] < fs[b]; });
        {
            auto s2 = s;
            auto f2 = fs;
            for (std::size_t i = 0; i <= n; ++i) s[i] = s2[idx[i]], fs[i] = f2[idx[i]];
        }
        double xspread = 0, fspread = 0;
        for (std::size_t i = 1; i <= n; ++i) {
            fspread = std::max(fspread, std::abs(fs[i] - fs[0]));
            for (std::size_t j = 0; j < n; ++j)
                xspread = std::max(xspread, std::abs(s[i][j] - s[0][j]));
        }
        if (xspread <= xatol && fspread <= fatol) break;
        std::vector<double> c(n, 0.0);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) c[j] += s[i][j] / double(n);
        auto lerp = [&](double t) {
            std::vector<double> p(n);
            for (std::size_t j = 0; j < n; ++j) p[j] = c[j] + t * (s[n][j] - c[j]);
            return p;
        };
        auto xr = lerp(-1.0);
        const double fr = f(xr);
        if (fr < fs[0]) {
            auto xe = lerp(-2.0);
            const double fe = f(xe);
            if (fe < fr) s[n] = xe, fs[n] = fe;
            else s[n] = xr, fs[n] = fr;
        } else if (fr < fs[n - 1]) {
            s[n] = xr, fs[n] = fr;
        } else {
            const bool outside = fr < fs[n];
            auto xc = lerp(outside ? -0.5 : 0.5);
            const double fc = f(xc);
            if (outside ? fc <= fr : fc < fs[n]) {
                s[n] = xc, fs[n] = fc;
            } else {
                for (std::size_t i = 1; i <= n; ++i) {
                    for (std::size_t j = 0; j < n; ++j) s[i][j] = s[0][j] + 0.5 * (s[i][j] - s[0][j]);
                    fs[i] = f(s[i]);
                }
            }
        }
    }
    const auto best = std::min_element(fs.begin(), fs.end()) - fs.begin();
    return {s[best], fs[best], it};
}

// Probabilists' Gauss-Hermite rule (numpy.polynomial.hermite_e.hermegauss), weights normalised
// to sum to one so the rule is an expectation over a standard normal.
struct Quadrature {
    std::vector<double> nodes, weights;
};

inline Quadrature gauss_hermite_e(int n) {
    // Golub-Welsch: the nodes are the eigenvalues of the symmetric tridiagonal Jacobi matrix of
    // the probabilists' Hermite recurrence (zero diagonal, off-diagonal sqrt(k)). Each eigenvalue
    // is isolated by Sturm-sequence bisection, which cannot converge onto a duplicate root.
    // Weights are 1 / (n He_{n-1}(x))^2 up to a constant, then normalised to sum to one.
    auto count_below = [n](double x) {  // number of eigenvalues < x
        int c = 0;
        double d = -x;
        if (d < 0) ++c;
        for (int k = 1; k < n; ++k) {
            const double off2 = double(k);
            d = -x - off2 / (d != 0.0 ? d : 1e-300);
            if (d < 0) ++c;
        }
        return c;
    };
    const double bound = 2.0 * std::sqrt(double(n)) + 1.0;  // Gershgorin
    Quadrature q;
    for (int i = 0; i < n; ++i) {
        double lo = -bound, hi = bound;
        for (int it = 0; it < 200 && hi - lo > 1e-15 * std::max(1.0, std::abs(lo)); ++it) {
            const double mid = 0.5 * (lo + hi);
            (count_below(mid) > i ? hi : lo) = mid;
        }
        q.nodes.push_back(0.5 * (lo + hi));
    }
    double total = 0.0;
    for (double x : q.nodes) {
        double p0 = 1.0, p1 = x;  // He_{n-1}(x) by the three-term recurrence
        if (n == 1) p1 = 1.0;
        for (int k = 1; k < n - 1; ++k) {
            const double p2 = x * p1 - double(k) * p0;
            p0 = p1;
            p1 = p2;
        }
        const double w = 1.0 / (double(n) * p1 * p1);
        q.weights.push_back(w);
        total += w;
    }
    for (auto& w : q.weights) w /= total;
    return q;
}

// numpy.searchsorted(a, v, side='left')
inline std::ptrdiff_t searchsorted(std::span<const double> a, double v) {
    return std::lower_bound(a.begin(), a.end(), v) - a.begin();
}

inline std::ptrdiff_t clip_index(std::ptrdiff_t i, std::ptrdiff_t lo, std::ptrdiff_t hi) {
    return std::clamp(i, lo, hi);
}

}  // namespace apt::num
