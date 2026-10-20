// tests/check.hpp - a dependency-free test harness: CHECK / CHECK_NEAR / CHECK_REL / CHECK_THROWS,
// one process per suite, non-zero exit on any failure.
#pragma once

#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace check {

inline int& failures() {
    static int f = 0;
    return f;
}
inline int& passes() {
    static int p = 0;
    return p;
}

struct Case {
    const char* name;
    std::function<void()> fn;
};
inline std::vector<Case>& registry() {
    static std::vector<Case> r;
    return r;
}
struct Reg {
    Reg(const char* n, std::function<void()> f) { registry().push_back({n, std::move(f)}); }
};

inline void report(bool ok, const char* file, int line, const std::string& what) {
    if (ok) {
        ++passes();
    } else {
        ++failures();
        std::fprintf(stderr, "  FAIL %s:%d  %s\n", file, line, what.c_str());
    }
}

inline int run_all() {
    for (auto& c : registry()) {
        const int before = failures();
        try {
            c.fn();
        } catch (const std::exception& e) {
            ++failures();
            std::fprintf(stderr, "  FAIL %s: unexpected exception: %s\n", c.name, e.what());
        }
        std::printf("[%s] %s\n", failures() == before ? " ok " : "FAIL", c.name);
    }
    std::printf("%d checks passed, %d failed\n", passes(), failures());
    return failures() ? 1 : 0;
}

}  // namespace check

#define CHECK_CAT2(a, b) a##b
#define CHECK_CAT(a, b) CHECK_CAT2(a, b)
#define TEST(name)                                                              \
    static void CHECK_CAT(test_fn_, __LINE__)();                                \
    static check::Reg CHECK_CAT(test_reg_, __LINE__)(name, CHECK_CAT(test_fn_, __LINE__)); \
    static void CHECK_CAT(test_fn_, __LINE__)()

#define CHECK(cond) check::report(static_cast<bool>(cond), __FILE__, __LINE__, #cond)

#define CHECK_NEAR(a, b, tol)                                                                  \
    do {                                                                                       \
        const double _a = (a), _b = (b);                                                       \
        check::report(std::abs(_a - _b) <= (tol), __FILE__, __LINE__,                          \
                      std::string(#a " ~ " #b ": ") + std::to_string(_a) + " vs " +            \
                          std::to_string(_b));                                                 \
    } while (0)

#define CHECK_REL(a, b, rtol)                                                                  \
    do {                                                                                       \
        const double _a = (a), _b = (b);                                                       \
        check::report(std::abs(_a - _b) <= (rtol) * std::abs(_b), __FILE__, __LINE__,         \
                      std::string(#a " ~ " #b " (rel): ") + std::to_string(_a) + " vs " +      \
                          std::to_string(_b));                                                 \
    } while (0)

#define CHECK_THROWS(expr, Exc)                                                                \
    do {                                                                                       \
        bool _thrown = false;                                                                  \
        try {                                                                                  \
            (void)(expr);                                                                      \
        } catch (const Exc&) {                                                                 \
            _thrown = true;                                                                    \
        }                                                                                      \
        check::report(_thrown, __FILE__, __LINE__, "expected " #Exc " from " #expr);           \
    } while (0)

#define TEST_MAIN() \
    int main() { return check::run_all(); }
