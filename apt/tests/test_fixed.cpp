// The fixed-point contract: rounding, saturation and the four multiply shapes.
#include "apt/fixed.hpp"
#include "check.hpp"

using namespace apt::fx;

TEST("round-half-up then arithmetic shift, symmetric cases") {
    CHECK(round_shift(3, 1) == 2);    // 1.5 -> 2
    CHECK(round_shift(-3, 1) == -1);  // -1.5 -> -1 (half-up, not half-away)
    CHECK(round_shift(5, 2) == 1);    // 1.25 -> 1
    CHECK(round_shift(-5, 2) == -1);  // -1.25 -> -1
    CHECK(round_shift(-6, 2) == -1);  // -1.5 -> -1
    CHECK(round_shift(-7, 2) == -2);  // -1.75 -> -2
}

TEST("saturation at both rails") {
    CHECK(add(kSigMax, 1) == kSigMax);
    CHECK(sub(kSigMin, 1) == kSigMin);
    CHECK(neg(kSigMin) == kSigMax);
    CHECK(apt::fx::abs(kSigMin) == kSigMax);
    CHECK(mul_ss(kSigMax, kSigMax) == kSigMax);
    CHECK(mul_ss(kSigMax, kSigMin) == kSigMin);
}

TEST("multiplies agree with real arithmetic to half an LSB") {
    const double vals[] = {-123.456, -1.0, -0.001, 0.0, 0.5, 3.14159, 777.25};
    const double coefs[] = {-1.9, -0.25, 1e-3, 0.976, 4.0, 80.0};
    for (double a : vals) {
        for (double c : coefs) {
            const double want = from_sig(to_sig(a)) * from_coef(to_coef(c));
            if (std::abs(want) < 2000) CHECK_NEAR(from_sig(mul_sc(to_sig(a), to_coef(c))), want, 0.5 / kSigOne + 1e-15);
        }
        for (double b : vals) {
            const double want = from_sig(to_sig(a)) * from_sig(to_sig(b));
            if (std::abs(want) < 2000) CHECK_NEAR(from_sig(mul_ss(to_sig(a), to_sig(b))), want, 0.5 / kSigOne + 1e-15);
        }
    }
}

TEST("ADC code and encoder scaling") {
    CHECK(mul_code(16384, to_sig(100.0)) == to_sig(50.0));    // half scale of 100 N
    CHECK(mul_code(-32768, to_sig(100.0)) == to_sig(-100.0));
    CHECK_NEAR(from_sig(mul_ic(2000, to_coef(0.005))), 10.0, 1e-5);  // 2000 counts * 5 um
    CHECK_NEAR(from_sig(mul_ic(-7, to_coef(0.005))), -0.035, 1e-5);
}

TEST("conversion range checks") {
    CHECK_THROWS(to_sig(5000.0), std::range_error);
    CHECK_THROWS(to_coef(200.0), std::range_error);
    CHECK(to_coef(1.0) == kCoefOne);
}

TEST_MAIN()
