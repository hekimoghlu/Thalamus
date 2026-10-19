// Golden-model semantics: the behaviours the RTL is required to reproduce.
#include "apt/core_model.hpp"
#include "apt/regmap.hpp"
#include "check.hpp"

using namespace apt;
using namespace apt::core;

static Config cfg() { return Config::from_design(DesignPoint{}); }

static Inputs rest(std::int32_t enc = 6000) {  // 30 mm, no force
    Inputs in;
    in.encoder = enc;
    in.heartbeat = true;
    return in;
}

TEST("bumpless start: no velocity transient, admittance starts at the measured position") {
    CoreModel m(cfg());
    const auto o = m.step(rest());
    CHECK(o.v_meas == 0);
    CHECK_NEAR(fx::from_sig(o.x_meas), 30.0, 1e-4);  // Q8.24 quantisation of 5 um/count
    CHECK(o.active && !o.fault);
    // the only motion is the virtual spring toward X_EQ (40 mm): small, positive
    CHECK(o.v_cmd > 0 && o.v_cmd < fx::to_sig(1.0));
}

TEST("constant interaction force accelerates the virtual mass at F/m") {
    auto c = cfg();
    c.adm_ck = 0;  // no spring
    c.adm_cb = fx::kCoefOne;  // no damping
    CoreModel m(c);
    auto in = rest();
    in.adc[kInteractionForce] = 3277;  // ~10 N at 100 N full scale
    fx::sig_t v_prev = 0;
    for (int i = 0; i < 100; ++i) {  // let the 80 Hz filter settle, stay below saturation
        const auto o = m.step(in);
        if (i == 99) {
            const double dv = fx::from_sig(o.v_cmd - v_prev);  // mm/s per tick
            CHECK_NEAR(dv, 1000.0 * fx::from_sig(o.force) / 1.0 * 1e-3, 1e-3);
        }
        v_prev = o.v_cmd;
    }
}

TEST("safety: force limit latches, clears only when the cause is gone") {
    CoreModel m(cfg());
    m.step(rest());
    auto in = rest();
    in.adc[kInteractionForce] = 32000;  // ~98 N > 80 N limit, after filter settling
    Outputs o;
    for (int i = 0; i < 50; ++i) o = m.step(in);
    CHECK(o.fault && (o.latched & kFaultForce) && o.duty == 0 && !o.active);
    auto clr = rest();
    clr.fault_clear = true;
    clr.adc[kInteractionForce] = 32000;
    o = m.step(clr);
    CHECK(o.fault);  // cause still present
    for (int i = 0; i < 50; ++i) o = m.step(rest());  // force removed; filter decays
    clr = rest();
    clr.fault_clear = true;
    o = m.step(clr);
    CHECK(!o.fault && o.active);
}

TEST("estop and watchdog") {
    CoreModel m(cfg());
    auto in = rest();
    in.estop = true;
    CHECK(m.step(in).latched & kFaultEstop);
    CoreModel w(cfg());
    auto quiet = rest();
    quiet.heartbeat = false;
    Outputs o;
    for (std::uint32_t i = 0; i < w.config().wdog_ticks; ++i) o = w.step(quiet);
    CHECK(o.latched & kFaultWatchdog);
}

TEST("passivity controller dissipates exactly the observed deficit (unsaturated)") {
    auto c = cfg();
    c.pc_alpha_max = fx::kSigMax;
    CoreModel m(c);
    m.step(rest(6000));
    // drive a force opposing motion so the observer goes negative
    Inputs in = rest(6000);
    bool saw = false;
    for (int i = 1; i < 300 && !saw; ++i) {
        in.encoder = 6000 + i * 20;          // +0.1 mm/tick = 100 mm/s
        in.adc[kInteractionForce] = -3277;   // -10 N
        const auto o = m.step(in);
        if (o.pc_active) {
            saw = true;
            CHECK(std::abs(fx::from_sig(o.energy)) < 1e-3);
            CHECK(o.f_pc < 0);  // extra damping opposes the (positive) velocity
        }
    }
    CHECK(saw);
}

TEST("duty is slew-limited and clamped") {
    CoreModel m(cfg());
    m.step(rest());
    auto in = rest();
    in.adc[kInteractionForce] = 8000;
    fx::sig_t prev = 0;
    for (int i = 0; i < 200; ++i) {
        const auto o = m.step(in);
        CHECK(std::abs(fx::from_sig(o.duty - prev)) <= fx::from_sig(m.config().slew) + 1e-6);
        CHECK(std::abs(o.duty) <= fx::kSigOne);
        prev = o.duty;
    }
}

TEST("afferent relay: events, gating flags, telemetry nibbles") {
    CoreModel m(cfg());
    auto in = rest();
    in.adc[kGripForce] = 8192;  // 100 N grip: fast train, inside the gating interval
    int spikes = 0, gated = 0;
    for (int i = 0; i < 2000; ++i) {
        const auto o = m.step(in);
        spikes += o.spikes & 1;
        gated += o.gated & 1;
        CHECK((o.telemetry.spikes & 0x0F) == o.spikes);
        CHECK((o.telemetry.spikes >> 4) == o.gated);
        if (o.spikes & 1) CHECK(o.stim_amp[0] == ((o.gated & 1) ? (0x40 * 102) >> 8 : 0x40));
    }
    CHECK(spikes > 50);
    CHECK(gated > 0);
}

TEST("repetition detector counts reach-and-return") {
    CoreModel m(cfg());
    for (int rep = 0; rep < 3; ++rep) {
        for (int i = 0; i <= 100; ++i) m.step(rest(2000 + i * 160));   // 10 -> 90 mm
        for (int i = 100; i >= 0; --i) m.step(rest(2000 + i * 160));
    }
    auto o = m.step(rest(2000));
    CHECK(o.rep_count == 3);
}

TEST("register image covers every RW register exactly once") {
    const auto regs = cfg().to_registers();
    std::size_t rw = 0;
    for (const auto& r : regmap::kRegs) rw += r.access == regmap::Access::RW;
    CHECK(regs.size() == rw);
}

TEST_MAIN()
