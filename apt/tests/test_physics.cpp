// Module E/O reproduction: afferent channel, latency-learning law, sensing specification and the
// admittance stability envelope, checked against the framework's published outputs.
#include "apt/afferent.hpp"
#include "apt/evidence.hpp"
#include "apt/io.hpp"
#include "apt/stability.hpp"
#include "check.hpp"

using namespace apt;

static const EvidenceBase& eb() {
    static const auto e = EvidenceBase::load_default();
    return e;
}

static std::map<std::string, double> ref_e() {
    return io::read_json_numbers(EvidenceBase::default_data_dir() / "reference" / "E_control_summary.json");
}

TEST("learning law is fixed exactly by the two verified adaptation rates") {
    const auto law = LatencyLearningLaw::from_evidence(eb());
    CHECK_NEAR(law.rate(60.0), eb().get("adaptation_rate_60ms"), 1e-12);
    CHECK_NEAR(law.rate(259.2), eb().get("adaptation_rate_259ms"), 1e-12);
    const auto r = ref_e();
    CHECK_REL(law.tau_half_ms(), r.at("learning_half_rate_latency_ms"), 1e-12);  // 152 ms
    CHECK_REL(law.a0(), r.at("learning_rate_at_zero_delay"), 1e-12);
    // the half-rate latency lies between imperceptible haptic delay and agency collapse
    CHECK(law.tau_half_ms() > eb().get("haptic_delay_imperceptible"));
    CHECK(law.tau_half_ms() < 200.0);
}

TEST("afferent channel rates reproduce Module E") {
    const auto ch = AfferentChannel::from_evidence(eb());
    const auto r = ref_e();
    CHECK_REL(ch.rate_biological(), r.at("afferent_rate_biological_bit_s"), 1e-12);
    CHECK_REL(ch.rate_device(), r.at("afferent_rate_device_bit_s"), 1e-12);
    // at 110 ms only ~15% of the movement bandwidth still receives feedback within one cycle
    CHECK_NEAR(AfferentChannel::usable_fraction(110, ch.bandwidth_hz), 0.1515, 1e-3);
    CHECK_NEAR(AfferentChannel::usable_fraction(10, ch.bandwidth_hz), 1.0, 0);
}

TEST("sensing specification: 15.0-19.4 bits/sample, 500-646 bit/s at the gating limit") {
    const auto s = sensing_specification(eb());
    const auto ref = read_csv(EvidenceBase::default_data_dir() / "reference" / "O_sensing_specification.csv");
    CHECK(s.channels.size() == ref.size() - 1);
    for (std::size_t i = 0; i < s.channels.size(); ++i)
        CHECK_REL(s.channels[i].required_bits, std::stod(ref[i + 1][4]), 1e-12);
    CHECK_NEAR(s.bits_independent(), 15.0, 0.05);
    CHECK_NEAR(s.bits_all(), 19.4, 0.05);
    CHECK_NEAR(s.rate_independent_bit_s(), 500, 1.0);
    CHECK_NEAR(s.rate_all_bit_s(), 646, 1.0);
}

TEST("stability envelope reproduces E_stability_envelope.csv") {
    const auto ref = read_csv(EvidenceBase::default_data_dir() / "reference" / "E_stability_envelope.csv");
    std::vector<double> lat;
    for (std::size_t i = 1; i < ref.size(); ++i) lat.push_back(std::stod(ref[i][0]));
    const auto env = stability_envelope(lat, AdmittanceDesign{}, 30.0);
    for (std::size_t i = 0; i < env.size(); ++i) {
        CHECK_REL(env[i].max_stable_stiffness_N_m + 1e-300, std::stod(ref[i + 1][1]) + 1e-300, 1e-9);
        CHECK_REL(env[i].usable_bandwidth_fraction, std::stod(ref[i + 1][2]), 1e-12);
    }
}

TEST("envelope closure is a design choice: it scales with rendered mass over damping") {
    // Delayed *damping* is what destabilises the loop, so closure moves with m/b. Across
    // physically reasonable renderings it runs from ~10 ms to beyond 300 ms (Section 4.6).
    const double c_def = closure_latency_ms(AdmittanceDesign{});
    CHECK(c_def > 50 && c_def <= 80);  // stable on the 50 ms grid point, closed by 80 ms
    CHECK(closure_latency_ms({1.0, 6.0, 1e-3}) > c_def);   // heavier rendering closes later
    CHECK(closure_latency_ms({0.25, 20.0, 1e-3}) < c_def); // more damping closes earlier
    CHECK(closure_latency_ms({0.1, 20.0, 1e-3}) <= 10.0);
    CHECK(closure_latency_ms({1.0, 6.0, 1e-3}) > 300.0);
}

TEST("Colgate-Schenkel: 1000 N/m at 1 kHz with 6 N s/m needs 6.5 N s/m of physical damping") {
    CHECK_NEAR(PassivityCondition::required_physical_damping(1000, 6, 1e-3), 6.5, 1e-12);
    CHECK_NEAR(PassivityCondition::max_passive_stiffness(6.5, 6, 1e-3), 1000, 1e-9);
}

TEST_MAIN()
