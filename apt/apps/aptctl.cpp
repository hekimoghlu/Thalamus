// aptctl - command-line driver for the Artificial Peripheral Thalamus device suite.
//
//   aptctl spec     [--out DIR]   the device specification: Modules E/O/M + latency budget + HIL
//   aptctl schedule [--budget H]  Module M optimal schedules and the contact-time threshold
//   aptctl hil      [--out DIR]   closed-loop latency sweep of the fixed-point controller
//   aptctl adapt    [--out DIR]   certify the latency-adaptive inner-loop schedule and compare
//   aptctl codegen  [--root DIR]  emit RTL/firmware headers from the C++ source of truth
//   aptctl vectors  [--out DIR] [--ticks N]  golden stimulus + expected telemetry for the RTL TB
//   aptctl decode   FILE          decode a captured UART telemetry stream to CSV on stdout
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "apt/adapt.hpp"
#include "apt/afferent.hpp"
#include "apt/codegen.hpp"
#include "apt/core_model.hpp"
#include "apt/dsp.hpp"
#include "apt/evidence.hpp"
#include "apt/io.hpp"
#include "apt/link.hpp"
#include "apt/numerics.hpp"
#include "apt/plant.hpp"
#include "apt/recovery.hpp"
#include "apt/regmap.hpp"
#include "apt/schedule.hpp"
#include "apt/stability.hpp"

using namespace apt;
namespace fs = std::filesystem;

namespace {

struct Args {
    std::string cmd;
    std::vector<std::string> pos;
    std::map<std::string, std::string> opt;
    std::string get(const std::string& k, const std::string& d) const {
        auto it = opt.find(k);
        return it == opt.end() ? d : it->second;
    }
};

Args parse(int argc, char** argv) {
    Args a;
    if (argc > 1) a.cmd = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string s = argv[i];
        if (s.rfind("--", 0) == 0 && i + 1 < argc) a.opt[s.substr(2)] = argv[++i];
        else a.pos.push_back(s);
    }
    return a;
}

void section(const char* title) {
    const std::size_t n = std::strlen(title);
    std::printf("\n== %s %s\n", title, std::string(n < 76 ? 76 - n : 2, '=').c_str());
}

// ------------------------------------------------------------------------------------- spec
int cmd_spec(const Args& a) {
    const fs::path out = a.get("out", "out/spec");
    fs::create_directories(out);
    const auto eb = EvidenceBase::load_default();

    section("Module E1 - afferent channel");
    const auto ch = AfferentChannel::from_evidence(eb);
    std::printf("  biological reference %.0f bit/s, device %.0f bit/s at %.0f Hz (not like-for-like)\n",
                ch.rate_biological(), ch.rate_device(), ch.bandwidth_hz);
    {
        io::CsvWriter w(out / "E_channel_rate.csv", {"latency_ms", "usable_bandwidth_fraction", "effective_rate_bit_s"});
        for (double L : {5, 10, 20, 30, 50, 80, 110, 150, 220, 300, 500})
            w.row(std::vector<double>{L, AfferentChannel::usable_fraction(L, ch.bandwidth_hz), ch.effective_rate(L)});
    }

    section("Module E3 - latency-to-learning law (exact two-point solution)");
    const auto law = LatencyLearningLaw::from_evidence(eb);
    std::printf("  a(tau) = %.4f / (1 + tau / %.1f ms)\n", law.a0(), law.tau_half_ms());

    section("Module E2 - admittance stability envelope and passivity");
    const std::vector<double> lat = {1, 2, 5, 10, 20, 30, 50, 80, 110, 150, 220, 300};
    const auto env = stability_envelope(lat, AdmittanceDesign{}, ch.bandwidth_hz);
    std::vector<std::vector<std::string>> rows;
    {
        io::CsvWriter w(out / "E_stability_envelope.csv", {"latency_ms", "max_stable_stiffness_N_per_m", "usable_bandwidth_fraction"});
        for (const auto& e : env) {
            w.row(std::vector<double>{e.latency_ms, e.max_stable_stiffness_N_m, e.usable_bandwidth_fraction});
            rows.push_back({io::fmt(e.latency_ms), io::fmt(e.max_stable_stiffness_N_m, 5), io::fmt(e.usable_bandwidth_fraction, 3)});
        }
    }
    std::cout << io::table({"latency ms", "max K N/m", "usable BW"}, rows);
    {
        io::CsvWriter w(out / "E_closure_latency.csv", {"mass_kg", "damping_Ns_m", "closure_latency_ms"});
        rows.clear();
        for (double m : {0.1, 0.25, 0.5, 1.0, 2.0})
            for (double b : {2.0, 6.0, 20.0}) {
                const double c = closure_latency_ms({m, b, 1e-3});
                w.row(std::vector<double>{m, b, c});
                rows.push_back({io::fmt(m), io::fmt(b), std::isinf(c) ? ">400" : io::fmt(c, 4)});
            }
        std::cout << "  closure latency is a design choice (rendered mass / damping):\n"
                  << io::table({"mass kg", "damping", "closes at ms"}, rows);
    }
    std::printf("  Colgate-Schenkel: 1000 N/m at 1 kHz with 6 N s/m needs b_phys > %.2f N s/m\n",
                PassivityCondition::required_physical_damping(1000, 6, 1e-3));

    section("Module O5 - sensing specification");
    const auto sens = sensing_specification(eb);
    rows.clear();
    for (const auto& c : sens.channels)
        rows.push_back({c.channel, io::fmt(c.biological_resolution, 4) + " " + c.unit, io::fmt(c.required_bits, 4),
                        c.independent ? "yes" : "no"});
    std::cout << io::table({"channel", "resolution", "bits", "independent"}, rows);
    std::printf("  %.1f-%.1f bits/sample, %.0f-%.0f bit/s at the %.1f Hz gating limit\n", sens.bits_independent(),
                sens.bits_all(), sens.rate_independent_bit_s(), sens.rate_all_bit_s(), sens.gating_limit_hz);

    section("Module D/E4-5 - design envelope (matched-dose latency contrast, explicit reference)");
    const DeviceDesignModel dm(eb, RecoveryParams::load_default(), num::kSeed + 5);
    std::printf("  dose exponent gamma = %.3f (322 reps/session -> +8 ARAT)\n", dm.gamma());
    const std::vector<double> lat2 = {5, 10, 20, 30, 50, 80, 110, 150, 220, 300};
    rows.clear();
    {
        io::CsvWriter w(out / "E_dose_for_mcid.csv", {"latency_ms", "dose_multiple", "reps_per_session", "exceeds_published_protocol"});
        for (const auto& d : dm.dose_for_mcid(lat2)) {
            w.row(std::vector<double>{d.latency_ms, d.dose_multiple, d.reps_per_session, double(d.exceeds_published_high_dose)});
            rows.push_back({io::fmt(d.latency_ms), io::fmt(d.dose_multiple, 3), io::fmt(d.reps_per_session, 4)});
        }
    }
    std::cout << io::table({"latency ms", "dose x usual for MDC", "reps/session"}, rows);
    {
        io::CsvWriter w(out / "E_latency_contrast_surface.csv", {"latency_ms", "reference_ms", "fm_difference", "smd"});
        for (double ref : {30.0, 60.0, 110.0, 150.0, 220.0, 300.0})
            for (double L : {10.0, 20.0, 30.0, 50.0, 80.0, 110.0, 150.0, 220.0}) {
                const auto c = dm.latency_contrast(L, ref, 4.0);
                w.row(std::vector<double>{L, ref, c.fm_difference, c.smd});
            }
    }
    std::printf("  10 ms contrast: SMD %.2f against a 30 ms reference, %.2f against 300 ms - the whole\n"
                "  surface is shipped (E_latency_contrast_surface.csv); no device-vs-therapy number exists.\n",
                dm.latency_contrast(10, 30, 4).smd, dm.latency_contrast(10, 300, 4).smd);

    section("Front end - latency budget and filter trade-off (Section 3.16)");
    const core::DesignPoint dp;
    const auto fe = dsp::butterworth_lowpass(2, dp.filter_fc_hz, dp.ctrl_hz);
    dsp::ElectronicsTiming et;
    for (const auto& [name, b] : {std::pair{"afferent (sense -> stimulus)", dsp::afferent_budget(fe, 30, et, 0.0)},
                                  std::pair{"efferent (sense -> motor force)", dsp::motor_budget(fe, 30, et, 1.0)}}) {
        std::printf("  %s\n", name);
        for (const auto& s : b.stages) std::printf("    %-62s %7.3f ms\n", s.name.c_str(), s.ms);
        std::printf("    %-62s %7.3f ms  (headroom to 10 ms: %.2f, to 30 ms imperceptible: %.2f)\n", "total",
                    b.total_ms(), b.headroom_ms(10), b.headroom_ms(eb.get("haptic_delay_imperceptible")));
    }
    {
        io::CsvWriter w(out / "O_filter_tradeoff.csv", {"order", "fc_hz", "group_delay_ms", "reject_150Hz_db", "pareto"});
        for (const auto& c : dsp::filter_tradeoff(dp.ctrl_hz, 30, 150))
            w.row(std::vector<double>{double(c.order), c.fc_hz, c.group_delay_ms, c.reject_artefact_db, double(c.pareto)});
    }

    section("Module M - schedule and the contact-time threshold");
    ScheduleModel sm(eb, RecoveryParams::load_default(), SensitiveWindow::fit(eb));
    sm.calibrate_scale(eb);
    const auto tu = sm.threshold_uncertainty(20000, 11);
    std::printf("  threshold %.2fx usual care (exists above usual care in %.0f%% of draws)\n", sm.threshold(), 100 * tu.p_exists);
    std::printf("  reached at %.0f min/week (usual-care rate) vs %.1f min/week (device-specified rate)\n",
                60 * sm.threshold_hours(sm.reps_usual()), 60 * sm.threshold_hours(sm.reps_device()));
    rows.clear();
    {
        io::CsvWriter w(out / "M_policy_outcomes.csv", {"budget_h", "delivery", "gain_optimal_fm", "gain_uniform_fm", "p_mcid", "weeks_treated"});
        for (double bud : {20.0, 60.0, 150.0})
            for (auto [lbl, reps] : {std::pair{"therapist", sm.reps_usual()}, std::pair{"device", sm.reps_device()}}) {
                const auto o = sm.evaluate(reps, bud, 2000, num::kSeed + 1);
                w.row({io::fmt(bud), lbl, io::fmt(o.gain_optimal), io::fmt(o.gain_uniform), io::fmt(o.prob_exceeds_mcid), io::fmt(o.weeks_treated)});
                rows.push_back({io::fmt(bud), lbl, io::fmt(o.gain_optimal, 3), io::fmt(o.gain_uniform, 3), io::fmt(o.prob_exceeds_mcid, 2)});
            }
    }
    std::cout << io::table({"budget h", "delivery", "optimal FM", "spread FM", "P(>=MCID)"}, rows);

    section("HIL - fixed-point controller on the reaching plant");
    rows.clear();
    {
        io::CsvWriter w(out / "HIL_latency_sweep.csv", {"latency_ms", "passivity_control", "tracking_rms_mm", "interaction_rms_N",
                                                         "min_energy_mJ", "pc_active_fraction", "faulted", "reps"});
        for (double L : {0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0})
            for (bool pc : {false, true}) {
                auto c = core::Config::from_design(dp);
                if (!pc) c.ctrl &= ~core::kCtrlPassivity;
                hil::PlantParams p;
                p.extra_latency_ms = L;
                const auto r = hil::run(c, p, hil::ReachingTask{}, 6.0, 7);
                w.row(std::vector<double>{L, double(pc), r.tracking_rms_mm, r.interaction_rms_N, r.min_energy_mJ,
                                          r.pc_active_fraction, double(r.faulted), double(r.reps)});
                rows.push_back({io::fmt(L), pc ? "on" : "off", io::fmt(r.interaction_rms_N, 3), io::fmt(r.min_energy_mJ, 4),
                                r.faulted ? "TRIPPED" : "ok"});
            }
    }
    std::cout << io::table({"latency ms", "PC", "F rms N", "min E mJ", "supervisor"}, rows);

    section("Provenance");
    std::printf("  verified parameters read: %zu (of %zu rows; %zu of %zu cells unverified and never used)\n",
                eb.ledger().size(), eb.size(), eb.unverified_cells(), eb.total_numeric_cells());
    {
        io::CsvWriter w(out / "provenance_ledger.csv", {"parameter", "value", "src", "table", "provenance"});
        for (const auto& n : eb.ledger()) {
            const auto& r = eb.row(n);
            w.row({n, io::fmt(*r.value, 8), r.src, r.table, r.provenance == Provenance::Primary ? "primary" : "secondary"});
        }
        io::CsvWriter s(out / "structural_constants.csv", {"name", "value", "unit", "rationale"});
        for (const auto& c : StructuralRegistry::instance().all()) {
            s.row({c.name, io::fmt(c.value, 8), c.unit, c.rationale});
            std::printf("  structural (unsourced): %-30s = %-8s %-8s %s\n", c.name.c_str(), io::fmt(c.value).c_str(), c.unit.c_str(), c.rationale.c_str());
        }
    }
    std::printf("\n  tables written to %s\n", out.string().c_str());
    return 0;
}

// ---------------------------------------------------------------------------------- schedule
int cmd_schedule(const Args& a) {
    const auto eb = EvidenceBase::load_default();
    ScheduleModel sm(eb, RecoveryParams::load_default(), SensitiveWindow::fit(eb));
    sm.calibrate_scale(eb);
    const double budget = std::stod(a.get("budget", "20"));
    std::printf("dose-response scale %.4g; threshold %.3fx; sigma/week %.4f\n", sm.phi_scale(), sm.threshold(), sm.sigma_week());
    for (auto [lbl, reps] : {std::pair{"therapist", sm.reps_usual()}, std::pair{"device", sm.reps_device()}}) {
        const auto o = sm.evaluate(reps, budget, 4000, num::kSeed + 1);
        std::printf("%-9s %5.0f h: optimal %.2f FM (P>=MCID %.2f, %.1f weeks treated, weeks %.0f-%.0f), spread evenly %.2f FM\n",
                    lbl, budget, o.gain_optimal, o.prob_exceeds_mcid, o.weeks_treated, o.median_first_week,
                    o.median_last_week, o.gain_uniform);
    }
    const double ac = sm.marginal_epoch_value(0, 4), sub = sm.marginal_epoch_value(9, 13), ch = sm.marginal_epoch_value(26, 30);
    std::printf("epoch ratios (marginal limit): acute %.2f : subacute 1.00 : chronic %.2f (published 0.76 : 1 : 0.35)\n",
                ac / sub, ch / sub);
    return 0;
}

// --------------------------------------------------------------------------------------- hil
int cmd_hil(const Args& a) {
    const fs::path out = a.get("out", "out/hil");
    fs::create_directories(out);
    const double L = std::stod(a.get("latency", "20"));
    auto c = core::Config::from_design(core::DesignPoint{});
    hil::PlantParams p;
    p.extra_latency_ms = L;
    std::vector<hil::Sample> tr;
    const auto r = hil::run(c, p, hil::ReachingTask{}, std::stod(a.get("seconds", "6")), 7, &tr);
    io::CsvWriter w(out / "trace.csv", {"t", "intent_mm", "hand_mm", "device_mm", "force_N", "x_cmd_mm", "energy_mJ", "duty", "pc", "spikes", "fault"});
    for (const auto& s : tr)
        w.row(std::vector<double>{s.t, s.intent_mm, s.x_hand_mm, s.x_dev_mm, s.force_true_N, fx::from_sig(s.out.x_cmd),
                                  fx::from_sig(s.out.energy), fx::from_sig(s.out.duty), double(s.out.pc_active),
                                  double(s.out.spikes), double(s.out.fault)});
    std::printf("latency %.0f ms: tracking %.2f mm rms, interaction %.2f N rms, min E %.2f mJ, PC %.1f%%, %s, %u reps, afferent %.1f Hz/ch (%.0f%% gated)\n",
                L, r.tracking_rms_mm, r.interaction_rms_N, r.min_energy_mJ, 100 * r.pc_active_fraction,
                r.faulted ? "SUPERVISOR TRIPPED" : "no fault", r.reps, r.afferent_rate_hz, 100 * r.gated_fraction);
    std::printf("trace: %s\n", (out / "trace.csv").string().c_str());
    return 0;
}

// ----------------------------------------------------------------------------------- adapt
int cmd_adapt(const Args& a) {
    const fs::path out = a.get("out", "out/adapt");
    fs::create_directories(out);
    const core::DesignPoint base{};
    const auto sched = adapt::certify_schedule(base);
    std::printf("certified inner-gain schedule (closed-loop HIL, 3 seeds, 1.25x delay / 1.5x gain margins)\n");
    std::printf("  delay<= ms   scale   analytic (40/tau)^2\n");
    const auto ana = adapt::analytic_schedule(sched.edge_ms);
    for (std::size_t i = 0; i < sched.edge_ms.size(); ++i)
        std::printf("  %9.0f   %5.3f   %5.3f\n", sched.edge_ms[i], sched.scale[i], ana.scale[i]);

    io::CsvWriter w(out / "fixed_vs_adaptive.csv",
                    {"delay_ms", "fixed_stable", "fixed_interaction_N", "adaptive_scale", "adaptive_stable",
                     "adaptive_interaction_N", "adaptive_cmd_follow_mm", "fixed_cmd_follow_mm", "adaptive_tracking_mm"});
    std::printf("\n  delay   fixed gains            adaptive gains (scale)     rendered-dynamics fidelity\n");
    for (double d : {0.0, 20.0, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0, 160.0, 200.0}) {
        const auto f = adapt::trial(base, 1.0, d, 7, 6.0);
        const double sc = sched.lookup(d);
        const auto g = adapt::trial(base, sc, d, 7, 6.0);
        std::printf("  %4.0f ms %-8s %5.1f N      %-8s x%.3f  %5.1f N       cmd-follow %.2f mm (fixed %.2f)\n", d,
                    f.stable ? "stable" : "UNSTABLE", f.summary.interaction_rms_N, g.stable ? "stable" : "UNSTABLE", sc,
                    g.summary.interaction_rms_N, g.summary.cmd_follow_rms_mm, f.summary.cmd_follow_rms_mm);
        w.row(std::vector<double>{d, double(f.stable), f.summary.interaction_rms_N, sc, double(g.stable),
                                  g.summary.interaction_rms_N, g.summary.cmd_follow_rms_mm, f.summary.cmd_follow_rms_mm,
                                  g.summary.tracking_rms_mm});
    }

    // dynamic scenario: the link degrades without warning (10 -> 120 ms at 4 s) and recovers (-> 30 ms at 12 s)
    hil::RunHooks hk;
    hk.latency_ms = [](double t) { return t < 4.0 ? 10.0 : (t < 12.0 ? 120.0 : 30.0); };
    for (int adaptive = 0; adaptive < 2; ++adaptive) {
        adapt::Supervisor sup(base, sched);
        hk.supervisor = adaptive ? std::function<void(double, double, core::CoreModel&)>(
                                       [&sup](double, double probe, core::CoreModel& m) { sup.on_probe(probe, m); })
                                 : std::function<void(double, double, core::CoreModel&)>{};
        hil::PlantParams p;
        std::vector<hil::Sample> tr;
        const auto r = hil::run_ex(core::Config::from_design(base), p, hil::ReachingTask{}, 18.0, 11, hk, &tr);
        std::printf("\ndelay step 10->120->30 ms, %s: %s, min E %.1f mJ, interaction %.2f N rms%s\n",
                    adaptive ? "adaptive" : "fixed   ", r.faulted ? "SUPERVISOR TRIPPED" : "no fault", r.min_energy_mJ,
                    r.interaction_rms_N, adaptive ? "" : "");
        if (adaptive) std::printf("  scale changes: %d, final scale %.3f, final estimate %.0f ms\n", sup.changes(), sup.scale(), sup.estimate_ms());
    }
    return 0;
}

// ----------------------------------------------------------------------------------- codegen
int cmd_codegen(const Args& a) {
    const auto eb = EvidenceBase::load_default();
    for (const auto& p : codegen::emit_all(a.get("root", "."), eb)) std::printf("wrote %s\n", p.string().c_str());
    return 0;
}

// ----------------------------------------------------------------------------------- vectors
// Closed-loop stimulus from the HIL plant (30 ms transport delay, so the passivity controller and
// its divider are exercised), with an e-stop and an SPI fault-clear injected. The RTL replays the
// sensor stream open-loop; because it is bit-exact it must reproduce every telemetry byte.
int cmd_vectors(const Args& a) {
    const fs::path out = a.get("out", "tb/vectors");
    const int ticks = std::stoi(a.get("ticks", "1200"));
    const int estop_from = ticks * 3 / 4, estop_to = estop_from + 5, clear_at = estop_to + 40;
    fs::create_directories(out);

    auto cfg = core::Config::from_design(core::DesignPoint{});
    cfg.wdog_ticks = 0;       // heartbeat is exercised by a directed test, not the replay
    cfg.lat_budget = 0;       // latency supervision likewise
    cfg.stim_width = 40;      // simulation-scale pulse timing (not part of the golden model)
    cfg.stim_gap = 10;
    cfg.stim_refract = 200;
    {
        std::ofstream f(out / "cfg.txt");
        for (const auto& [addr, v] : cfg.to_registers())
            if (addr != regmap::find("CTRL").addr) f << std::hex << int(addr) << ' ' << v << '\n';
        f << std::hex << int(regmap::find("CTRL").addr) << ' ' << (cfg.ctrl | core::kCtrlRun) << '\n';
    }
    core::CoreModel m(cfg);
    hil::PlantParams p;
    p.extra_latency_ms = 30;
    hil::Plant plant(p, 99);
    hil::ReachingTask task;
    std::ofstream st(out / "stim.txt"), ex(out / "expect.txt");
    double duty = 0;
    int pc_ticks = 0, spikes = 0, faults = 0;
    for (int t = 0; t < ticks; ++t) {
        auto in = plant.advance(1e-3, duty, task);
        in.heartbeat = false;
        in.estop = t >= estop_from && t < estop_to;
        in.fault_clear = t == clear_at;
        const auto o = m.step(in);
        duty = fx::from_sig(o.duty);
        pc_ticks += o.pc_active;
        spikes += std::popcount(o.spikes);
        faults += o.fault;
        char buf[128];
        for (auto c : in.adc) {
            std::snprintf(buf, sizeof buf, "%04X ", static_cast<unsigned>(static_cast<std::uint16_t>(c)));
            st << buf;
        }
        std::snprintf(buf, sizeof buf, "%08X %02X\n", static_cast<unsigned>(in.encoder),
                      unsigned(in.estop) | (unsigned(in.fault_clear) << 1));
        st << buf;
        for (auto b : o.telemetry.pack()) {
            std::snprintf(buf, sizeof buf, "%02X", b);
            ex << buf;
        }
        ex << '\n';
    }
    std::printf("wrote %d ticks to %s (PC active %d ticks, %d afferent events, %d fault ticks, e-stop %d-%d, clear %d)\n",
                ticks, out.string().c_str(), pc_ticks, spikes, faults, estop_from, estop_to, clear_at);
    return 0;
}

// ------------------------------------------------------------------------------------ decode
int cmd_decode(const Args& a) {
    if (a.pos.empty()) {
        std::fprintf(stderr, "usage: aptctl decode FILE\n");
        return 2;
    }
    std::ifstream f(a.pos[0], std::ios::binary);
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(f)), {});
    link::DecodeStats st;
    const auto frames = link::decode_stream(bytes, &st);
    std::printf("tick,status,force_N,x_mm,v_mm_s,x_cmd_mm,v_cmd_mm_s,energy_mJ,duty,spikes,reps,lat_motor,lat_afferent\n");
    for (const auto& t : frames)
        std::printf("%u,0x%04X,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,0x%02X,%u,%u,%u\n", t.tick, t.status, fx::from_sig(t.force),
                    fx::from_sig(t.x_meas), fx::from_sig(t.v_meas), fx::from_sig(t.x_cmd), fx::from_sig(t.v_cmd),
                    fx::from_sig(t.energy), fx::from_sig(t.duty), t.spikes, t.rep_count, t.lat_motor, t.lat_afferent);
    std::fprintf(stderr, "%zu frames, %zu CRC errors, %zu length errors\n", st.frames, st.crc_errors, st.length_errors);
    return st.crc_errors ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    const auto a = parse(argc, argv);
    try {
        if (a.cmd == "spec") return cmd_spec(a);
        if (a.cmd == "schedule") return cmd_schedule(a);
        if (a.cmd == "hil") return cmd_hil(a);
        if (a.cmd == "adapt") return cmd_adapt(a);
        if (a.cmd == "codegen") return cmd_codegen(a);
        if (a.cmd == "vectors") return cmd_vectors(a);
        if (a.cmd == "decode") return cmd_decode(a);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "aptctl: %s\n", e.what());
        return 1;
    }
    std::fprintf(stderr,
                 "usage: aptctl <spec|schedule|hil|codegen|vectors|decode> [options]\n"
                 "  spec     [--out DIR]                 device specification report + tables\n"
                 "  schedule [--budget HOURS]            Module M optimal schedule\n"
                 "  hil      [--latency MS] [--out DIR]  closed-loop run with trace\n"
                 "  codegen  [--root DIR]                emit RTL / firmware headers\n"
                 "  vectors  [--out DIR] [--ticks N]     golden vectors for the RTL testbench\n"
                 "  decode   FILE                        decode captured telemetry\n");
    return 2;
}
