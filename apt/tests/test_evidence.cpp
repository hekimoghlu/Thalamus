// The verification policy (manuscript Section 2.3), enforced.
#include <filesystem>
#include <fstream>

#include "apt/evidence.hpp"
#include "check.hpp"

using namespace apt;
namespace fs = std::filesystem;

static const EvidenceBase& eb() {
    static const auto e = EvidenceBase::load_default();
    return e;
}

TEST("verified values load with provenance") {
    CHECK_NEAR(eb().get("adaptation_rate_60ms"), 0.128, 0);
    CHECK(eb().src("adaptation_rate_60ms") == "441");
    CHECK(eb().row("weber_fraction_force").provenance == Provenance::Secondary);
    CHECK_NEAR(eb().get("dose_arat_50_100pct", Field::Lo), -6.79, 0);
}

TEST("an UNVERIFIED cell raises rather than being imputed") {
    CHECK_THROWS(eb().get("haptic_delay_imperceptible", Field::Lo), UnverifiedError);
    CHECK_THROWS(eb().get("reps_usual_care", Field::Hi), UnverifiedError);
}

TEST("an absent parameter raises") {
    CHECK_THROWS(eb().get("feedback_latency_of_conventional_therapy"), MissingParameterError);
}

TEST("duplicate parameter names across tables refuse to load") {
    const fs::path dir = fs::temp_directory_path() / "apt_dup_test";
    fs::create_directories(dir);
    const std::string hdr_p = "param,value,value_lo,value_hi,unit,population,src,verified,use\n";
    const std::string hdr_b = "benchmark,value,value_lo,value_hi,unit,population,src,verified,use_as\n";
    std::ofstream(dir / "sensorimotor_params.csv") << hdr_p << "x,1,UNVERIFIED,UNVERIFIED,u,p,1,primary,m\n";
    std::ofstream(dir / "rehab_benchmarks.csv") << hdr_b << "x,2,UNVERIFIED,UNVERIFIED,u,p,2,primary,m\n";
    CHECK_THROWS(EvidenceBase::load(dir), std::runtime_error);
    fs::remove_all(dir);
}

TEST("quoted CSV fields with embedded commas parse") {
    const fs::path p = fs::temp_directory_path() / "apt_csv_test.csv";
    std::ofstream(p) << "a,b\n\"x, y\",\"he said \"\"hi\"\"\"\r\n";
    const auto rows = read_csv(p);
    CHECK(rows.size() == 2 && rows[1][0] == "x, y" && rows[1][1] == "he said \"hi\"");
    fs::remove(p);
}

TEST("the access ledger records what was read, in order") {
    const auto e = EvidenceBase::load_default();
    e.get("n20_latency");
    e.get("weber_fraction_force");
    e.get("n20_latency");
    const auto l = e.ledger();
    CHECK(l.size() == 2 && l[0] == "n20_latency" && l[1] == "weber_fraction_force");
}

TEST_MAIN()
