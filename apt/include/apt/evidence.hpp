// SPDX-License-Identifier: see repository LICENSE
//
// apt/evidence.hpp - the verified evidence base, with the framework's provenance discipline.
//
// The manuscript's design rule (Section 2.3) is carried over unchanged: a value enters a
// computation only if it was verified against a public source. A cell written literally as
// UNVERIFIED raises rather than being imputed, and two tables defining the same parameter name
// refuse to load at all. Every successful read is recorded in an access ledger so that any
// artefact the suite emits (a specification, a header for the RTL, a firmware table) can carry
// the exact list of verified values it was built from.
//
// A third class of constant - structural assumptions with no source - is registered separately
// through `StructuralRegistry`, named where it is used, and reported next to the verified ledger.
#pragma once

#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace apt {

enum class Provenance { Primary, Secondary, Unverified };

enum class Field { Value, Lo, Hi };

struct EvidenceRow {
    std::string name;
    std::optional<double> value, lo, hi;
    std::string unit, population, src, table;
    Provenance provenance = Provenance::Unverified;
};

class UnverifiedError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class MissingParameterError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class EvidenceBase {
public:
    // Loads every verified table the device layer reads. Throws if a table is missing, if a
    // parameter name appears twice across tables, or if a row is malformed.
    static EvidenceBase load(const std::filesystem::path& verified_dir);

    // Default location: $APT_DATA/verified if set, else the source-tree data directory.
    static EvidenceBase load_default();
    static std::filesystem::path default_data_dir();

    // Fetch a verified value. Throws MissingParameterError if absent and UnverifiedError if the
    // requested cell exists but was never verified.
    double get(std::string_view name, Field f = Field::Value) const;
    const EvidenceRow& row(std::string_view name) const;
    bool has(std::string_view name) const;
    std::string src(std::string_view name) const;

    std::size_t size() const { return rows_.size(); }
    std::size_t unverified_cells() const;
    std::size_t total_numeric_cells() const { return 3 * rows_.size(); }

    // Names of every parameter read through get() since load, in first-access order.
    std::vector<std::string> ledger() const;

private:
    struct Ledger {
        std::mutex mu;
        std::vector<std::string> order;
        std::set<std::string, std::less<>> seen;
    };
    std::map<std::string, EvidenceRow, std::less<>> rows_;
    std::shared_ptr<Ledger> ledger_ = std::make_shared<Ledger>();
};

// Structural constants: unsourced design choices. They are not hidden defaults - each one is
// registered with its rationale, and the specification report lists them next to the ledger.
struct StructuralConstant {
    std::string name;
    double value;
    std::string unit;
    std::string rationale;
};

class StructuralRegistry {
public:
    static StructuralRegistry& instance();
    double declare(std::string name, double value, std::string unit, std::string rationale);
    std::vector<StructuralConstant> all() const;

private:
    mutable std::mutex mu_;
    std::vector<StructuralConstant> items_;
};

// Shorthand used throughout the suite: STRUCTURAL("m_virtual", 0.25, "kg", "why ...").
inline double STRUCTURAL(std::string name, double value, std::string unit, std::string why) {
    return StructuralRegistry::instance().declare(std::move(name), value, std::move(unit),
                                                  std::move(why));
}

// RFC-4180 CSV reader (quoted fields, embedded commas, CRLF tolerant).
std::vector<std::vector<std::string>> read_csv(const std::filesystem::path& p);

}  // namespace apt
