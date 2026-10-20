#include "apt/evidence.hpp"

#include <cstdlib>
#include <fstream>
#include <sstream>

#ifndef APT_SOURCE_DATA_DIR
#define APT_SOURCE_DATA_DIR "data"
#endif

namespace apt {

namespace {

constexpr std::string_view kUnverified = "UNVERIFIED";

std::optional<double> parse_cell(const std::string& s) {
    if (s.empty() || s == kUnverified || s == "NA") return std::nullopt;
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    if (end == s.c_str() || *end != '\0')
        throw std::runtime_error("non-numeric cell '" + s + "' in a numeric column");
    return v;
}

Provenance parse_provenance(const std::string& s) {
    if (s == "primary") return Provenance::Primary;
    if (s == "secondary") return Provenance::Secondary;
    return Provenance::Unverified;
}

}  // namespace

std::vector<std::vector<std::string>> read_csv(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + p.string());
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string text = ss.str();

    std::vector<std::vector<std::string>> rows;
    std::vector<std::string> row;
    std::string cell;
    bool quoted = false;
    for (std::size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        if (quoted) {
            if (c == '"') {
                if (i + 1 < text.size() && text[i + 1] == '"') {
                    cell.push_back('"');
                    ++i;
                } else {
                    quoted = false;
                }
            } else {
                cell.push_back(c);
            }
            continue;
        }
        switch (c) {
            case '"': quoted = true; break;
            case ',': row.push_back(std::move(cell)); cell.clear(); break;
            case '\r': break;
            case '\n':
                row.push_back(std::move(cell));
                cell.clear();
                if (!(row.size() == 1 && row[0].empty())) rows.push_back(std::move(row));
                row.clear();
                break;
            default: cell.push_back(c);
        }
    }
    if (!cell.empty() || !row.empty()) {
        row.push_back(std::move(cell));
        rows.push_back(std::move(row));
    }
    return rows;
}

EvidenceBase EvidenceBase::load(const std::filesystem::path& dir) {
    // (table, key column) - mirrors kcore.getp(); only the device-layer tables are vendored.
    static const std::pair<const char*, const char*> kTables[] = {
        {"sensorimotor_params", "param"},
        {"rehab_benchmarks", "benchmark"},
    };
    EvidenceBase eb;
    std::map<std::string, std::string> owner;
    for (const auto& [table, key] : kTables) {
        const auto rows = read_csv(dir / (std::string(table) + ".csv"));
        if (rows.empty()) throw std::runtime_error(std::string("empty table ") + table);
        const auto& hdr = rows.front();
        auto col = [&](std::string_view name) -> std::size_t {
            for (std::size_t i = 0; i < hdr.size(); ++i)
                if (hdr[i] == name) return i;
            throw std::runtime_error(std::string(table) + ": missing column " + std::string(name));
        };
        const auto ik = col(key), iv = col("value"), ilo = col("value_lo"), ihi = col("value_hi"),
                   iu = col("unit"), ip = col("population"), is = col("src"),
                   iver = col("verified");
        for (std::size_t r = 1; r < rows.size(); ++r) {
            const auto& f = rows[r];
            if (f.size() != hdr.size())
                throw std::runtime_error(std::string(table) + ": malformed row " +
                                         std::to_string(r));
            EvidenceRow e;
            e.name = f[ik];
            e.value = parse_cell(f[iv]);
            e.lo = parse_cell(f[ilo]);
            e.hi = parse_cell(f[ihi]);
            e.unit = f[iu];
            e.population = f[ip];
            e.src = f[is];
            e.table = table;
            e.provenance = parse_provenance(f[iver]);
            if (auto it = owner.find(e.name); it != owner.end())
                throw std::runtime_error("duplicate parameter name across verified tables: " +
                                         e.name + " (" + it->second + ", " + table + ")");
            owner[e.name] = table;
            eb.rows_.emplace(e.name, std::move(e));
        }
    }
    return eb;
}

std::filesystem::path EvidenceBase::default_data_dir() {
    if (const char* env = std::getenv("APT_DATA"); env && *env) return env;
    return APT_SOURCE_DATA_DIR;
}

EvidenceBase EvidenceBase::load_default() { return load(default_data_dir() / "verified"); }

const EvidenceRow& EvidenceBase::row(std::string_view name) const {
    auto it = rows_.find(name);
    if (it == rows_.end())
        throw MissingParameterError(std::string(name) + " is not in the verified parameter tables");
    return it->second;
}

bool EvidenceBase::has(std::string_view name) const { return rows_.find(name) != rows_.end(); }

double EvidenceBase::get(std::string_view name, Field f) const {
    const auto& r = row(name);
    const std::optional<double>& cell = f == Field::Value ? r.value : f == Field::Lo ? r.lo : r.hi;
    if (r.provenance == Provenance::Unverified || !cell) {
        static const char* kField[] = {"value", "value_lo", "value_hi"};
        throw UnverifiedError(std::string(name) + "." + kField[static_cast<int>(f)] +
                              " was never verified - it must not be used");
    }
    std::lock_guard lk(ledger_->mu);
    if (ledger_->seen.insert(std::string(name)).second) ledger_->order.emplace_back(name);
    return *cell;
}

std::string EvidenceBase::src(std::string_view name) const { return row(name).src; }

std::size_t EvidenceBase::unverified_cells() const {
    std::size_t n = 0;
    for (const auto& [_, r] : rows_) n += !r.value + !r.lo + !r.hi;
    return n;
}

std::vector<std::string> EvidenceBase::ledger() const {
    std::lock_guard lk(ledger_->mu);
    return ledger_->order;
}

StructuralRegistry& StructuralRegistry::instance() {
    static StructuralRegistry r;
    return r;
}

double StructuralRegistry::declare(std::string name, double value, std::string unit,
                                   std::string rationale) {
    std::lock_guard lk(mu_);
    for (const auto& s : items_)
        if (s.name == name) return value;  // same constant used from several places
    items_.push_back({std::move(name), value, std::move(unit), std::move(rationale)});
    return value;
}

std::vector<StructuralConstant> StructuralRegistry::all() const {
    std::lock_guard lk(mu_);
    return items_;
}

}  // namespace apt
