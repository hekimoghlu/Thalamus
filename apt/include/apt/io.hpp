// apt/io.hpp - minimal I/O: a flat JSON number reader (for the framework's result summaries),
// a CSV writer, and a fixed-width console table.
#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace apt::io {

// Parses a JSON document and returns every numeric leaf keyed by its dotted path; array
// elements are keyed by index ("prr_slope_range.0"). Strings, booleans and null are skipped.
std::map<std::string, double> read_json_numbers(const std::filesystem::path& p);

class CsvWriter {
public:
    CsvWriter(const std::filesystem::path& p, std::vector<std::string> header);
    ~CsvWriter();
    CsvWriter& row(const std::vector<std::string>& cells);
    CsvWriter& row(const std::vector<double>& cells);
    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
    struct Impl;
    Impl* impl_;
};

std::string fmt(double v, int prec = 4);

// Pretty console table: columns are right-aligned, first column left-aligned.
std::string table(const std::vector<std::string>& header,
                  const std::vector<std::vector<std::string>>& rows);

}  // namespace apt::io
