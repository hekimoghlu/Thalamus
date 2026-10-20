#include "apt/io.hpp"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace apt::io {

namespace {

struct JsonParser {
    const std::string& s;
    std::size_t i = 0;
    std::map<std::string, double>& out;

    void ws() {
        while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    }
    [[noreturn]] void fail(const char* what) {
        throw std::runtime_error(std::string("json: ") + what + " at offset " + std::to_string(i));
    }
    std::string str() {
        if (s[i] != '"') fail("expected string");
        ++i;
        std::string r;
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\') ++i;
            r.push_back(s[i++]);
        }
        ++i;
        return r;
    }
    void value(const std::string& path) {
        ws();
        if (i >= s.size()) fail("unexpected end");
        const char c = s[i];
        if (c == '{') {
            ++i;
            ws();
            if (s[i] == '}') { ++i; return; }
            for (;;) {
                ws();
                const auto k = str();
                ws();
                if (s[i++] != ':') fail("expected ':'");
                value(path.empty() ? k : path + "." + k);
                ws();
                if (s[i] == ',') { ++i; continue; }
                if (s[i] == '}') { ++i; return; }
                fail("expected ',' or '}'");
            }
        } else if (c == '[') {
            ++i;
            ws();
            if (s[i] == ']') { ++i; return; }
            for (int n = 0;; ++n) {
                value(path + "." + std::to_string(n));
                ws();
                if (s[i] == ',') { ++i; continue; }
                if (s[i] == ']') { ++i; return; }
                fail("expected ',' or ']'");
            }
        } else if (c == '"') {
            str();
        } else if (s.compare(i, 4, "true") == 0 || s.compare(i, 4, "null") == 0) {
            i += 4;
        } else if (s.compare(i, 5, "false") == 0) {
            i += 5;
        } else {
            char* end = nullptr;
            const double v = std::strtod(s.c_str() + i, &end);
            if (end == s.c_str() + i) {
                // NaN / Infinity as Python's json module writes them
                if (s.compare(i, 3, "NaN") == 0) { out[path] = NAN; i += 3; return; }
                if (s.compare(i, 8, "Infinity") == 0) { out[path] = INFINITY; i += 8; return; }
                fail("bad number");
            }
            out[path] = v;
            i = static_cast<std::size_t>(end - s.c_str());
        }
    }
};

}  // namespace

std::map<std::string, double> read_json_numbers(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + p.string());
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string text = ss.str();
    std::map<std::string, double> out;
    JsonParser jp{text, 0, out};
    jp.value("");
    return out;
}

struct CsvWriter::Impl {
    std::ofstream os;
};

CsvWriter::CsvWriter(const std::filesystem::path& p, std::vector<std::string> header)
    : path_(p), impl_(new Impl) {
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
    impl_->os.open(p, std::ios::binary);
    if (!impl_->os) throw std::runtime_error("cannot write " + p.string());
    row(header);
}

CsvWriter::~CsvWriter() { delete impl_; }

CsvWriter& CsvWriter::row(const std::vector<std::string>& cells) {
    for (std::size_t i = 0; i < cells.size(); ++i) {
        if (i) impl_->os << ',';
        const auto& c = cells[i];
        if (c.find_first_of(",\"\n") != std::string::npos) {
            impl_->os << '"';
            for (char ch : c) impl_->os << (ch == '"' ? "\"\"" : std::string(1, ch));
            impl_->os << '"';
        } else {
            impl_->os << c;
        }
    }
    impl_->os << '\n';
    return *this;
}

CsvWriter& CsvWriter::row(const std::vector<double>& cells) {
    std::vector<std::string> s;
    s.reserve(cells.size());
    for (double v : cells) s.push_back(fmt(v, 10));
    return row(s);
}

std::string fmt(double v, int prec) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.*g", prec, v);
    return buf;
}

std::string table(const std::vector<std::string>& header,
                  const std::vector<std::vector<std::string>>& rows) {
    std::vector<std::size_t> w(header.size());
    for (std::size_t c = 0; c < header.size(); ++c) w[c] = header[c].size();
    for (const auto& r : rows)
        for (std::size_t c = 0; c < r.size() && c < w.size(); ++c) w[c] = std::max(w[c], r[c].size());
    std::ostringstream os;
    auto line = [&](const std::vector<std::string>& r) {
        for (std::size_t c = 0; c < w.size(); ++c) {
            const std::string cell = c < r.size() ? r[c] : "";
            const std::string pad(w[c] - cell.size(), ' ');
            os << (c ? "  " : "  ") << (c == 0 ? cell + pad : pad + cell);
        }
        os << '\n';
    };
    line(header);
    std::size_t total = 0;
    for (auto x : w) total += x + 2;
    os << "  " << std::string(total - 2, '-') << '\n';
    for (const auto& r : rows) line(r);
    return os.str();
}

}  // namespace apt::io
