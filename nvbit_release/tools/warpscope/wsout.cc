#include "wsout.hh"
#include <fstream>
#include <memory>

static std::ostream* g_wsout_stream = &std::cerr;
static std::unique_ptr<std::ofstream> g_wsout_file;

std::ostream& wsout() {
    return *g_wsout_stream << "#warpscope: ";
}

std::ostream& wsout_stream() {
    return *g_wsout_stream;
}

void set_out_file(const std::string& filename) {
    auto fs = std::make_unique<std::ofstream>(filename);
    if (fs->is_open()) {
        g_wsout_file = std::move(fs);
        g_wsout_stream = g_wsout_file.get();
    } else {
        wsout() << "failed to open log file '" << filename << "'" << std::endl;
    }
}
