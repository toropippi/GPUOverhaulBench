#include "bench_support.hpp"

int main(int argc, char** argv) {
    bench::Options options{};
    try {
        options = bench::parse_common_args(argc, argv);
        bench::emit_json(
            "{"
            "\"status\":\"ok\","
            "\"primary_metric\":\"replace_with_primary_metric_name\","
            "\"unit\":\"replace_with_unit\","
            "\"parameters\":{"
            "\"iterations\":" + std::to_string(options.iterations) + ","
            "\"warmup\":" + std::to_string(options.warmup) + ","
            "\"sizes_mb\":" + bench::sizes_to_json(options.sizes_mb) +
                "},"
                "\"measurement\":{"
                "\"timing_backend\":\"cuda_event\""
                "},"
                "\"validation\":{"
                "\"passed\":true"
                "}"
                "}");
        return 0;
    } catch (const std::exception& ex) {
        bench::emit_json(bench::make_error_json("failed", ex.what(), options));
        return 1;
    }
}
