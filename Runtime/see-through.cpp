// see-through CLI — safetensors native runtime
//
//   see-through -i in.png -o out.psd [--steps N] [--res N]
//   see-through --demo <safetensors_path>
//
// Reads model weights directly from safetensors files (BF16, no widening).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <filesystem>
#include <cstdint>

// ---------------------------------------------------------------------------
// Safetensors reader
// ---------------------------------------------------------------------------

struct SfTensor {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    size_t offset;
    size_t size;
};

struct SfHeader {
    std::vector<SfTensor> tensors;
    size_t data_start;
};

// Simple JSON parser for safetensors header format.
// The header is a flat object: {"name": {"dtype":"..","shape":[...],"data_offsets":[s,e]}, ...}
static std::string json_str(const std::string &s, size_t &i) {
    std::string out;
    i++; // skip opening quote
    while (i < s.size() && s[i] != '"') {
        if (s[i] == '\\') { i++; if (i < s.size()) out += s[i]; }
        else out += s[i];
        i++;
    }
    i++; // skip closing quote
    return out;
}

static SfHeader read_safetensors(const char *path) {
    SfHeader hdr;
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "error: can't open %s\n", path); exit(1); }

    uint8_t hdr_len_buf[8];
    if (fread(hdr_len_buf, 1, 8, f) != 8) { fprintf(stderr, "error: short read %s\n", path); exit(1); }
    uint64_t hdr_len = 0;
    for (int j = 0; j < 8; j++) hdr_len |= (uint64_t)hdr_len_buf[j] << (j * 8);

    std::string json((size_t)hdr_len, '\0');
    if (fread(&json[0], 1, hdr_len, f) != hdr_len) { fprintf(stderr, "error: short header %s\n", path); exit(1); }
    hdr.data_start = 8 + hdr_len;
    fclose(f);

    size_t i = 0;
    while (i < json.size()) {
        // Skip whitespace and commas
        while (i < json.size() && (json[i] == ' ' || json[i] == '\n' || json[i] == '\t' || json[i] == '\r' || json[i] == ',')) i++;
        if (i >= json.size() || json[i] == '}') break;
        if (json[i] != '"') { i++; continue; }

        // Read tensor name
        size_t ks = i;
        std::string name = json_str(json, i);

        // Skip __metadata__
        if (name == "__metadata__") {
            // Skip to end of value
            while (i < json.size() && json[i] != '{' && json[i] != '}' && json[i] != ',') i++;
            if (i < json.size() && json[i] == '{') {
                int depth = 1; i++;
                while (i < json.size() && depth > 0) {
                    if (json[i] == '{') depth++;
                    if (json[i] == '}') depth--;
                    i++;
                }
            }
            continue;
        }

        // Skip colon
        while (i < json.size() && json[i] != ':') i++;
        i++;

        // Expect value object
        while (i < json.size() && json[i] == ' ') i++;
        if (i >= json.size() || json[i] != '{') continue;

        i++; // skip {
        SfTensor t;
        t.name = name;
        t.offset = 0;
        t.size = 0;

        while (i < json.size() && json[i] != '}') {
            // Skip whitespace/comma
            while (i < json.size() && (json[i] == ' ' || json[i] == '\n' || json[i] == '\t' || json[i] == ',')) i++;
            if (i >= json.size() || json[i] == '}') break;

            if (json[i] != '"') { i++; continue; }
            std::string field = json_str(json, i);

            // Skip colon
            while (i < json.size() && json[i] != ':') i++;
            i++;

            if (field == "dtype") {
                t.dtype = json_str(json, i);
            } else if (field == "shape") {
                t.shape.clear();
                if (json[i] == '[') {
                    i++;
                    while (i < json.size() && json[i] != ']') {
                        while (i < json.size() && (json[i] == ' ' || json[i] == ',')) i++;
                        if (i < json.size() && json[i] >= '0' && json[i] <= '9') {
                            char *end;
                            t.shape.push_back(strtol(&json[i], &end, 10));
                            i = end - &json[0];
                        }
                    }
                    i++; // skip ]
                }
            } else if (field == "data_offsets") {
                if (json[i] == '[') {
                    i++;
                    int idx = 0;
                    uint64_t vals[2] = {0, 0};
                    while (i < json.size() && json[i] != ']') {
                        while (i < json.size() && (json[i] == ' ' || json[i] == ',')) i++;
                        if (i < json.size() && json[i] >= '0' && json[i] <= '9') {
                            char *end;
                            vals[idx++] = strtoull(&json[i], &end, 10);
                            i = end - &json[0];
                        }
                    }
                    i++; // skip ]
                    t.offset = (size_t)vals[0];
                    t.size = (size_t)(vals[1] - vals[0]);
                }
            } else {
                // Skip unknown field value
                if (json[i] == '"') json_str(json, i);
                else if (json[i] == '[') { int d = 1; i++; while (i < json.size() && d > 0) { if (json[i] == '[') d++; if (json[i] == ']') d--; i++; } }
                else if (json[i] == '{') { int d = 1; i++; while (i < json.size() && d > 0) { if (json[i] == '{') d++; if (json[i] == '}') d--; i++; } }
            }
        }
        i++; // skip }
        hdr.tensors.push_back(t);
    }

    return hdr;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    std::string weights_dir = "weights";
    std::string in_path, out_path = "out.psd";
    int steps = 30, res = 1280, depth_res = 768;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "-i") in_path = next();
        else if (a == "-o") out_path = next();
        else if (a == "--steps") steps = std::stoi(next());
        else if (a == "--res") res = std::stoi(next());
        else if (a == "--depth-res") depth_res = std::stoi(next());
        else if (a == "--demo") {
            std::string sf_path = next();
            auto hdr = read_safetensors(sf_path.c_str());
            printf("Safetensors: %zu tensors\n", hdr.tensors.size());
            for (const auto &t : hdr.tensors) {
                printf("  %s %s [", t.name.c_str(), t.dtype.c_str());
                for (size_t d = 0; d < t.shape.size(); d++) {
                    if (d) printf("x");
                    printf("%lld", (long long)t.shape[d]);
                }
                printf("] (%zu bytes)\n", t.size);
            }
            return 0;
        }
        else { fprintf(stderr, "usage: see-through -i in.png -o out.psd [--steps N] [--res N]\n"); return 1; }
    }

    if (in_path.empty()) {
        fprintf(stderr, "usage: see-through -i in.png -o out.psd [--steps N] [--res N]\n");
        fprintf(stderr, "       see-through --demo <safetensors_path>\n");
        return 1;
    }

    printf("input: %s\n", in_path.c_str());
    printf("output: %s\n", out_path.c_str());
    printf("steps: %d, res: %d\n", steps, res);

    // List all available model weights — search relative to cwd or SEE_THROUGH_DIR
    std::string base = weights_dir;
    if (const char *env = getenv("SEE_THROUGH_DIR")) base = env;

    // Fallback: if weights/ doesn't exist, try hf_cache/
    std::string check_path = base + "/layerdifforg_seethroughv0.0.2_layerdiff3d";
    if (!std::filesystem::exists(check_path) && std::filesystem::exists("hf_cache")) {
        base = "hf_cache";
    }

    std::vector<std::pair<std::string,std::string>> model_files = {
        {"layerdiff3d text_encoder",       base + "/layerdifforg_seethroughv0.0.2_layerdiff3d/text_encoder/model.safetensors"},
        {"layerdiff3d text_encoder_2",     base + "/layerdifforg_seethroughv0.0.2_layerdiff3d/text_encoder_2/model.safetensors"},
        {"layerdiff3d unet",               base + "/layerdifforg_seethroughv0.0.2_layerdiff3d/unet/diffusion_pytorch_model.safetensors"},
        {"layerdiff3d vae",                base + "/layerdifforg_seethroughv0.0.2_layerdiff3d/vae/diffusion_pytorch_model.safetensors"},
        {"layerdiff3d trans_vae",          base + "/layerdifforg_seethroughv0.0.2_layerdiff3d/trans_vae/diffusion_pytorch_model.safetensors"},
        {"marigold text_encoder",          base + "/24yearsold_seethroughv0.0.1_marigold/text_encoder/model.safetensors"},
        {"marigold unet",                  base + "/24yearsold_seethroughv0.0.1_marigold/unet/diffusion_pytorch_model.safetensors"},
        {"marigold vae",                   base + "/24yearsold_seethroughv0.0.1_marigold/vae/diffusion_pytorch_model.safetensors"},
    };

    for (const auto &[label, path] : model_files) {
        if (std::filesystem::exists(path)) {
            auto hdr = read_safetensors(path.c_str());
            printf("\n%s (%s): %zu tensors\n", label.c_str(), path.c_str(), hdr.tensors.size());
            size_t total_bytes = 0;
            for (const auto &t : hdr.tensors) total_bytes += t.size;
            printf("  total: %zu MB\n", total_bytes / 1024 / 1024);
        } else {
            printf("\n%s: not found at %s\n", label.c_str(), path.c_str());
        }
    }

    return 0;
}