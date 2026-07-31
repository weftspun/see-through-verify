// GGUF → Parquet converter (zstd-compressed, via DuckDB SQL)
// Reads model weights from GGUF format and writes them as Parquet files
// that lean-duckdb can consume. Each tensor becomes one Parquet file:
//   weights/<model>/<tensor_name>.parquet
//
// Usage: gguf-to-parquet <model.gguf> <output-dir>
//
// Example: gguf-to-parquet models/layerdiff-unet.gguf weights/layerdiff-unet

#include "gguf.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <cstdint>

namespace fs = std::filesystem;

// Write a DuckDB-compatible SQL script that creates the Parquet file
// from the raw tensor data. The SQL uses DuckDB's ability to read raw
// binary files and write Parquet.
static void write_parquet_sql(
    const std::string & out_dir,
    const std::string & tensor_name,
    const std::vector<int64_t> & shape,
    ggml_type dtype,
    const std::vector<uint8_t> & raw_data)
{
    fs::create_directories(out_dir);
    std::string raw_path = out_dir + "/" + tensor_name + ".raw";
    std::string parquet_path = out_dir + "/" + tensor_name + ".parquet";
    std::string sql_path = out_dir + "/" + tensor_name + ".sql";

    // Write raw binary data
    {
        FILE * f = fopen(raw_path.c_str(), "wb");
        if (!f) { fprintf(stderr, "failed to write %s\n", raw_path.c_str()); exit(1); }
        fwrite(raw_data.data(), 1, raw_data.size(), f);
        fclose(f);
    }

    // Write SQL script that DuckDB runs to create the Parquet
    // DuckDB can read raw binary files via read_csv_auto with no header
    FILE * f = fopen(sql_path.c_str(), "w");
    if (!f) { fprintf(stderr, "failed to write %s\n", sql_path.c_str()); exit(1); }

    fprintf(f, "-- DuckDB SQL: convert %s tensor to Parquet with zstd\n", tensor_name.c_str());
    fprintf(f, "-- Run: duckdb < %s\n", sql_path.c_str());
    fprintf(f, "\n");
    fprintf(f, "CREATE TABLE tensor_%s AS\n", tensor_name.c_str());
    fprintf(f, "  SELECT * FROM read_csv_auto('%s', header=false, columns={'val': 'FLOAT'}, delim='\\n');\n", raw_path.c_str());
    fprintf(f, "\n");
    fprintf(f, "COPY (SELECT row_number() OVER () - 1 AS idx, val FROM tensor_%s)\n", tensor_name.c_str());
    fprintf(f, "  TO '%s' (FORMAT PARQUET, CODEC ZSTD, COMPRESSION_LEVEL 19);\n", parquet_path.c_str());
    fprintf(f, "\n");

    // Write metadata as a comment
    fprintf(f, "-- Metadata:\n");
    fprintf(f, "--   name:   %s\n", tensor_name.c_str());
    fprintf(f, "--   dtype:  %s\n", ggml_type_name(dtype));
    fprintf(f, "--   shape:  [");
    for (size_t i = 0; i < shape.size(); i++) {
        if (i > 0) fprintf(f, ", ");
        fprintf(f, "%lld", (long long) shape[i]);
    }
    fprintf(f, "]\n");
    fprintf(f, "--   nelems: %zu\n", raw_data.size() / ggml_type_size(dtype));
    fprintf(f, "--   bytes:  %zu\n", raw_data.size());

    fclose(f);
    printf("  wrote %s -> %s\n", tensor_name.c_str(), parquet_path.c_str());
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <output-dir>\n", argv[0]);
        return 1;
    }

    const char * gguf_path = argv[1];
    const char * out_dir = argv[2];

    printf("reading %s -> %s\n", gguf_path, out_dir);

    // Open GGUF file
    gguf_context * g = gguf_init_from_file(gguf_path, { false, nullptr });
    if (!g) {
        fprintf(stderr, "failed to load GGUF: %s\n", gguf_path);
        return 1;
    }

    ggml_context * ctx = nullptr;
    gguf_init_params params = { true, &ctx };
    g = gguf_init_from_file(gguf_path, params);
    if (!g || !ctx) {
        fprintf(stderr, "failed to init GGUF context\n");
        return 1;
    }

    int n_tensors = gguf_get_n_tensors(g);
    printf("tensors: %d\n", n_tensors);

    // Get the data offset (where the raw tensor data starts)
    size_t data_off = gguf_get_data_offset(g);

    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(g, i);
        size_t offset = gguf_get_tensor_offset(g, i);
        ggml_tensor * t = ggml_get_tensor(ctx, name);
        if (!t) {
            fprintf(stderr, "  skipping %s (not found in context)\n", name);
            continue;
        }

        // Read tensor dimensions
        std::vector<int64_t> shape;
        for (int d = 0; d < t->n_dims; d++) {
            shape.push_back(t->ne[d]);
        }

        ggml_type dtype = t->type;
        size_t nbytes = ggml_nbytes(t);
        size_t nelems = ggml_nelements(t);

        printf("  [%d/%d] %s: %s %s (%zu elems, %zu bytes)\n",
               i + 1, n_tensors, name,
               ggml_type_name(dtype),
               [&]() {
                   std::string s = "[";
                   for (int d = 0; d < t->n_dims; d++) {
                       if (d > 0) s += "x";
                       s += std::to_string(t->ne[d]);
                   }
                   s += "]";
                   return s;
               }().c_str(),
               nelems, nbytes);

        // Read raw tensor data from the GGUF file
        FILE * f = fopen(gguf_path, "rb");
        if (!f) { fprintf(stderr, "failed to open %s\n", gguf_path); exit(1); }
        fseeko(f, (off_t)(data_off + offset), SEEK_SET);
        std::vector<uint8_t> raw(nbytes);
        if (fread(raw.data(), 1, nbytes, f) != nbytes) {
            fprintf(stderr, "failed to read tensor %s\n", name);
            fclose(f);
            continue;
        }
        fclose(f);

        // Convert f16 to f32 for Parquet output (DuckDB reads FLOAT)
        if (dtype == GGML_TYPE_F16) {
            std::vector<float> f32_data(nelems);
            for (size_t j = 0; j < nelems; j++) {
                f32_data[j] = ggml_fp16_to_fp32(((uint16_t*)raw.data())[j]);
            }
            std::vector<uint8_t> f32_raw(nelems * sizeof(float));
            memcpy(f32_raw.data(), f32_data.data(), f32_raw.size());
            write_parquet_sql(out_dir, name, shape, GGML_TYPE_F32, f32_raw);
        } else if (dtype == GGML_TYPE_F32) {
            write_parquet_sql(out_dir, name, shape, dtype, raw);
        } else {
            // For quantized types, write raw bytes (user needs to know the format)
            write_parquet_sql(out_dir, name, shape, dtype, raw);
        }
    }

    printf("\nTo convert to Parquet, run:\n");
    printf("  for f in %s/*.sql; do duckdb < \"$f\"; done\n", out_dir);
    printf("Then remove the .raw and .sql files.\n");

    gguf_free(g);
    ggml_free(ctx);
    return 0;
}