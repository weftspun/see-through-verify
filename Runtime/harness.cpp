// See-through runtime harness: Vulkan compute + DuckDB Parquet loader
// Compile: clang++ harness.cpp -o harness -lduckdb -lvulkan
// Usage:   harness <input.parquet> <shader.spv> <M> <N> <K>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <cstdint>

// ---------------------------------------------------------------------------
// Parquet loader via DuckDB C API
// ---------------------------------------------------------------------------

// DuckDB C API is header-only — include duckdb.h from lean-duckdb vendor
#include "duckdb.h"

static std::vector<float> load_parquet(const char * path, size_t * out_n) {
    duckdb_database db;
    duckdb_connection con;
    if (duckdb_open(NULL, &db) == DuckDBError) {
        fprintf(stderr, "duckdb_open failed\n");
        exit(1);
    }
    if (duckdb_connect(db, &con) == DuckDBError) {
        fprintf(stderr, "duckdb_connect failed\n");
        exit(1);
    }

    char sql[4096];
    snprintf(sql, sizeof(sql),
        "SELECT val FROM read_parquet('%s') ORDER BY idx", path);

    duckdb_result result;
    if (duckdb_query(con, sql, &result) == DuckDBError) {
        fprintf(stderr, "duckdb_query failed for %s\n", path);
        fprintf(stderr, "error: %s\n", duckdb_result_error(&result));
        exit(1);
    }

    idx_t n = duckdb_row_count(&result);
    std::vector<float> data(n);
    for (idx_t i = 0; i < n; i++) {
        data[i] = (float) duckdb_value_double(&result, i, 0);
    }

    *out_n = n;
    duckdb_destroy_result(&result);
    duckdb_disconnect(&con);
    duckdb_close(&db);
    return data;
}

// ---------------------------------------------------------------------------
// SPIR-V loader
// ---------------------------------------------------------------------------

static std::vector<uint32_t> load_spirv(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "failed to open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint32_t> code(size / 4);
    if (fread(code.data(), 1, size, f) != size) {
        fprintf(stderr, "failed to read %s\n", path); exit(1);
    }
    fclose(f);
    return code;
}

// ---------------------------------------------------------------------------
// Vulkan compute dispatch
// ---------------------------------------------------------------------------

#define VK_NO_PROTOTYPES
#include "vulkan/vulkan.h"

static VkInstance g_vk;
static VkDevice g_dev;
static VkQueue g_queue;
static VkCommandPool g_pool;
static VkCommandBuffer g_cmd;
static uint32_t g_queue_family;

static void vk_init() {
    // Create instance
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "see-through";
    app.apiVersion = VK_API_VERSION_1_3;

    const char * ext[] = {VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = 1;
    ici.ppEnabledExtensionNames = ext;
    if (vkCreateInstance(&ici, NULL, &g_vk) != VK_SUCCESS) exit(1);

    // Pick first GPU
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(g_vk, &n, NULL);
    std::vector<VkPhysicalDevice> phys(n);
    vkEnumeratePhysicalDevices(g_vk, &n, phys.data());
    if (n == 0) exit(1);

    uint32_t qf = UINT32_MAX;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t m = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys[i], &m, NULL);
        std::vector<VkQueueFamilyProperties> qp(m);
        vkGetPhysicalDeviceQueueFamilyProperties(phys[i], &m, qp.data());
        for (uint32_t j = 0; j < m; j++) {
            if (qp[j].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                qf = j; break;
            }
        }
        if (qf != UINT32_MAX) { g_queue_family = qf; break; }
    }
    if (qf == UINT32_MAX) exit(1);

    VkPhysicalDevice pdev = phys[qf]; // first with compute

    // Create logical device
    float qp = 1.0f;
    VkDeviceQueueCreateInfo dq = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    dq.queueFamilyIndex = g_queue_family;
    dq.queueCount = 1;
    dq.pQueuePriorities = &qp;

    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &dq;
    if (vkCreateDevice(pdev, &dci, NULL, &g_dev) != VK_SUCCESS) exit(1);

    vkGetDeviceQueue(g_dev, g_queue_family, 0, &g_queue);

    // Command pool + buffer
    VkCommandPoolCreateInfo cp = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cp.queueFamilyIndex = g_queue_family;
    vkCreateCommandPool(g_dev, &cp, NULL, &g_pool);

    VkCommandBufferAllocateInfo cb = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cb.commandPool = g_pool;
    cb.commandBufferCount = 1;
    vkAllocateCommandBuffers(g_dev, &cb, &g_cmd);
}

static void vk_shutdown() {
    vkDeviceWaitIdle(g_dev);
    vkDestroyCommandPool(g_dev, g_pool, NULL);
    vkDestroyDevice(g_dev, NULL);
    vkDestroyInstance(g_vk, NULL);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <weights.parquet> <shader.spv> <M> <N> <K>\n", argv[0]);
        return 1;
    }

    const char * parquet_path = argv[1];
    const char * spv_path = argv[2];
    int M = atoi(argv[3]), N = atoi(argv[4]), K = atoi(argv[5]);

    // Load weights
    size_t n_weights = 0;
    auto weights = load_parquet(parquet_path, &n_weights);
    printf("weights: %zu floats from %s\n", n_weights, parquet_path);

    // Load shader
    auto spv = load_spirv(spv_path);
    printf("SPIR-V: %zu words from %s\n", spv.size(), spv_path);

    // Initialize Vulkan
    vk_init();
    printf("Vulkan: device ready\n");

    // Create buffers
    VkBuffer a_buf, b_buf, c_buf;
    VkDeviceMemory a_mem, b_mem, c_mem;
    VkDeviceSize a_size = M * K * sizeof(float);
    VkDeviceSize b_size = K * N * sizeof(float);
    VkDeviceSize c_size = M * N * sizeof(float);

    auto create_buffer = [&](VkDeviceSize size, VkBuffer & buf, VkDeviceMemory & mem) {
        VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = size;
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        vkCreateBuffer(g_dev, &bi, NULL, &buf);

        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(g_dev, buf, &mr);

        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = mr.size;
        ai.memoryTypeIndex = 0; // simplified — pick the first memory type
        vkAllocateMemory(g_dev, &ai, NULL, &mem);
        vkBindBufferMemory(g_dev, buf, mem, 0);
    };

    create_buffer(a_size, a_buf, a_mem);
    create_buffer(b_size, b_buf, b_mem);
    create_buffer(c_size, c_buf, c_mem);

    // Upload weights
    vkMapMemory(g_dev, a_mem, 0, a_size, 0, (void**)&a_mem);
    memcpy(a_mem, weights.data(), a_size);
    vkUnmapMemory(g_dev, a_mem);

    // Create shader module
    VkShaderModuleCreateInfo sm = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    sm.codeSize = spv.size() * 4;
    sm.pCode = spv.data();
    VkShaderModule mod;
    vkCreateShaderModule(g_dev, &sm, NULL, &mod);

    // Create compute pipeline
    VkPipelineShaderStageCreateInfo ss = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ss.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ss.module = mod;
    ss.pName = "main";

    VkComputePipelineCreateInfo pi = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pi.stage = ss;
    pi.layout = VK_NULL_HANDLE; // null — no descriptors for this minimal test

    VkPipeline pipeline;
    vkCreateComputePipelines(g_dev, VK_NULL_HANDLE, 1, &pi, NULL, &pipeline);

    // Dispatch
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(g_cmd, &bi);

    vkCmdBindPipeline(g_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdDispatch(g_cmd, (M + 63) / 64, (N + 63) / 64, 1);

    vkEndCommandBuffer(g_cmd);

    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &g_cmd;
    vkQueueSubmit(g_queue, 1, &si, VK_NULL_HANDLE);
    vkDeviceWaitIdle(g_dev);

    printf("dispatch complete\n");

    // Read back result
    std::vector<float> result(M * N);
    void * mapped;
    vkMapMemory(g_dev, c_mem, 0, c_size, 0, &mapped);
    memcpy(result.data(), mapped, c_size);
    vkUnmapMemory(g_dev, c_mem);

    printf("result[0] = %f\n", result[0]);

    vkDestroyPipeline(g_dev, pipeline, NULL);
    vkDestroyShaderModule(g_dev, mod, NULL);
    vkDestroyBuffer(g_dev, a_buf, NULL);
    vkDestroyBuffer(g_dev, b_buf, NULL);
    vkDestroyBuffer(g_dev, c_buf, NULL);
    vkFreeMemory(g_dev, a_mem, NULL);
    vkFreeMemory(g_dev, b_mem, NULL);
    vkFreeMemory(g_dev, c_mem, NULL);

    vk_shutdown();
    return 0;
}