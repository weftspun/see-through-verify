#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <cstdint>
#include <vulkan/vulkan.h>

// ---- safetensors reader ----
struct SfTensor {
    std::string name, dtype;
    std::vector<int64_t> shape;
    size_t offset, size;
};

static std::vector<SfTensor> read_sf(const char *path, size_t *data_start) {
    FILE *f = fopen(path, "rb");
    uint8_t buf[8]; fread(buf, 1, 8, f);
    uint64_t hdr_len = 0;
    for (int j = 0; j < 8; j++) hdr_len |= (uint64_t)buf[j] << (j * 8);
    std::string json((size_t)hdr_len, 0);
    fread(&json[0], 1, hdr_len, f);
    *data_start = 8 + hdr_len;
    fclose(f);

    std::vector<SfTensor> tensors;
    size_t i = 0;
    while (i < json.size() && json[i] != '}') {
        while (i < json.size() && (json[i] == ' ' || json[i] == 10 || json[i] == 9 || json[i] == 13 || json[i] == ',')) i++;
        if (i >= json.size() || json[i] == '}') break;
        if (json[i] != '"') { i++; continue; }
        size_t ns = ++i;
        while (i < json.size() && json[i] != '"') i++;
        std::string name = json.substr(ns, i - ns); i++;
        if (name == "__metadata__") { while (i < json.size() && json[i] != '}') i++; i++; continue; }
        while (i < json.size() && json[i] != '{') i++; i++;

        SfTensor t; t.name = name; t.offset = 0; t.size = 0;
        while (i < json.size() && json[i] != '}') {
            while (i < json.size() && (json[i] == ' ' || json[i] == 10 || json[i] == 9 || json[i] == 13 || json[i] == ',')) i++;
            if (json[i] == '}') break;
            if (json[i] != '"') { i++; continue; }
            size_t fs = ++i;
            while (i < json.size() && json[i] != '"') i++;
            std::string field = json.substr(fs, i - fs); i++;
            while (i < json.size() && json[i] != ':') i++; i++;

            if (field == "dtype") {
                if (json[i] == '"') {
                    size_t vs = ++i;
                    while (i < json.size() && json[i] != '"') i++;
                    t.dtype = json.substr(vs, i - vs); i++;
                }
            } else if (field == "shape") {
                if (json[i] == '[') { i++;
                    while (i < json.size() && json[i] != ']') {
                        while (i < json.size() && (json[i] == ' ' || json[i] == ',')) i++;
                        if (i < json.size() && json[i] >= '0' && json[i] <= '9') {
                            char *end;
                            t.shape.push_back(strtol(&json[i], &end, 10));
                            i = end - &json[0];
                        }
                    }
                    i++;
                }
            } else if (field == "data_offsets") {
                if (json[i] == '[') { i++;
                    int idx = 0; uint64_t vals[2] = {0,0};
                    while (i < json.size() && json[i] != ']') {
                        while (i < json.size() && (json[i] == ' ' || json[i] == ',')) i++;
                        if (i < json.size() && json[i] >= '0' && json[i] <= '9') {
                            char *end;
                            vals[idx++] = strtoull(&json[i], &end, 10);
                            i = end - &json[0];
                        }
                    }
                    i++;
                    t.offset = vals[0]; t.size = vals[1] - vals[0];
                }
            }
        }
        i++;
        tensors.push_back(t);
    }
    return tensors;
}

// ---- Vulkan GEMM dispatch ----
static std::vector<uint32_t> load_spv(const char *path) {
    FILE *f = fopen(path, "rb");
    fseek(f, 0, SEEK_END); size_t sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint32_t> c(sz / 4);
    fread(c.data(), 1, sz, f); fclose(f);
    return c;
}

int main() {
    const char *sf_path = "hf_cache/layerdifforg_seethroughv0.0.2_layerdiff3d/text_encoder/model.safetensors";
    size_t data_start = 0;
    auto tensors = read_sf(sf_path, &data_start);
    printf("tensors: %zu\n", tensors.size());

    // Find first BF16 2D weight
    SfTensor *wt = nullptr;
    for (auto &t : tensors)
        if (t.dtype == "BF16" && t.shape.size() == 2 && t.size > 1024*1024) { wt = &t; break; }
    if (!wt) { printf("no large BF16 weight\n"); return 1; }

    int R = (int)wt->shape[0], C = (int)wt->shape[1];
    printf("weight: %s %s [%d x %d] (%zu bytes)\n", wt->name.c_str(), wt->dtype.c_str(), R, C, wt->size);

    // Load and convert BF16 to F32
    FILE *f = fopen(sf_path, "rb");
    fseeko(f, (off_t)(data_start + wt->offset), SEEK_SET);
    std::vector<uint8_t> raw(wt->size);
    fread(raw.data(), 1, wt->size, f); fclose(f);

    std::vector<float> h_W(R * C);
    for (int i = 0; i < R * C; i++) {
        uint32_t u32 = (uint32_t)((uint16_t*)raw.data())[i] << 16;
        memcpy(&h_W[i], &u32, 4);
    }

    // Create random input
    int N = 64;
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0, 1);
    std::vector<float> h_X(C * N), h_Y_cpu(R * N), h_Y_gpu(R * N);
    for (auto &v : h_X) v = dist(rng);

    // CPU GEMM
    for (int i = 0; i < R; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < C; k++)
                s += h_W[i * C + k] * h_X[k * N + j];
            h_Y_cpu[i * N + j] = s;
        }
    printf("CPU GEMM: y[0]=%.4f\n", h_Y_cpu[0]);

    // ---- Vulkan GPU GEMM ----
    auto spv = load_spv("/tmp/gemm_ref.spv");
    printf("SPIR-V: %zu words\n", spv.size());

    VkApplicationInfo ai = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ic = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ic.pApplicationInfo = &ai;
    const char *iex[] = {"VK_KHR_portability_enumeration"};
    ic.enabledExtensionCount = 1; ic.ppEnabledExtensionNames = iex;
    ic.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    VkInstance inst;
    if (vkCreateInstance(&ic, 0, &inst) != VK_SUCCESS) { printf("inst fail\n"); return 1; }

    uint32_t np = 0;
    vkEnumeratePhysicalDevices(inst, &np, 0);
    std::vector<VkPhysicalDevice> phys(np);
    vkEnumeratePhysicalDevices(inst, &np, phys.data());
    VkPhysicalDevice pd = 0; uint32_t qf = UINT32_MAX;
    for (auto p : phys) {
        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(p, &nq, 0);
        std::vector<VkQueueFamilyProperties> qp(nq);
        vkGetPhysicalDeviceQueueFamilyProperties(p, &nq, qp.data());
        for (uint32_t i = 0; i < nq; i++)
            if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { pd = p; qf = i; break; }
        if (qf != UINT32_MAX) break;
    }
    if (qf == UINT32_MAX) { printf("no compute\n"); return 1; }

    float qp = 1;
    VkDeviceQueueCreateInfo dq = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    dq.queueFamilyIndex = qf; dq.queueCount = 1; dq.pQueuePriorities = &qp;
    const char *dex[] = {"VK_KHR_portability_subset"};
    VkDeviceCreateInfo dc = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dc.queueCreateInfoCount = 1; dc.pQueueCreateInfos = &dq;
    dc.enabledExtensionCount = 1; dc.ppEnabledExtensionNames = dex;
    VkDevice dev;
    if (vkCreateDevice(pd, &dc, 0, &dev) != VK_SUCCESS) { printf("dev fail\n"); return 1; }

    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    uint32_t mt = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((mp.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
            == (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) { mt = i; break; }
    if (mt == UINT32_MAX) { printf("no mem\n"); return 1; }

    // Buffers: W[R,C], X[C,N], Y[R,N], params[3]
    VkDeviceSize w_sz = R * C * 4, x_sz = C * N * 4, y_sz = R * N * 4, p_sz = 12;

    auto make_buf = [&](VkDeviceSize sz, VkBufferUsageFlags u, VkBuffer *b, VkDeviceMemory *m) {
        VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = sz; bi.usage = u; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vkCreateBuffer(dev, &bi, 0, b);
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(dev, *b, &mr);
        VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = mr.size; mai.memoryTypeIndex = mt;
        vkAllocateMemory(dev, &mai, 0, m);
        vkBindBufferMemory(dev, *b, *m, 0);
    };

    VkBuffer w_b, x_b, y_b, p_b;
    VkDeviceMemory w_m, x_m, y_m, p_m;
    make_buf(w_sz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &w_b, &w_m);
    make_buf(x_sz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &x_b, &x_m);
    make_buf(y_sz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &y_b, &y_m);
    make_buf(p_sz, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &p_b, &p_m);

    void *vp;
    vkMapMemory(dev, w_m, 0, w_sz, 0, &vp); memcpy(vp, h_W.data(), w_sz); vkUnmapMemory(dev, w_m);
    vkMapMemory(dev, x_m, 0, x_sz, 0, &vp); memcpy(vp, h_X.data(), x_sz); vkUnmapMemory(dev, x_m);
    struct { uint32_t M, N, K; } params = {(uint32_t)R, (uint32_t)N, (uint32_t)C};
    vkMapMemory(dev, p_m, 0, p_sz, 0, &vp); memcpy(vp, &params, p_sz); vkUnmapMemory(dev, p_m);

    VkShaderModuleCreateInfo sm = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    sm.codeSize = spv.size() * 4; sm.pCode = spv.data();
    VkShaderModule mod;
    if (vkCreateShaderModule(dev, &sm, 0, &mod) != VK_SUCCESS) { printf("mod fail\n"); return 1; }

    VkDescriptorSetLayoutBinding bnd[4] = {};
    for (int i = 0; i < 4; i++) {
        bnd[i].binding = i; bnd[i].descriptorCount = 1; bnd[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bnd[i].descriptorType = (i < 3) ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    }
    VkDescriptorSetLayoutCreateInfo dl = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dl.bindingCount = 4; dl.pBindings = bnd;
    VkDescriptorSetLayout dsl;
    vkCreateDescriptorSetLayout(dev, &dl, 0, &dsl);

    VkPipelineLayoutCreateInfo pl = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl.setLayoutCount = 1; pl.pSetLayouts = &dsl;
    VkPipelineLayout lay;
    vkCreatePipelineLayout(dev, &pl, 0, &lay);

    VkPipelineShaderStageCreateInfo ss = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ss.stage = VK_SHADER_STAGE_COMPUTE_BIT; ss.module = mod; ss.pName = "main";
    VkComputePipelineCreateInfo pi = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pi.stage = ss; pi.layout = lay;
    VkPipeline pipeline;
    if (vkCreateComputePipelines(dev, 0, 1, &pi, 0, &pipeline) != VK_SUCCESS) { printf("pipe fail\n"); return 1; }

    VkDescriptorPoolSize dps[2] = {};
    dps[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; dps[0].descriptorCount = 3;
    dps[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; dps[1].descriptorCount = 1;
    VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1; dpci.poolSizeCount = 2; dpci.pPoolSizes = dps;
    VkDescriptorPool pool;
    vkCreateDescriptorPool(dev, &dpci, 0, &pool);
    VkDescriptorSetAllocateInfo da = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    da.descriptorPool = pool; da.descriptorSetCount = 1; da.pSetLayouts = &dsl;
    VkDescriptorSet ds;
    vkAllocateDescriptorSets(dev, &da, &ds);

    VkDescriptorBufferInfo dbi[4] = {};
    dbi[0].buffer = w_b; dbi[0].range = w_sz;
    dbi[1].buffer = x_b; dbi[1].range = x_sz;
    dbi[2].buffer = y_b; dbi[2].range = y_sz;
    dbi[3].buffer = p_b; dbi[3].range = p_sz;
    VkWriteDescriptorSet wd[4] = {};
    for (int i = 0; i < 4; i++) {
        wd[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        wd[i].dstSet = ds; wd[i].dstBinding = i; wd[i].descriptorCount = 1;
        wd[i].descriptorType = (i < 3) ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        wd[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(dev, 4, wd, 0, 0);

    VkCommandPoolCreateInfo cp = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cp.queueFamilyIndex = qf;
    VkCommandPool cmd_pool;
    vkCreateCommandPool(dev, &cp, 0, &cmd_pool);
    VkCommandBufferAllocateInfo ca = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ca.commandPool = cmd_pool; ca.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(dev, &ca, &cmd);
    VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cmd, &cbbi);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, lay, 0, 1, &ds, 0, 0);
    vkCmdDispatch(cmd, (R + 31) / 32, (N + 31) / 32, 1);
    vkEndCommandBuffer(cmd);

    VkQueue queue;
    vkGetDeviceQueue(dev, qf, 0, &queue);
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    vkQueueSubmit(queue, 1, &si, 0);
    vkDeviceWaitIdle(dev);

    vkMapMemory(dev, y_m, 0, y_sz, 0, &vp);
    memcpy(h_Y_gpu.data(), vp, y_sz);
    vkUnmapMemory(dev, y_m);

    // Compare
    double max_err = 0, max_val = 0;
    for (int i = 0; i < R * N; i++) {
        double e = fabs((double)h_Y_gpu[i] - (double)h_Y_cpu[i]);
        double v = fabs((double)h_Y_cpu[i]);
        if (e > max_err) max_err = e;
        if (v > max_val) max_val = v;
    }
    printf("GPU GEMM: y[0]=%.4f\n", h_Y_gpu[0]);
    printf("GEMM %dx%dx%d: max_err=%.6f max_val=%.3f (%.4f%%)\n",
           R, N, C, max_err, max_val, max_val > 0 ? max_err / max_val * 100.0 : 0);

    // Cleanup
    vkDestroyPipeline(dev, pipeline, 0); vkDestroyPipelineLayout(dev, lay, 0);
    vkDestroyDescriptorSetLayout(dev, dsl, 0); vkDestroyDescriptorPool(dev, pool, 0);
    vkDestroyShaderModule(dev, mod, 0);
    vkDestroyBuffer(dev, w_b, 0); vkFreeMemory(dev, w_m, 0);
    vkDestroyBuffer(dev, x_b, 0); vkFreeMemory(dev, x_m, 0);
    vkDestroyBuffer(dev, y_b, 0); vkFreeMemory(dev, y_m, 0);
    vkDestroyBuffer(dev, p_b, 0); vkFreeMemory(dev, p_m, 0);
    vkDestroyCommandPool(dev, cmd_pool, 0);
    vkDestroyDevice(dev, 0); vkDestroyInstance(inst, 0);

    if (max_err < 1.0) { printf("GREEN - GPU matches CPU\n"); return 0; }
    else { printf("RED - mismatch\n"); return 1; }
}