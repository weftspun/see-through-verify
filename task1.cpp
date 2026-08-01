#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <cmath>
#include <vulkan/vulkan.h>

static void cpu_gemm(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < K; k++) s += A[i * K + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

static std::vector<uint32_t> load_spv(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); size_t sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint32_t> c(sz / 4);
    if (fread(c.data(), 1, sz, f) != sz) { fprintf(stderr, "read\n"); exit(1); }
    fclose(f); return c;
}

int main() {
    const int M = 64, N = 64, K = 64;
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0, 1);
    std::vector<float> h_A(M * K), h_B(K * N), h_C_cpu(M * N);
    for (auto &v : h_A) v = dist(rng);
    for (auto &v : h_B) v = dist(rng);
    cpu_gemm(h_A.data(), h_B.data(), h_C_cpu.data(), M, N, K);
    auto spv = load_spv("/tmp/gemm_ref.spv");
    printf("[PASS] SPIR-V loaded (%zu words)\n", spv.size());

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
    printf("[PASS] GPU queue=%u\n", qf);

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

    VkDeviceSize a_sz = M * K * 4, b_sz = K * N * 4, c_sz = M * N * 4, p_sz = 12;

    auto make_buf = [&](VkDeviceSize sz, VkBufferUsageFlags u, VkBuffer *b, VkDeviceMemory *m) -> void {
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

    VkBuffer a_b, b_b, c_b, p_b;
    VkDeviceMemory a_m, b_m, c_m, p_m;
    make_buf(a_sz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &a_b, &a_m);
    make_buf(b_sz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &b_b, &b_m);
    make_buf(c_sz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &c_b, &c_m);
    make_buf(p_sz, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &p_b, &p_m);

    void *vp;
    vkMapMemory(dev, a_m, 0, a_sz, 0, &vp); memcpy(vp, h_A.data(), a_sz); vkUnmapMemory(dev, a_m);
    vkMapMemory(dev, b_m, 0, b_sz, 0, &vp); memcpy(vp, h_B.data(), b_sz); vkUnmapMemory(dev, b_m);
    struct { uint32_t m, n, k; } params = {(uint32_t)M, (uint32_t)N, (uint32_t)K};
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
    dbi[0].buffer = a_b; dbi[0].range = a_sz;
    dbi[1].buffer = b_b; dbi[1].range = b_sz;
    dbi[2].buffer = c_b; dbi[2].range = c_sz;
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
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cmd, &bi);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, lay, 0, 1, &ds, 0, 0);
    vkCmdDispatch(cmd, (M + 31) / 32, (N + 31) / 32, 1);
    vkEndCommandBuffer(cmd);

    VkQueue queue;
    vkGetDeviceQueue(dev, qf, 0, &queue);
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    vkQueueSubmit(queue, 1, &si, 0);
    vkDeviceWaitIdle(dev);

    std::vector<float> h_C_gpu(M * N);
    vkMapMemory(dev, c_m, 0, c_sz, 0, &vp);
    memcpy(h_C_gpu.data(), vp, c_sz);
    vkUnmapMemory(dev, c_m);

    double max_err = 0, max_val = 0;
    for (int i = 0; i < M * N; i++) {
        double e = fabs((double)h_C_gpu[i] - (double)h_C_cpu[i]);
        double v = fabs((double)h_C_cpu[i]);
        if (e > max_err) max_err = e;
        if (v > max_val) max_val = v;
    }
    printf("GEMM %dx%dx%d: max_err=%.6f max_val=%.3f (%.4f%%)\n",
           M, N, K, max_err, max_val, max_val > 0 ? max_err / max_val * 100.0 : 0);

    vkDestroyPipeline(dev, pipeline, 0);
    vkDestroyPipelineLayout(dev, lay, 0);
    vkDestroyDescriptorSetLayout(dev, dsl, 0);
    vkDestroyDescriptorPool(dev, pool, 0);
    vkDestroyShaderModule(dev, mod, 0);
    vkDestroyBuffer(dev, a_b, 0); vkFreeMemory(dev, a_m, 0);
    vkDestroyBuffer(dev, b_b, 0); vkFreeMemory(dev, b_m, 0);
    vkDestroyBuffer(dev, c_b, 0); vkFreeMemory(dev, c_m, 0);
    vkDestroyBuffer(dev, p_b, 0); vkFreeMemory(dev, p_m, 0);
    vkDestroyCommandPool(dev, cmd_pool, 0);
    vkDestroyDevice(dev, 0);
    vkDestroyInstance(inst, 0);

    if (max_err < 1.0) { printf("GREEN - test passes\n"); return 0; }
    else { printf("RED - mismatch\n"); return 1; }
}