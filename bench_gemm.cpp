#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <vulkan/vulkan.h>

static std::vector<uint32_t> load_spv(const char *path) {
    FILE *f = fopen(path, "rb");
    fseek(f, 0, SEEK_END); size_t sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint32_t> c(sz / 4);
    fread(c.data(), 1, sz, f); fclose(f);
    return c;
}

int main() {
    const int M = 4096, N = 4096, K = 4096;
    auto spv = load_spv("/tmp/gemm_ref.spv");

    VkApplicationInfo ai = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ic = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ic.pApplicationInfo = &ai;
    const char *iex[] = {"VK_KHR_portability_enumeration"};
    ic.enabledExtensionCount = 1; ic.ppEnabledExtensionNames = iex;
    ic.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    VkInstance inst;
    vkCreateInstance(&ic, 0, &inst);

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

    float qp = 1;
    VkDeviceQueueCreateInfo dq = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    dq.queueFamilyIndex = qf; dq.queueCount = 1; dq.pQueuePriorities = &qp;
    const char *dex[] = {"VK_KHR_portability_subset"};
    VkDeviceCreateInfo dc = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dc.queueCreateInfoCount = 1; dc.pQueueCreateInfos = &dq;
    dc.enabledExtensionCount = 1; dc.ppEnabledExtensionNames = dex;
    VkDevice dev;
    vkCreateDevice(pd, &dc, 0, &dev);

    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    uint32_t mt = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((mp.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
            == (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) { mt = i; break; }

    VkDeviceSize w_sz = M * K * 4, x_sz = K * N * 4, y_sz = M * N * 4, p_sz = 12;

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

    // Upload dummy data
    void *vp;
    std::vector<float> dummy_w(M*K, 1.0f), dummy_x(K*N, 1.0f);
    vkMapMemory(dev, w_m, 0, w_sz, 0, &vp); memcpy(vp, dummy_w.data(), w_sz); vkUnmapMemory(dev, w_m);
    vkMapMemory(dev, x_m, 0, x_sz, 0, &vp); memcpy(vp, dummy_x.data(), x_sz); vkUnmapMemory(dev, x_m);
    struct { uint32_t M, N, K; } params = {(uint32_t)M, (uint32_t)N, (uint32_t)K};
    vkMapMemory(dev, p_m, 0, p_sz, 0, &vp); memcpy(vp, &params, p_sz); vkUnmapMemory(dev, p_m);

    VkShaderModuleCreateInfo sm = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    sm.codeSize = spv.size() * 4; sm.pCode = spv.data();
    VkShaderModule mod;
    vkCreateShaderModule(dev, &sm, 0, &mod);

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
    vkCreateComputePipelines(dev, 0, 1, &pi, 0, &pipeline);

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

    // Warm up
    VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cmd, &cbbi);
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

    // Benchmark: 10 dispatches
    auto t0 = std::chrono::steady_clock::now();
    for (int iter = 0; iter < 10; iter++) {
        vkBeginCommandBuffer(cmd, &cbbi);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, lay, 0, 1, &ds, 0, 0);
        vkCmdDispatch(cmd, (M + 31) / 32, (N + 31) / 32, 1);
        vkEndCommandBuffer(cmd);
        vkQueueSubmit(queue, 1, &si, 0);
        vkDeviceWaitIdle(dev);
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("GEMM %dx%dx%d: %.3f ms total, %.3f ms per dispatch (avg of 10)\\n", M, N, K, ms, ms / 10.0);

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
    return 0;
}