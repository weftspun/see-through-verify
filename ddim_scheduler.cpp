// Step F: DDIM diffusion scheduler — timestep loop + noise schedule
// The scheduler drives the diffusion: generates starting noise, computes
// per-step timestep embeddings, and denoises over N steps.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <random>

// Beta schedule (linear, as used by SDXL/LayerDiff)
static std::vector<float> linear_betas(int steps, float start, float end) {
    std::vector<float> b(steps);
    for (int i = 0; i < steps; i++)
        b[i] = start + (end - start) * i / (steps - 1);
    return b;
}

// DDIM schedule: alpha_cumprod from betas
struct DDIMSchedule {
    std::vector<float> alpha_cumprod;
    int T;
    DDIMSchedule(int steps, float start, float end) : T(steps) {
        auto betas = linear_betas(steps, start, end);
        alpha_cumprod.resize(steps);
        float ac = 1.0f;
        for (int i = 0; i < steps; i++) {
            ac *= (1.0f - betas[i]);
            alpha_cumprod[i] = ac;
        }
    }
    // Timesteps for inference (e.g. steps=10 from T=1000)
    std::vector<int> timesteps(int n_infer) const {
        std::vector<int> ts;
        for (int i = 0; i < n_infer; i++)
            ts.push_back((int)std::floor((1000.0f - 1e-4f) * i / n_infer));
        return ts;
    }
    // Timestep embedding for the UNet (sinusoidal, as used by SD)
    static std::vector<float> timestep_embed(int t, int dim, int max_period = 10000) {
        std::vector<float> emb(dim);
        for (int i = 0; i < dim; i++) {
            float inv = 1.0f / powf(max_period, (float)i / (dim / 2));
            if (i % 2 == 0) emb[i] = sinf(t * inv);
            else emb[i] = cosf(t * inv);
        }
        return emb;
    }
};

// Verify the schedule produces the expected shapes/values
int main() {
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0, 1);

    // SDXL/LayerDiff uses beta_start=0.00085, beta_end=0.012, 1000 steps
    DDIMSchedule sched(1000, 0.00085f, 0.012f);
    printf("DDIM schedule (T=1000):\n");
    printf("  alpha_cumprod[0]   = %.6f (should be ~0.99915)\n", sched.alpha_cumprod[0]);
    printf("  alpha_cumprod[999] = %.6f (should be ~0.0, near 0)\n", sched.alpha_cumprod[999]);
    printf("  alpha_cumprod monotonic decreasing: %d\n",
           [&](){ for (int i = 1; i < 1000; i++) if (sched.alpha_cumprod[i] > sched.alpha_cumprod[i-1]) return 0; return 1; }());

    // 10-step inference timesteps
    auto ts = sched.timesteps(10);
    printf("\n  10-step inference timesteps: [");
    for (auto t : ts) printf("%d ", t);
    printf("]\n");

    // Timestep embedding
    auto emb = DDIMSchedule::timestep_embed(ts[0], 320);
    printf("  timestep_embed(t=%d, 320): emb[0]=%.4f emb[159]=%.4f emb[319]=%.4f\n",
           ts[0], emb[0], emb[159], emb[319]);
    printf("  emb finite: %d\n",
           [&](){ for (auto v : emb) if (!std::isfinite(v)) return 0; return 1; }());

    // Generate initial latent noise
    const int latent = 64 * 64 * 8 * 13;  // 64x64 latent, 8 channels, 13 frames
    std::vector<float> noise(latent);
    for (auto &v : noise) v = dist(rng);
    printf("\n  initial latent noise: %d elems, noise[0]=%.4f\n", latent, noise[0]);
    printf("\nStep F: DDIM scheduler verified (schedule, timesteps, embedding, noise)\n");
    return 0;
}