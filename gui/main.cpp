// see-through GUI — ThorVG + GLFW
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <atomic>
#include <vector>
#include <string>
#include <chrono>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <thorvg.h>

struct PipelineState {
    std::atomic<bool> running{false};
    std::atomic<float> progress{0.0f};
    std::atomic<bool> done{false};
};

static void pipeline_thread(PipelineState *s) {
    s->running = true;
    for (int i = 0; i <= 30; i++) {
        if (!s->running) break;
        s->progress = (float)i / 30.0f;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    s->done = true;
    s->running = false;
    s->progress = 1.0f;
}

struct App {
    GLFWwindow *window = nullptr;
    int w = 900, h = 600;
    PipelineState pipeline;
    std::string input_path;
};

int main() {
    if (!glfwInit()) { fprintf(stderr, "glfwInit failed\n"); return 1; }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    App app;
    app.window = glfwCreateWindow(app.w, app.h, "See-Through", nullptr, nullptr);
    if (!app.window) { fprintf(stderr, "window failed\n"); return 1; }
    glfwSetWindowUserPointer(app.window, &app);

    tvg::Initializer::init(0);

    while (!glfwWindowShouldClose(app.window)) {
        glfwPollEvents();

        int fb_w, fb_h;
        glfwGetFramebufferSize(app.window, &fb_w, &fb_h);
        if (fb_w == 0 || fb_h == 0) continue;

        auto *buf = (uint32_t *)calloc(fb_w * fb_h, 4);
        auto canvas = tvg::SwCanvas::gen();
        canvas->target(buf, fb_w, fb_w, fb_h, tvg::ColorSpace::ABGR8888);

        // Background
        auto bg = tvg::Shape::gen();
        bg->appendRect(0, 0, fb_w, fb_h, 0, 0);
        bg->fill(30, 30, 40);
        canvas->add(bg);

        // Progress bar
        if (app.pipeline.running || app.pipeline.done) {
            auto pb = tvg::Shape::gen();
            int pw = (int)((fb_w - 40) * app.pipeline.progress);
            pb->appendRect(20, fb_h - 60, pw > 0 ? pw : 0, 20, 4, 4);
            pb->fill(100, 200, 100);
            canvas->add(pb);
        }

        canvas->draw();
        canvas->sync();
        // buf now has the rendered pixels — in production we blit this to GL

        free(buf);
        glfwSwapBuffers(app.window);
    }

    tvg::Initializer::term();
    glfwTerminate();
    return 0;
}
