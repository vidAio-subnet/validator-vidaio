#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

void cuda_check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        std::cerr << operation << " failed: " << cudaGetErrorString(result) << "\n";
        std::exit(70);
    }
}

__global__ void probe_kernel(std::uint8_t* byte) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *byte = static_cast<std::uint8_t>(*byte + 1);
    }
}

__device__ float channel_at(
    const std::uint8_t* input,
    int width,
    int height,
    int x,
    int y,
    int channel
) {
    x = max(0, min(width - 1, x));
    y = max(0, min(height - 1, y));
    return static_cast<float>(input[(y * width + x) * 3 + channel]);
}

__device__ float sample_bilinear(
    const std::uint8_t* input,
    int width,
    int height,
    float x,
    float y,
    int channel
) {
    const int x0 = static_cast<int>(floorf(x));
    const int y0 = static_cast<int>(floorf(y));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const float fx = x - static_cast<float>(x0);
    const float fy = y - static_cast<float>(y0);
    const float top = channel_at(input, width, height, x0, y0, channel) * (1.0f - fx)
        + channel_at(input, width, height, x1, y0, channel) * fx;
    const float bottom = channel_at(input, width, height, x0, y1, channel) * (1.0f - fx)
        + channel_at(input, width, height, x1, y1, channel) * fx;
    return top * (1.0f - fy) + bottom * fy;
}

__device__ float sample(
    const std::uint8_t* input,
    int width,
    int height,
    float x,
    float y,
    int channel,
    bool bilinear
) {
    if (bilinear) {
        return sample_bilinear(input, width, height, x, y, channel);
    }
    return channel_at(
        input,
        width,
        height,
        static_cast<int>(roundf(x)),
        static_cast<int>(roundf(y)),
        channel
    );
}

__global__ void transform_kernel(
    const std::uint8_t* input,
    std::uint8_t* output,
    int width,
    int height,
    int output_width,
    int output_height,
    bool bilinear,
    float sharpen
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= output_width || y >= output_height) {
        return;
    }
    const float source_x =
        (static_cast<float>(x) + 0.5f) * width / output_width - 0.5f;
    const float source_y =
        (static_cast<float>(y) + 0.5f) * height / output_height - 0.5f;
    for (int channel = 0; channel < 3; ++channel) {
        const float center = sample(
            input, width, height, source_x, source_y, channel, bilinear
        );
        const float neighbors =
            sample(input, width, height, source_x - 1.0f, source_y, channel, bilinear)
            + sample(input, width, height, source_x + 1.0f, source_y, channel, bilinear)
            + sample(input, width, height, source_x, source_y - 1.0f, channel, bilinear)
            + sample(input, width, height, source_x, source_y + 1.0f, channel, bilinear);
        const float value = center + sharpen * (4.0f * center - neighbors);
        output[(y * output_width + x) * 3 + channel] = static_cast<std::uint8_t>(
            fminf(255.0f, fmaxf(0.0f, value)) + 0.5f
        );
    }
}

int parse_positive(const char* value, const char* name) {
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed <= 0 || parsed > 16384) {
        std::cerr << "invalid " << name << "\n";
        std::exit(65);
    }
    return static_cast<int>(parsed);
}

float parse_sharpen(const char* value) {
    errno = 0;
    char* end = nullptr;
    const float parsed = std::strtof(value, &end);
    if (errno != 0 || end == value || *end != '\0' || !std::isfinite(parsed)
        || parsed < -0.25f || parsed > 0.5f) {
        std::cerr << "invalid sharpen value\n";
        std::exit(65);
    }
    return parsed;
}

int probe() {
    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count < 1) {
        std::cerr << "no CUDA device\n";
        return 69;
    }
    std::uint8_t initial = 41;
    std::uint8_t result = 0;
    std::uint8_t* device = nullptr;
    cuda_check(cudaMalloc(&device, 1), "cudaMalloc probe");
    cuda_check(cudaMemcpy(device, &initial, 1, cudaMemcpyHostToDevice), "probe upload");
    probe_kernel<<<1, 1>>>(device);
    cuda_check(cudaGetLastError(), "probe kernel launch");
    cuda_check(cudaDeviceSynchronize(), "probe kernel synchronize");
    cuda_check(cudaMemcpy(&result, device, 1, cudaMemcpyDeviceToHost), "probe download");
    cuda_check(cudaFree(device), "cudaFree probe");
    if (result != 42) {
        std::cerr << "CUDA probe returned wrong value\n";
        return 70;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--probe") == 0) {
        return probe();
    }
    if (argc != 7) {
        std::cerr << "usage: gpu_transform WIDTH HEIGHT OUTPUT_WIDTH OUTPUT_HEIGHT "
                     "nearest|bilinear SHARPEN\n";
        return 64;
    }
    const int width = parse_positive(argv[1], "width");
    const int height = parse_positive(argv[2], "height");
    const int output_width = parse_positive(argv[3], "output_width");
    const int output_height = parse_positive(argv[4], "output_height");
    const std::string mode(argv[5]);
    if (mode != "nearest" && mode != "bilinear") {
        std::cerr << "interpolation must be nearest or bilinear\n";
        return 65;
    }
    const float sharpen = parse_sharpen(argv[6]);

    const std::size_t input_bytes = static_cast<std::size_t>(width) * height * 3;
    const std::size_t output_bytes =
        static_cast<std::size_t>(output_width) * output_height * 3;
    if (input_bytes > (static_cast<std::size_t>(1) << 31)
        || output_bytes > (static_cast<std::size_t>(1) << 33)) {
        std::cerr << "frame allocation exceeds safety limit\n";
        return 65;
    }
    std::vector<std::uint8_t> input(input_bytes);
    std::vector<std::uint8_t> output(output_bytes);
    std::uint8_t* device_input = nullptr;
    std::uint8_t* device_output = nullptr;
    cuda_check(cudaMalloc(&device_input, input_bytes), "cudaMalloc input");
    cuda_check(cudaMalloc(&device_output, output_bytes), "cudaMalloc output");

    const dim3 threads(16, 16);
    const dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y
    );
    std::uint64_t frames = 0;
    while (true) {
        std::cin.read(reinterpret_cast<char*>(input.data()), input_bytes);
        const std::streamsize received = std::cin.gcount();
        if (received == 0 && std::cin.eof()) {
            break;
        }
        if (received != static_cast<std::streamsize>(input_bytes)) {
            std::cerr << "partial RGB frame\n";
            return 70;
        }
        cuda_check(
            cudaMemcpy(device_input, input.data(), input_bytes, cudaMemcpyHostToDevice),
            "frame upload"
        );
        transform_kernel<<<blocks, threads>>>(
            device_input,
            device_output,
            width,
            height,
            output_width,
            output_height,
            mode == "bilinear",
            sharpen
        );
        cuda_check(cudaGetLastError(), "transform kernel launch");
        cuda_check(cudaDeviceSynchronize(), "transform kernel synchronize");
        cuda_check(
            cudaMemcpy(output.data(), device_output, output_bytes, cudaMemcpyDeviceToHost),
            "frame download"
        );
        std::cout.write(reinterpret_cast<const char*>(output.data()), output_bytes);
        if (!std::cout) {
            std::cerr << "could not write transformed frame\n";
            return 70;
        }
        ++frames;
    }
    cuda_check(cudaFree(device_output), "cudaFree output");
    cuda_check(cudaFree(device_input), "cudaFree input");
    if (frames == 0) {
        std::cerr << "no decoded frames\n";
        return 70;
    }
    std::cerr << "vidaio-next CUDA frames=" << frames << "\n";
    return 0;
}
