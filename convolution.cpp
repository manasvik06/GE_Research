/*
 * Direct 2D Convolution in C++
 * Author: Manasvi
 *
 * This program implements standard spatial-domain convolution using
 * nested loops and benchmarks runtime on a few image sizes.
 *
 * Compile:
 * g++ -O2 -std=c++17 convolution.cpp -o conv
 */

#include <iostream>
#include <vector>
#include <chrono>

using Image = std::vector<std::vector<double>>;


/*
 * Create a rows x cols image initialized to `fill`.
 */
Image create_image(int rows, int cols, double fill = 0.0) {
    return Image(rows, std::vector<double>(cols, fill));
}


/*
 * Create a normalized box blur kernel.
 *
 * Every value is the same, so each output pixel becomes the average
 * of its local neighborhood.
 */
Image make_box_kernel(int size) {
    double value = 1.0 / (size * size);
    return Image(size, std::vector<double>(size, value));
}


/*
 * Same-size direct 2D convolution with zero-padding at boundaries.
 *
 * Complexity:
 * O(N^2 * K^2)
 *
 * N = image dimension
 * K = kernel dimension
 */
Image direct_convolve2d(const Image& image, const Image& kernel) {
    int img_h = image.size();
    int img_w = image[0].size();

    int ker_h = kernel.size();
    int ker_w = kernel[0].size();

    int pad_h = ker_h / 2;
    int pad_w = ker_w / 2;

    Image output = create_image(img_h, img_w);

    // For each output pixel
    for (int i = 0; i < img_h; ++i) {
        for (int j = 0; j < img_w; ++j) {

            double sum = 0.0;

            // Slide across kernel window
            for (int ki = 0; ki < ker_h; ++ki) {
                for (int kj = 0; kj < ker_w; ++kj) {

                    int img_i = i + ki - pad_h;
                    int img_j = j + kj - pad_w;

                    // Zero-padding: ignore out-of-bounds locations
                    if (img_i >= 0 && img_i < img_h &&
                        img_j >= 0 && img_j < img_w) {
                        sum += image[img_i][img_j] * kernel[ki][kj];
                    }
                }
            }

            output[i][j] = sum;
        }
    }

    return output;
}


/*
 * Generate a simple deterministic test image.
 *
 * This avoids file loading and gives repeatable values.
 */
Image generate_test_image(int size) {
    Image image = create_image(size, size);

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            image[i][j] = static_cast<double>((i * j) % 256) / 255.0;
        }
    }

    return image;
}


int main() {
    std::vector<int> image_sizes = {64, 128, 256};
    int kernel_size = 15;

    std::cout << "Direct 2D Convolution Benchmark (C++)\n";
    std::cout << "Kernel size: "
              << kernel_size << "x" << kernel_size << "\n\n";

    Image kernel = make_box_kernel(kernel_size);

    for (int size : image_sizes) {
        Image image = generate_test_image(size);

        auto start = std::chrono::high_resolution_clock::now();

        Image result = direct_convolve2d(image, kernel);

        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;

        std::cout << size << "x" << size
                  << " image -> "
                  << elapsed.count()
                  << " seconds\n";
    }

    return 0;
}