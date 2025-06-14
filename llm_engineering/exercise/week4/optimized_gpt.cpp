#include <iostream>
#include <iomanip>
#include <chrono>

double calculate(int iterations, int param1, int param2) {
    double result = 1.0;
    for (int i = 1; i <= iterations; ++i) {
        int64_t ip = static_cast<int64_t>(i) * param1;
        result -= 1.0 / (ip - param2);
        result += 1.0 / (ip + param2);
    }
    return result;
}

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();

    double result = calculate(100000000, 4, 1) * 4.0;

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> exec_time = end_time - start_time;

    std::cout << "Result: " << std::fixed << std::setprecision(12) << result << std::endl;
    std::cout << "Execution Time: " << std::fixed << std::setprecision(6)
              << exec_time.count() << " seconds" << std::endl;

    return 0;
}