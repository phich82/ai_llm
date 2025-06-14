#include <iostream>
#include <iomanip>
#include <chrono>

double calculate(int64_t iterations, int64_t param1, int64_t param2) {
    double result = 1.0;
    #pragma omp parallel for reduction(-:result)
    for (int64_t i = 1; i <= iterations; ++i) {
        double j1 = i * param1 - param2;
        double j2 = i * param1 + param2;
        result -= (1.0 / j1);
        result += (1.0 / j2);
    }
    return result;
}

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    double result = calculate(100'000'000, 4, 1) * 4;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    std::cout << std::fixed << std::setprecision(12);
    std::cout << "Result: " << result << std::endl;
    std::cout << "Execution Time: " << duration.count() / 1e6 << " seconds" << std::endl;
    
    return 0;
}