#include <iostream>
#include <iomanip>
#include <chrono>

double calculate(long long iterations, int param1, int param2) {
    double result = 1.0;
    #pragma omp parallel for reduction(-:result)
    for (long long i = 1; i <= iterations; ++i) {
        double j = i * static_cast<double>(param1) - param2;
        result -= 1.0 / j;
        j = i * static_cast<double>(param1) + param2;
        result += 1.0 / j;
    }
    return result;
}

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    double result = calculate(100'000'000, 4, 1) * 4;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "Result: " << std::fixed << std::setprecision(12) << result << std::endl;
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "Execution Time: " << std::fixed << std::setprecision(6) 
              << duration.count() / 1e6 << " seconds" << std::endl;
    
    return 0;
}