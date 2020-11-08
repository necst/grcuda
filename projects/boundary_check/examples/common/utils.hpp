#pragma once

#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <random>
#include <time.h> /* time */
#include <set>

#define DEBUG 1

// Error check;
#undef cudaCheckError
#ifdef DEBUG
#define WHERE " at: " << __FILE__ << ':' << __LINE__
#define cudaCheckError()                                                    \
    {                                                                       \
        cudaError_t e = cudaGetLastError();                                 \
        if (e != cudaSuccess) {                                             \
            std::cerr << "Cuda failure: " << cudaGetErrorString(e) << WHERE \
                      << std::endl;                                         \
        }                                                                   \
    }
#else
#define cudaCheckError()
#define WHERE ""
#endif

///////////////////////////////
///////////////////////////////

template <typename T>
inline void print_array_indexed(T *v, int n) {
    for (int i = 0; i < n; i++) {
        std::cout << i << ") " << v[i] << std::endl;
    }
    std::cout << std::endl;
}

template <typename T>
inline void print_matrix_indexed(T *m, int dim_col, int dim_row, int max_r = 4, int max_c = 4) {
    // Print 2 decimal digits;
    std::cout << std::setprecision(6) << std::fixed;
    // Save the old flags to restore them later;
    std::ios::fmtflags old_settings = std::cout.flags();

    std::cout << "[" << std::endl;

    for (int c = 0; c < std::min(max_r, dim_col); c++) {
        std::cout << "[";
        for (int r = 0; r < std::min(max_c, dim_row); r++) {
            std::cout << m[dim_row * c + r] << (r < dim_row - 1 ? ", " : "");
        }
        std::cout << "]" << (c < dim_col - 1 ? "," : "") << std::endl;
    }

    std::cout << "]" << std::endl;
    // Reset printing format;
    std::cout.flags(old_settings);
}

template <typename T>
inline int check_array_equality(T *x, T *y, int n, float tol = 0.0000001f, bool human_readable = false, int max_print = 20) {
    int num_errors = 0;
    for (int i = 0; i < n; i++) {
        float diff = std::abs(x[i] - y[i]);
        if (diff > tol) {
            num_errors++;
            if (human_readable && num_errors < max_print) {
                std::cout << i << ") X: " << x[i] << ", Y: " << y[i] << ", diff: " << diff << std::endl;
            }
        }
    }
    return num_errors;
}

template <typename T>
inline bool check_equality(T x, T y, float tol = 0.0000001f, bool human_readable = false) {
    bool equal = true;

    float diff = std::abs(x - y);
    if (diff > tol) {
        equal = false;
        if (human_readable) {
            std::cout << "x: " << x << ", y: " << y << ", diff: " << diff << std::endl;
        }
    }
    return equal;
}

inline void create_sample_vector(float *vector, int size, bool random = false, bool normalize = true) {

    if (random) {
        std::random_device rd;
        std::mt19937 engine(rd());
        std::uniform_real_distribution<float> dist(0, 1);
        for (int i = 0; i < size; i++) {
            vector[i] = dist(engine);
        }
    } else {
        for (int i = 0; i < size; i++) {
            vector[i] = 1;
        }
    }

    if (normalize) {
        float sum = 0;
        for (int i = 0; i < size; i++) {
            sum += vector[i];
        }
        for (int i = 0; i < size; i++) {
            vector[i] /= sum;
        }
    }
}

inline void normalize_vector(float *vector, int size) {
    float mean = 0;
    for (int i = 0; i < size; i++) {
        mean += vector[i];
    }
    mean /= size;
    for (int i = 0; i < size; i++) {
        vector[i] -= mean;
    }
}

inline void create_random_graph(std::vector<int> &ptr, std::vector<int> &idx, int max_degree = 10, bool avoid_self_edges = true) {
    srand(time(NULL));
    int N = ptr.size() - 1;
    // For each vertex, generate a random number of edges, with a given max degree; 
    for (int v = 1; v < ptr.size(); v++) {
        int num_edges = rand() % std::min(N, max_degree);
        // Generate edges;
        std::set<int> edge_set;
        for (int e = 0; e < num_edges; e++) {
            edge_set.insert(rand() % N);
        }
        // Avoid self-edges;
        if (avoid_self_edges ) {
            edge_set.erase(v - 1);
        }
        for (auto e : edge_set) {  
            idx.push_back(e); 
        }
        ptr[v] = edge_set.size() + ptr[v - 1];
    }
}

inline void print_graph(std::vector<int> &ptr, std::vector<int> &idx, int max_N = 20, int max_E = 20) {
    std::cout << "-) degree: " << ptr[0] << std::endl;
    for (int v = 1; v < std::min((int) ptr.size(), max_N); v++) {
        std::cout << v - 1 << ") degree: " << ptr[v] - ptr[v - 1] << ", edges: ";
        for (int e = 0; e < ptr[v] - ptr[v - 1]; e++) {
            if (e < max_E) {
                std::cout << idx[ptr[v - 1] + e] << ", ";
            }
        }
        std::cout << std::endl;
    }
}
