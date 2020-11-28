#pragma once

#include "cxxabi.h"
#include <string>
#include <set>
#include <map>
#include "llvm/IR/Value.h"

// Obtain the original function name, without LLVM mangling;
inline std::string demangle_function_name(const char *mangled_name) {
    int status = 0;
    const char *demangled_name = abi::__cxa_demangle(mangled_name, NULL, NULL, &status);
    // If status != 0, the function name has not been unmangled: keep the original name;
    std::string demangled_name_str = status == 0 ? std::string(demangled_name) : std::string(mangled_name);
    // The unmangled name contains also the function parameters.
    // We obtain the original name by keeping all characters until "(" occurs;
    delete[] demangled_name;
    return demangled_name_str.substr(0, demangled_name_str.find('('));
}

template <typename T>
inline bool is_in_set(std::set<T *> *set, T &t) {
    return set->find(&t) != set->end();
}

template <typename T>
inline bool is_in_set(std::set<T> *set, T t) {
    return set->find(t) != set->end();
}

template <typename T, typename V>
inline bool is_in_map(std::map<T *, V *> *map, T &t) {
    return map->find(&t) != map->end();
}

template <typename T, typename V>
inline bool is_in_map(std::map<T, V> *map, T t) {
    return map->find(t) != map->end();
}

inline bool check_if_same_index(llvm::Value *x, llvm::Value *y) {
    bool same_index = x == y;
    // If the index refers to a load instruction, check the original values;
    if (!same_index) {
        llvm::Value *x_orig = x;
        llvm::Value *y_orig = y;
        if (auto loadI1 = llvm::dyn_cast<llvm::LoadInst>(x)) {
            x_orig = loadI1->getPointerOperand();
        }
        if (auto loadI2 = llvm::dyn_cast<llvm::LoadInst>(y)) {
            y_orig = loadI2->getPointerOperand();
        }
        same_index = x_orig == y_orig;
    }
    return same_index;
}