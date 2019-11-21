#pragma once

#include "cxxabi.h"
#include <string>
#include <set>
#include <map>

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