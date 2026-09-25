/*
 * cJSON wrapper
 * Author: GPT-5.6 Luna
 */

#ifndef JSON_CONFIG_HPP
#define JSON_CONFIG_HPP

#include <cJSON.h>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>


class JsonConfig {
public:
    explicit JsonConfig(const std::string &filename)
        : root_(nullptr), filename_(filename) {
        load(filename);
    }

    ~JsonConfig() {
        if (root_ != nullptr) {
            cJSON_Delete(root_);
        }
    }

    // No copy
    JsonConfig(const JsonConfig &) = delete;
    JsonConfig &operator=(const JsonConfig &) = delete;

    // Move construction only
    JsonConfig(JsonConfig &&other) noexcept
        : root_(other.root_), filename_(std::move(other.filename_)) {
        other.root_ = nullptr;
    }

public:
    // ---------------------------------------------------------
    // Check whether a path exists
    // ---------------------------------------------------------

    bool has(const std::string &path) const {
        return find(path) != nullptr;
    }

    // ---------------------------------------------------------
    // Get scalar value
    //
    // Example:
    //
    // get<int>("base.nt")
    // get<float>("base.dt")
    // get<bool>("grid.PML")
    // get<std::string>("grid.fluid.K")
    // ---------------------------------------------------------

    template <typename T>
    T get(const std::string &path) const {
        cJSON *node = find(path);

        if (node == nullptr) {
            throw std::runtime_error(
                "JSON path not found: " + path);
        }

        return convert<T>(node, path);
    }

    // ---------------------------------------------------------
    // Get array element
    //
    // Example:
    //
    // get<int>("grid.src_pos.posx", i)
    // get<int>("grid.src_pos.posz", i)
    // ---------------------------------------------------------

    template <typename T>
    T get(const std::string &path, std::size_t index) const {
        cJSON *array = find(path);

        if (array == nullptr) {
            throw std::runtime_error(
                "JSON path not found: " + path);
        }

        if (!cJSON_IsArray(array)) {
            throw std::runtime_error(
                "JSON node is not an array: " + path);
        }

        const std::size_t n =
            static_cast<std::size_t>(cJSON_GetArraySize(array));

        if (index >= n) {
            throw std::runtime_error(
                "Array index out of range: " +
                path + "[" + std::to_string(index) + "]");
        }

        cJSON *node =
            cJSON_GetArrayItem(
                array,
                static_cast<int>(index));

        return convert<T>(
            node,
            path + "[" + std::to_string(index) + "]");
    }

    // ---------------------------------------------------------
    // Get array size
    // ---------------------------------------------------------

    std::size_t size(const std::string &path) const {
        cJSON *node = find(path);

        if (node == nullptr) {
            throw std::runtime_error(
                "JSON path not found: " + path);
        }

        if (!cJSON_IsArray(node)) {
            throw std::runtime_error(
                "JSON node is not an array: " + path);
        }

        return static_cast<std::size_t>(
            cJSON_GetArraySize(node));
    }

    // ---------------------------------------------------------
    // Get whole array
    //
    // Example:
    //
    // std::vector<int> posx =
    //     config.getArray<int>("grid.src_pos.posx");
    //
    // std::vector<float> values =
    //     config.getArray<float>("some.array");
    // ---------------------------------------------------------

    template <typename T>
    std::vector<T> getArray(const std::string &path) const {
        cJSON *array = find(path);

        if (array == nullptr) {
            throw std::runtime_error(
                "JSON path not found: " + path);
        }

        if (!cJSON_IsArray(array)) {
            throw std::runtime_error(
                "JSON node is not an array: " + path);
        }

        const int n = cJSON_GetArraySize(array);

        std::vector<T> result;
        result.reserve(n);

        for (int i = 0; i < n; ++i) {
            cJSON *node = cJSON_GetArrayItem(array, i);

            result.push_back(
                convert<T>(
                    node,
                    path + "[" + std::to_string(i) + "]"));
        }

        return result;
    }

private:
    cJSON *root_;
    std::string filename_;

private:
    // =========================================================
    // Load JSON
    // =========================================================

    void load(const std::string &filename) {
        std::ifstream file(filename.c_str());

        if (!file.is_open()) {
            throw std::runtime_error(
                "Cannot open JSON file: " + filename);
        }

        std::stringstream buffer;
        buffer << file.rdbuf();

        const std::string text = buffer.str();

        if (text.empty()) {
            throw std::runtime_error(
                "JSON file is empty: " + filename);
        }

        cJSON *parsed =
            cJSON_Parse(text.c_str());

        if (parsed == nullptr) {
            const char *error = cJSON_GetErrorPtr();

            std::string message =
                "Failed to parse JSON file: " + filename;

            if (error != nullptr) {
                message += "\nNear: ";
                message += error;
            }

            throw std::runtime_error(message);
        }

        root_ = parsed;
    }

    // =========================================================
    // Find node
    //
    // Supported:
    //
    // a.b.c
    // a.b[0]
    // a.b[0].c
    // a.b[0].c[3].d
    //
    // Object and array can be nested arbitrarily.
    // =========================================================

    cJSON *find(const std::string &path) const {
        if (root_ == nullptr) {
            return nullptr;
        }

        if (path.empty()) {
            return root_;
        }

        cJSON *current = root_;

        std::size_t pos = 0;

        while (pos < path.size()) {

            // -------------------------------------------------
            // Skip '.'
            // -------------------------------------------------

            if (path[pos] == '.') {
                ++pos;
                continue;
            }

            // -------------------------------------------------
            // Object key
            // -------------------------------------------------

            if (path[pos] != '[') {

                const std::size_t begin = pos;

                while (pos < path.size()
                       && path[pos] != '.'
                       && path[pos] != '[') {
                    ++pos;
                }

                const std::string key =
                    path.substr(begin, pos - begin);

                if (!cJSON_IsObject(current)) {
                    return nullptr;
                }

                current =
                    cJSON_GetObjectItemCaseSensitive(
                        current,
                        key.c_str());

                if (current == nullptr) {
                    return nullptr;
                }
            }

            // -------------------------------------------------
            // Array index
            // -------------------------------------------------

            while (pos < path.size()
                   && path[pos] == '[') {

                const std::size_t close =
                    path.find(']', pos);

                if (close == std::string::npos) {
                    return nullptr;
                }

                const std::string index_text =
                    path.substr(
                        pos + 1,
                        close - pos - 1);

                if (index_text.empty()) {
                    return nullptr;
                }

                char *end = nullptr;

                const long index =
                    std::strtol(
                        index_text.c_str(),
                        &end,
                        10);

                if (end == index_text.c_str()
                    || *end != '\0'
                    || index < 0) {
                    return nullptr;
                }

                if (!cJSON_IsArray(current)) {
                    return nullptr;
                }

                current =
                    cJSON_GetArrayItem(
                        current,
                        static_cast<int>(index));

                if (current == nullptr) {
                    return nullptr;
                }

                pos = close + 1;
            }

            if (pos < path.size()
                && path[pos] == '.') {
                ++pos;
            }
        }

        return current;
    }

    // =========================================================
    // Type conversion
    //
    // cJSON stores JSON numbers as double.
    // The float specialization converts valuedouble -> float.
    // =========================================================

    template <typename T>
    static T convert(
        cJSON *node,
        const std::string &path);
};


// =============================================================
// int
// =============================================================

template <>
inline int JsonConfig::convert<int>(
    cJSON *node,
    const std::string &path) {
    if (!cJSON_IsNumber(node)) {
        throw std::runtime_error(
            "JSON type mismatch at '" +
            path + "': expected int");
    }

    return node->valueint;
}


// =============================================================
// float
// =============================================================

template <>
inline float JsonConfig::convert<float>(
    cJSON *node,
    const std::string &path) {
    if (!cJSON_IsNumber(node)) {
        throw std::runtime_error(
            "JSON type mismatch at '" +
            path + "': expected float");
    }

    return static_cast<float>(
        node->valuedouble);
}


// =============================================================
// bool
// =============================================================

template <>
inline bool JsonConfig::convert<bool>(
    cJSON *node,
    const std::string &path) {
    if (!cJSON_IsBool(node)) {
        throw std::runtime_error(
            "JSON type mismatch at '" +
            path + "': expected bool");
    }

    return cJSON_IsTrue(node);
}


// =============================================================
// std::string
// =============================================================

template <>
inline std::string JsonConfig::convert<std::string>(
    cJSON *node,
    const std::string &path) {
    if (!cJSON_IsString(node)) {
        throw std::runtime_error(
            "JSON type mismatch at '" +
            path + "': expected string");
    }

    return std::string(node->valuestring);
}


// =============================================================
// std::vector<int>
// =============================================================

template <>
inline std::vector<int> JsonConfig::convert<std::vector<int>>(
    cJSON *node,
    const std::string &path) {
    if (!cJSON_IsArray(node)) {
        throw std::runtime_error(
            "JSON type mismatch at '" +
            path + "': expected int array");
    }

    const int n = cJSON_GetArraySize(node);

    std::vector<int> result;
    result.reserve(n);

    for (int i = 0; i < n; ++i) {
        cJSON *item = cJSON_GetArrayItem(node, i);

        result.push_back(
            convert<int>(
                item,
                path + "[" + std::to_string(i) + "]"));
    }

    return result;
}


// =============================================================
// std::vector<float>
// =============================================================

template <>
inline std::vector<float> JsonConfig::convert<std::vector<float>>(
    cJSON *node,
    const std::string &path) {
    if (!cJSON_IsArray(node)) {
        throw std::runtime_error(
            "JSON type mismatch at '" +
            path + "': expected float array");
    }

    const int n = cJSON_GetArraySize(node);

    std::vector<float> result;
    result.reserve(n);

    for (int i = 0; i < n; ++i) {
        cJSON *item = cJSON_GetArrayItem(node, i);

        result.push_back(
            convert<float>(
                item,
                path + "[" + std::to_string(i) + "]"));
    }

    return result;
}


// =============================================================
// std::vector<bool>
// =============================================================

template <>
inline std::vector<bool> JsonConfig::convert<std::vector<bool>>(
    cJSON *node,
    const std::string &path) {
    if (!cJSON_IsArray(node)) {
        throw std::runtime_error(
            "JSON type mismatch at '" +
            path + "': expected bool array");
    }

    const int n = cJSON_GetArraySize(node);

    std::vector<bool> result;
    result.reserve(n);

    for (int i = 0; i < n; ++i) {
        cJSON *item = cJSON_GetArrayItem(node, i);

        result.push_back(
            convert<bool>(
                item,
                path + "[" + std::to_string(i) + "]"));
    }

    return result;
}


// =============================================================
// std::vector<std::string>
// =============================================================

template <>
inline std::vector<std::string>
JsonConfig::convert<std::vector<std::string>>(
    cJSON *node,
    const std::string &path) {
    if (!cJSON_IsArray(node)) {
        throw std::runtime_error(
            "JSON type mismatch at '" +
            path + "': expected string array");
    }

    const int n = cJSON_GetArraySize(node);

    std::vector<std::string> result;
    result.reserve(n);

    for (int i = 0; i < n; ++i) {
        cJSON *item = cJSON_GetArrayItem(node, i);

        result.push_back(
            convert<std::string>(
                item,
                path + "[" + std::to_string(i) + "]"));
    }

    return result;
}


#endif // JSON_CONFIG_HPP