//
// Created by liang on 7/28/18.
//
#include <fstream>
#include <gflags/gflags.h>
#include <nlohmann/json.hpp>

#ifndef HYBRID_TO_JSON_H
#define HYBRID_TO_JSON_H
namespace utils
{
    using json = nlohmann::json;

    class JsonWriter
    {
    private:
        std::ofstream o;
        json j;

        JsonWriter();

        ~JsonWriter();

    public:
        JsonWriter(const JsonWriter &other) = delete;

        void operator=(const JsonWriter &other) = delete;

        static JsonWriter &getInst();

        void write(const std::string &key, float value);

        void write(const std::string &key, int value);

        void write(const std::string &key, std::string value);

        void write(const std::string &key, std::vector<std::string> &value);
        
        float get_float(const std::string &key);
    };
}
#endif //HYBRID_TO_JSON_H
