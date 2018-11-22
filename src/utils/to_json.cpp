//
// Created by liang on 7/28/18.
//

#include <utils/to_json.h>
//#include <utils/app_skeleton.h>

DECLARE_string(json);

namespace utils
{
    JsonWriter::JsonWriter()
    {
        if (!FLAGS_json.empty())
        {
            o = std::ofstream(FLAGS_json.data());
        }
    }

    JsonWriter::~JsonWriter()
    {
        if (o.is_open())
        {
            o << std::setw(4) << j << std::endl;
            o.close();
        }
    }

    JsonWriter &JsonWriter::getInst()
    {
        static JsonWriter inst;

        return inst;
    }

    void JsonWriter::write(const std::string &key, float value)
    {
        j[key] = value;
    }

    void JsonWriter::write(const std::string &key, int value)
    {
        j[key] = value;
    }

    void JsonWriter::write(const std::string &key, std::string value)
    {
        j[key] = value;
    }

    void JsonWriter::write(const std::string &key, std::vector<std::string> &value)
    {
        j[key] = value;
    }

    float JsonWriter::get_float(const std::string &key)
    {
        return j.value(key, 0.0f);
    }
}
