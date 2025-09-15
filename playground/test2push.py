// ---- Read/Write factories for Eigen types ----
template<typename EigenVec>
static FieldHandle make_eigen_vector_field_rw(EigenVec bdr_rtc_config::* mem,
                                              const char* tn = "vector<double>") {
    return FieldHandle{
        tn,
        // Getter -> JSON
        [mem]() -> nlohmann::json {
            std::lock_guard<std::mutex> lk(rtc_mutex);
            return eigen_vector_to_json(rtc_config.*mem);
        },
        // Setter <- JSON
        [mem](const nlohmann::json& j) -> bool {
            try {
                EigenVec tmp;
                if (!json_to_eigen_vector(j, tmp)) return false;
                std::lock_guard<std::mutex> lk(rtc_mutex);
                (rtc_config.*mem) = std::move(tmp);
                return true;
            } catch (...) {
                return false;
            }
        }
    };
}

template<typename EigenMat>
static FieldHandle make_eigen_matrix_field_rw(EigenMat bdr_rtc_config::* mem,
                                              const char* tn = "matrix<double>") {
    return FieldHandle{
        tn,
        // Getter -> JSON
        [mem]() -> nlohmann::json {
            std::lock_guard<std::mutex> lk(rtc_mutex);
            return eigen_matrix_to_json(rtc_config.*mem);
        },
        // Setter <- JSON
        [mem](const nlohmann::json& j) -> bool {
            try {
                EigenMat tmp;
                if (!json_to_eigen_matrix(j, tmp)) return false;
                std::lock_guard<std::mutex> lk(rtc_mutex);
                (rtc_config.*mem) = std::move(tmp);
                return true;
            } catch (...) {
                return false;
            }
        }
    };
}