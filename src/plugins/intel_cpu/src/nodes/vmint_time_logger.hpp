// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>
#include <vector>
#include <iostream>
#include <iomanip>

namespace vmint {

// ====================================================================================================================================
//
using namespace std::chrono;
class TimeLogger {
    struct Record {
        explicit Record(const char* label) : pt_created_{steady_clock::now()}, label_{label} {}
        void finalize() { pt_finalized_ = steady_clock::now(); }

        steady_clock::time_point pt_created_;
        steady_clock::time_point pt_finalized_;
        std::string label_;
    };

    bool enabled_;
    Record origin_;
    std::vector<Record> records_;

public:
    TimeLogger(const char* label = "TimeLogger", bool enabled = true) : enabled_{enabled}, origin_{label} { origin_.finalize(); }
    void add_record(const char* label) {
        if (!enabled_) return;
        records_.emplace_back(label);
        records_.back().finalize();
    }
    void log_records() const {
        if (!enabled_) return;
        std::cout << std::setfill(' ');
        std::cout << "-- [" << origin_.label_ << "] -------------------------------------------------" << std::endl;
        size_t label_width = 0;
        for (const Record& r : records_) {
            label_width = std::max(label_width, r.label_.size());
        }
        const Record* r_prev = &origin_;
        size_t duration_total {};
        for (const Record& r : records_) {
            auto duration = duration_cast<microseconds>(r.pt_created_ - r_prev->pt_finalized_).count();
            std::cout << "| " << std::setw(label_width) << r.label_ << ": "
                        << "duration: \t" << duration << " us"
                        << ".\n";
            duration_total += duration;
            r_prev = &r;
        }
        std::cout << "| " << std::setw(label_width) << "TIME TOTAL" << ": "
                    << "duration: \t" << duration_total << " us"
                    << ".\n";
    }
};

// ====================================================================================================================================
//
template <typename T0, typename F>
void parallel_for(const T0& D0, const F& func) {
    for (T0 i0 = 0; i0 < D0; ++i0)
        func(i0);
}

template <typename T0, typename T1, typename F>
void parallel_for2d(const T0& D0, const T1& D1, const F& func) {
    for (T0 i0 = 0; i0 < D0; ++i0)
        for (T1 i1 = 0; i1 < D1; ++i1)
            func(i0, i1);
}

} // namespace vmint
