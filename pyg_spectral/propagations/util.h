/*
 * Author: nyLiao
 * File Created: 2024-06-12
 */
#ifndef UTIL_H
#define UTIL_H

#include <unistd.h>
#include <fstream>
#include <sstream>
#include <chrono>
#include <sys/time.h>
#include <sys/resource.h>

double get_curr_time() {
    long long time = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    return static_cast<double>(time) / 1000000.0;
}

float get_proc_memory(){
    struct rusage r_usage;
    getrusage(RUSAGE_SELF,&r_usage);
    return r_usage.ru_maxrss/1000000.0;
}

float get_stat_memory(){
    long rss;
    std::string ignore;
    std::ifstream ifs("/proc/self/stat", std::ios_base::in);
    ifs >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
        >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
        >> ignore >> ignore >> ignore >> rss;

    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024;
    return rss * page_size_kb / 1000000.0;
}

inline void update_maxr(const float r, float &maxrp, float &maxrn) {
    if (r > maxrp)
        maxrp = r;
    else if (r < maxrn)
        maxrn = r;
}

#endif // UTIL_H
