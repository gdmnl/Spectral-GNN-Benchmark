/*
 * Author: nyLiao
 * File Created: 2023-04-19
 * Ref: [AGP](https://github.com/wanghzccls/AGP-Approximate_Graph_Propagation)
 */
#include <thread>
#include <omp.h>

#include "prop.h"
#include "util.h"
#include "alg/unifews.h"

using namespace std;

namespace propagation {

// Load graph related data
void PropComp::load(string dataset, uint mm, uint nn, uint seedd) {
    m = mm;
    n = nn;
    seed = seedd;

    // Load graph adjacency
    el = vector<uint>(m);   // edge list sorted by source node degree
    pl = vector<uint>(n + 1);
    string dataset_el = dataset + "/adj_el.bin";
    const char *p1 = dataset_el.c_str();
    if (FILE *f1 = fopen(p1, "rb")) {
        size_t rtn = fread(el.data(), sizeof el[0], el.size(), f1);
        if (rtn != m)
            cout << "Error! " << dataset_el << " Incorrect read!" << endl;
        fclose(f1);
    } else {
        cout << dataset_el << " Not Exists." << endl;
        exit(1);
    }
    string dataset_pl = dataset + "/adj_pl.bin";
    const char *p2 = dataset_pl.c_str();
    if (FILE *f2 = fopen(p2, "rb")) {
        size_t rtn = fread(pl.data(), sizeof pl[0], pl.size(), f2);
        if (rtn != n + 1)
            cout << "Error! " << dataset_pl << " Incorrect read!" << endl;
        fclose(f2);
    } else {
        cout << dataset_pl << " Not Exists." << endl;
        exit(1);
    }

    deg = Eigen::ArrayXf::Zero(n);
    for (uint i = 0; i < n; i++) {
        deg(i)   = pl[i + 1] - pl[i];
        if (deg(i) <= 0) {
            deg(i) = 1;
            // cout << i << " ";
        }
    }
}

// Computation call entry
float PropComp::compute(uint nchnn, Channel* chnss, Eigen::Map<Eigen::MatrixXf> &feat) {
    // Node-specific array
    chns = chnss;
    assert(nchnn <= 4);
    dega = Eigen::ArrayXf::Zero(n);
    dinva = Eigen::ArrayXf::Zero(n);
    dinvb = Eigen::ArrayXf::Zero(n);
    for (uint c = 0; c < nchnn; c++) {
        dega = deg.pow(chns[c].rra);
        dinva = 1 / dega;
        dinvb = 1 / deg.pow(chns[c].rrb);
    }

    // Feat is ColMajor, shape: (n, c*F)
    int fsum = feat.cols();
    int it = 0;
    map_feat = Eigen::ArrayXf::LinSpaced(fsum, 0, fsum - 1);
    // random_shuffle(map_feat.data(), map_feat.data() + map_feat.size());
    cout << "feat dim: " << feat.cols() << ", nodes: " << feat.rows() << ", edges: " << m <<  ". ";

    // Feature-specific array
    dlt_p = Eigen::ArrayXf::Zero(fsum);
    dlt_n = Eigen::ArrayXf::Zero(fsum);
    maxf_p = Eigen::ArrayXf::Zero(fsum);
    maxf_n = Eigen::ArrayXf::Zero(fsum);
    map_chn = Eigen::ArrayXi::Zero(fsum);
    // Loop each feature index `it`, inside channel index `i`
    for (uint c = 0; c < nchnn; c++) {
        for (int i = 0; i < chns[c].dim; i++) {
            for (uint u = 0; u < n; u++) {
                if (feat(u, i) > 0)
                    dlt_p(it) += feat(u, it) * pow(deg(u), chns[c].rrb);
                else
                    dlt_n(it) += feat(u, it) * pow(deg(u), chns[c].rrb);
                update_maxr(feat(u, it), maxf_p(it), maxf_n(it));
            }
            if (dlt_p(it) == 0)
                dlt_p(it) = 1e-12;
            if (dlt_n(it) == 0)
                dlt_n(it) = -1e-12;
            dlt_p(it) *= chns[c].delta / (1 - chns[c].alpha);
            dlt_n(it) *= chns[c].delta / (1 - chns[c].alpha);
            map_chn(it) = c;
            it++;
        }
    }

    // Begin propagation
    cout << "Propagating..." << endl;
    struct timeval ttod_start, ttod_end;
    double ttod, tclk;
    gettimeofday(&ttod_start, NULL);
    tclk = get_curr_time();
    int dim_top = 0;
    int start, ends = dim_top;

    vector<thread> threads;
    for (it = 1; it <= fsum % NUMTHREAD; it++) {
        start = ends;
        ends += ceil((float)fsum / NUMTHREAD);
        threads.push_back(thread(&PropComp::alg_unifews, this, feat, start, ends));
    }
    for (; it <= NUMTHREAD; it++) {
        start = ends;
        ends += fsum / NUMTHREAD;
        threads.push_back(thread(&PropComp::alg_unifews, this, feat, start, ends));
    }
    for (int t = 0; t < NUMTHREAD; t++)
        threads[t].join();
    vector<thread>().swap(threads);

    tclk = get_curr_time() - tclk;
    gettimeofday(&ttod_end, NULL);
    ttod = ttod_end.tv_sec - ttod_start.tv_sec + (ttod_end.tv_usec - ttod_start.tv_usec) / 1000000.0;
    cout << "[Pre] Prop  time: " << ttod << " s, ";
    cout << "Clock time: " << tclk << " s, ";
    cout << "Max   PRAM: " << get_proc_memory() << " GB, ";
    cout << "End    RAM: " << get_stat_memory() << " GB, ";
    return ttod;
}

} // namespace propagation
