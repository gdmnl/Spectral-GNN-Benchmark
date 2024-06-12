/*
 * Author: nyLiao
 * File Created: 2023-04-19
 */
#include "prop.h"
#include "util.h"

namespace propagation {

void PropComp::alg_unifews(Eigen::Ref<Eigen::MatrixXf> feats, int st, int ed) {
    uint seedt = seed;
    Eigen::VectorXf res0(n), res1(n);
    Eigen::Map<Eigen::VectorXf> rprev(res1.data(), n), rcurr(res0.data(), n);

    // Loop each feature `ift`, index `it`
    for (int it = st; it < ed; it++) {
        const uint ift = map_feat(it);
        const Channel chn = chns[0];
        const float alpha = chn.alpha;
        std::vector<uint> plshort(pl), plshort2(pl);
        Eigen::Map<Eigen::VectorXf> feati(feats.col(ift).data(), n);

        const float dlti_p = dlt_p(ift);
        const float dlti_n = dlt_n(ift);
        const float dltinv_p = 1 / dlti_p;
        const float dltinv_n = 1 / dlti_n;
        float maxr_p = maxf_p(ift);     // max positive residue
        float maxr_n = maxf_n(ift);     // max negative residue
        uint maccnt = 0;

        // Init residue
        res1.setZero();
        res0 = feats.col(ift);
        feati.setZero();
        rprev = res1;
        rcurr = res0;

        // Loop each hop `il`
        int il;
        for (il = 0; il < chn.hop; il++) {
            // Early termination
            if ((maxr_p <= dlti_p) && (maxr_n >= dlti_n))
                break;
            rcurr.swap(rprev);
            rcurr.setZero();

            // Loop each node `u`
            for (uint u = 0; u < n; u++) {
                const float old = rprev(u);
                float thr_p = old * dltinv_p;
                float thr_n = old * dltinv_n;
                // if ((!chn.is_acc) && (m < 1e9)) {
                if (!chn.is_acc) {
                    rcurr(u) += old;
                }
                if (thr_p > 1 || thr_n > 1) {
                    float oldb = 0;
                    if (chn.is_acc) {
                        feati(u) += old * alpha;
                        oldb = old * (1-alpha) * dinvb(u);
                    }

                    // Loop each neighbor index `im`, node `v`
                    uint iv, iv2;
                    const uint ivmax = (chn.is_thr) ? plshort[u+1] : pl[u+1];
                    for (iv = pl[u]; iv < ivmax; iv++) {
                        const uint v = el[iv];
                        const float da_v = dega(v);
                        if (thr_p > da_v || thr_n > da_v) {
                            maccnt++;
                            if (chn.is_acc)
                                rcurr(v) += oldb * dinva(v);
                            else
                                rcurr(v) += old / deg(v);
                            update_maxr(rcurr(v), maxr_p, maxr_n);
                        } else {
                            // plshort[u+1] = iv;
                            break;
                        }
                    }

                    iv2 = iv;
                    const float ran = (float)RAND_MAX / (rand_r(&seedt) % RAND_MAX);
                    thr_p *= ran;
                    thr_n *= ran;
                    const uint ivmax2 = (chn.is_thr) ? plshort2[u+1] : pl[u+1];
                    for (; iv < ivmax2; iv++) {
                        const uint v = el[iv];
                        const float da_v = dega(v);
                        if (thr_p > da_v) {
                            maccnt++;
                            rcurr(v) += dlti_p * dinva(v);
                            update_maxr(rcurr(v), maxr_p, maxr_n);
                        } else if (thr_n > da_v) {
                            maccnt++;
                            rcurr(v) += dlti_n * dinva(v);
                            update_maxr(rcurr(v), maxr_p, maxr_n);
                        } else {
                            break;
                        }
                    }
                    plshort[u+1]  = (iv + iv2) / 2;
                    plshort2[u+1] = (iv + pl[u+1]) / 2;

                } else {
                    if (chn.is_acc)
                        feati(u) += old;
                }
            }
        }

        feati += rcurr;
    }
}

} // namespace propagation
