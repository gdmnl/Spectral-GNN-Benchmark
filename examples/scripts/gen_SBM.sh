g++ -std=c++17 ../gen_SBM/gen_SBM.cpp -o rd_dynamic
./rd_dynamic -n 5000 -c 50 -ind 20 -outd 1
python ../gen_SBM/gen_feature.py --n 5000