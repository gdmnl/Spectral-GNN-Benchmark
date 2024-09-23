g++ -std=c++17 ./gen_SBM.cpp -o log
./log -n 5000 -c 50 -ind 20 -outd 1
python ./gen_feature.py --n 5000