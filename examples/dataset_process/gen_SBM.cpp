#include <stdio.h>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <functional>
#include <time.h>
#include <vector>
#include <string.h>
#include <filesystem>


namespace fs = std::filesystem;
using namespace std;

bool cmp(const int& a,const int&b){
	return a<b;
}

//Check parameters
long check_inc(long i, long max) {
    if (i == max) {
        //usage();
        cout<<"i==max"<<endl;
	exit(1);
    }
    return i + 1;
}



int main(int argc,char **argv){
    //srand(time(NULL));
    srand(20);
    char *endptr;
    uint vert=10000;
    int cluster=2;
    double in_com=1;
    double between_com=1;
    int in_degree = 20;
    int out_degree = 5;
    uint changeNum = 5;
    int snapeNum=0; 
    int i=1;
    // Sparsity factor (approximate percentage of non-zero entries)
    double sparsity = 0.05; // 5% non-zero
    int dimension = 128; //node feature dimension
    while (i < argc) {
        if (!strcmp(argv[i], "-n")) {
            i = check_inc(i, argc);
            vert = strtod(argv[i],&endptr);
        } else if (!strcmp(argv[i], "-c")) {
            i = check_inc(i, argc);
            cluster = strtod(argv[i],&endptr);
        } else if (!strcmp(argv[i], "-ind")) {
            i = check_inc(i, argc);
            in_degree = strtod(argv[i],&endptr);
        } else if (!strcmp(argv[i], "-outd")) {
            i = check_inc(i, argc);
            out_degree = strtod(argv[i], &endptr);
        } else if (!strcmp(argv[i], "-inp")) {
            i = check_inc(i, argc);
            in_com = strtod(argv[i],&endptr);
        } else if (!strcmp(argv[i], "-outp")) {
            i = check_inc(i, argc);
            between_com = strtod(argv[i], &endptr);
        }else if (!strcmp(argv[i], "-change")) {
            i = check_inc(i, argc);
            changeNum = strtod(argv[i], &endptr);
        }else if (!strcmp(argv[i], "-snap")) {
            i = check_inc(i, argc);
            snapeNum = strtod(argv[i], &endptr);
        }else if (!strcmp(argv[i], "-sparsity")) {
            i = check_inc(i, argc);
            sparsity = strtod(argv[i], &endptr);
        }else if (!strcmp(argv[i], "-dimension")) {
            i = check_inc(i, argc);
            dimension = strtod(argv[i], &endptr);
        } else {
            cout<<"ERROR parameter!!!"<<endl;
            exit(1);
        }
        i++;
    }
    // Specify the path of the directory you want to check and create
    fs::path dir_path = "../../data/SBM";

    // Check if directory exists
    if (!fs::exists(dir_path)) {
        std::cout << "Directory does not exist, creating now..." << std::endl;
        
        // Try to create the directory
        if (fs::create_directory(dir_path)) {
            std::cout << "Directory created successfully." << std::endl;
        } else {
            std::cerr << "Failed to create directory." << std::endl;
        }
    } else {
        std::cout << "Directory already exists." << std::endl;
    }
    uint N_perCluster=vert/cluster;
    if(in_com<1 && between_com<1){
        in_degree = in_com*N_perCluster;
        out_degree = between_com*N_perCluster;
    }
    
    cout<<"vert="<<vert<<endl;
    cout<<"cluster="<<cluster<<endl;
    cout<<"in_com="<<in_com<<endl;
    cout<<"between_com="<<between_com<<endl;
    cout<<"in_degree="<<in_degree<<endl;
    cout<<"out_degree="<<out_degree<<endl;
    cout<<"N_perCluster="<<N_perCluster<<endl;
    cout<<"snapNum="<<snapeNum<<endl;
    cout<<"sparsity="<<sparsity<<endl;
    cout<<"dimension="<<dimension<<endl;

    int *clusterID=new int[vert];
    for(uint i=0;i<vert;i++){
        int clusterFlag=i/N_perCluster;
        if(clusterFlag>=cluster){
            clusterFlag=cluster-1;
        }
        clusterID[i]=clusterFlag;
    }
    

    stringstream label_out;
    label_out<<"../../data/SBM/SBM-"<<vert<<"-"<<cluster<<"-"<<in_degree<<"+"<<out_degree<<"_label.txt";
    cout<<label_out.str()<<endl;
    ofstream f1;
    f1.open(label_out.str());
    for(uint i=0;i<vert;i++){
        f1<<clusterID[i]<<"\n";
    }
    f1.close();

    stringstream ss_out;
    ss_out<<"../../data/SBM/SBM-"<<vert<<"-"<<cluster<<"-"<<in_degree<<"+"<<out_degree<<"_init.txt";
    cout<<ss_out.str()<<endl;

    ofstream fout;
    fout.open(ss_out.str());
    if(!fout){
        cout<<"ERROR:can not open out file"<<endl;
        return 0;
    }
    vector<vector<uint>> Adj;
    vector<vector<uint>> out_Adj;
    vector<uint> random_w = vector<uint>(vert);

    for (uint i = 0; i < vert; i++)
    {
        vector<uint> templst;
        Adj.push_back(templst);
        out_Adj.push_back(templst);
        random_w[i] = i;
    }
    random_shuffle(random_w.begin(),random_w.end());
    
    for(uint i=0;i<vert;i++){
        uint w = random_w[i];
        int di = Adj[w].size();
        for(int j=0;j<(in_degree - di);j++){
            uint tmp_node=rand()%N_perCluster;
            tmp_node+=clusterID[w]*N_perCluster;
            while(find(Adj[w].begin(),Adj[w].end(),tmp_node)!=Adj[w].end() || tmp_node==w )
            {
                tmp_node=rand()%N_perCluster;
                tmp_node+=clusterID[w]*N_perCluster;
            }

            Adj[w].push_back(tmp_node);
            if( find(Adj[tmp_node].begin(),Adj[tmp_node].end(),w)==Adj[tmp_node].end() ){
                Adj[tmp_node].push_back(w);
            }

        }
        for(int j=0;j<out_degree;j++){
            uint tmp_node=rand()%vert;
            while(clusterID[tmp_node]==clusterID[w]){
                tmp_node=rand()%vert;
            }
            if(find(out_Adj[w].begin(),out_Adj[w].end(),tmp_node)==out_Adj[w].end() && tmp_node!=w ){
                out_Adj[w].push_back(tmp_node);
                if( find(out_Adj[tmp_node].begin(),out_Adj[tmp_node].end(),w)==out_Adj[tmp_node].end() ){
                    out_Adj[tmp_node].push_back(w);
                }
            }
        }

    }
    int edges = 0;
    for(uint j=0; j<vert; j++){  //init
        sort(Adj[j].begin(),Adj[j].end(),cmp);
        for(int k=0; k<Adj[j].size(); k++){
            fout<<j<<" "<<Adj[j][k]<<"\n";
            edges += 1;
        }
        sort(out_Adj[j].begin(),out_Adj[j].end(),cmp);
        for(int k=0; k<out_Adj[j].size(); k++){
            fout<<j<<" "<<out_Adj[j][k]<<"\n";
        }
    }
    fout.close();
    
    int mean_degree = edges / vert;
    cout << "m=" << edges << ", mean_degree=" << mean_degree << endl;
    
    delete[] clusterID;
    return 0;
}