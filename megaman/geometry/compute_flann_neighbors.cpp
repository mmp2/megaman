#include <flann/flann.hpp> 
#include <stdio.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <ctime>
using namespace flann;
using namespace std;

// usage: ./flann_test_cpp [num_pts] [num_dims] [radius] [raw_file_name] 
    // num_pts (int), num_dims (int), radius (float), 
    // raw_file_name: tab or space delimited text file with num_pts rows and num_dims columns
// output: [neighbors.dat], [distances.dat]

int main(int argc, char** argv)
{
    clock_t t0 = clock();
    // get command line arguments 
    int num_pts = atoi(argv[1]);
    int num_dims = atoi(argv[2]);
    float radius = atof(argv[3]);
    const char* fname = argv[4];
    
    // print input parameters 
    cout << "number of points: " << num_pts << endl;
    cout << "number of dimensions: " << num_dims << endl;
    cout << "radius: " << radius << endl;
    cout << "data file name: " << fname << endl;
    cout << endl;
    
    // read in data from file 
    Matrix<float> dataset(new float[num_pts*num_dims], num_pts, num_dims);
    clock_t t1 = clock();

    ifstream in_file;
    
    // METHOD 1 (space separated text file) 
    //    in_file.open(fname);
    //    for (int i =0; i < num_pts; i++){
    //        for (int j = 0; j < num_dims; j++){
    //            in_file >> dataset[i][j];
    //        }
    //     }

    // METHOD 2 (binary file)
    double* myArray = new double[num_pts*num_dims];
    in_file.open(fname, ios::binary | ios::in );
    in_file.read((char*)myArray, num_pts * num_dims * sizeof(double));
    in_file.close();
    for(int n = 0; n < num_pts; n++){
      for(int d = 0; d < num_dims; d++){
	dataset[n][d]=myArray[num_dims*n + d];
      }
    }
    delete [] myArray;

    clock_t t2 = clock();
    double read_duration = (t2 - t1)/ static_cast<double>( CLOCKS_PER_SEC );
    cout << "time spent reading: " << read_duration << " seconds" << endl;

    // initialize the indices and dists 2D vectors
    vector< vector<int> > indices;
    vector< vector<float> > dists; 
    
    // construct KMeans index 
    Index<L2<float> > index(dataset, KMeansIndexParams());
    index.buildIndex();                                                                                               

    // do a radius search
    index.radiusSearch(dataset, indices, dists, radius, SearchParams());
    
    // save neighbors to file 
    clock_t t3 = clock();
    double radius_duration = (t3 - t2)/ static_cast<double>( CLOCKS_PER_SEC );
    cout << "time spent on radius neighbors: " << radius_duration << " seconds" << endl;
    ofstream nbrfile;
    nbrfile.open("neighbors.dat");
    for ( vector<vector<int> >::size_type i = 0; i < indices.size(); i++ )
    {
        for ( vector<int>::size_type j = 0; j < indices[i].size(); j++ )
        {
            nbrfile << indices[i][j] << " "; 
        }
        nbrfile << endl;
    }
    cout << endl;
    nbrfile.close();

    // save distances to file 
    ofstream distfile;
    distfile.open("distances.dat");
    for ( vector<vector<int> >::size_type i = 0; i < dists.size(); i++ )
    {
        for ( vector<int>::size_type j = 0; j < dists[i].size(); j++ )
        {
            distfile << dists[i][j] << " ";
        }
        distfile << endl;
    }
    distfile.close();
    clock_t t4 = clock();
    double write_duration = (t4 - t3)/ static_cast<double>( CLOCKS_PER_SEC );
    cout << "time spent writing files: " << write_duration << " seconds" << endl;    
    double total_duration = (t4 - t0)/ static_cast<double>( CLOCKS_PER_SEC );   
    cout << "total time spent: " << total_duration << " seconds" << endl;    
    
    ofstream timefile;
    timefile.open("timing.dat");
    timefile << "time spent reading: " << read_duration << " seconds" << endl;
    timefile << "time spent on radius neighbors: " << radius_duration << " seconds" << endl;
    timefile << "time spent writing files: " << write_duration << " seconds" << endl;    
    timefile << "total time spent: " << total_duration << " seconds" << endl;
    timefile.close();
    return 0;
}
