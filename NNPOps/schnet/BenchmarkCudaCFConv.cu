#include "CudaCFConv.h"
#include <cmath>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

void computeBoxVectors(float a, float b, float c, float alpha, float beta, float gamma, vector<float>& periodicBoxVectors) {
    float bx = b*cos(gamma);
    float by = b*sin(gamma);
    float cx = c*cos(beta);
    float cy = c*(cos(alpha)-cos(beta)*cos(gamma))/sin(gamma);
    float cz = sqrt(c*c-cx*cx-cy*cy);
    float scale1 = std::round(cy/by);
    cx -= bx*scale1;
    cy -= by*scale1;
    float scale2 = std::round(cx/a);
    cx -= a*scale2;
    float scale3 = std::round(bx/a);
    bx -= a*scale3;
    periodicBoxVectors.push_back(a);
    periodicBoxVectors.push_back(0);
    periodicBoxVectors.push_back(0);
    periodicBoxVectors.push_back(bx);
    periodicBoxVectors.push_back(by);
    periodicBoxVectors.push_back(0);
    periodicBoxVectors.push_back(cx);
    periodicBoxVectors.push_back(cy);
    periodicBoxVectors.push_back(cz);
}

void loadPdb(string filename, vector<float>& positions, vector<float>& periodicBoxVectors) {
    ifstream file(filename);
    if (!file.is_open())
        throw runtime_error("Failed to open PDB file");
    string line;
    while (getline(file, line)) {
        if (line.rfind("ATOM", 0) == 0 || line.rfind("HETATM", 0) == 0) {
            positions.push_back(stof(line.substr(30, 8)));
            positions.push_back(stof(line.substr(38, 8)));
            positions.push_back(stof(line.substr(46, 8)));
        }
        if (line.rfind("CRYST1", 0) == 0) {
            float a = stof(line.substr(6, 9));
            float b = stof(line.substr(15, 9));
            float c = stof(line.substr(24, 9));
            float alpha = stof(line.substr(33, 7))*M_PI/180;
            float beta = stof(line.substr(40, 7))*M_PI/180;
            float gamma = stof(line.substr(47, 7))*M_PI/180;
            computeBoxVectors(a, b, c, alpha, beta, gamma, periodicBoxVectors);
        }
    }
    file.close();
}

void runBenchmark(int iterations, vector<float>& positions, vector<float>& periodicBoxVectors) {
    int numAtoms = positions.size()/3;
    int width = 128;
    int numGaussians = 50;
    float cutoff = 10;

    // Generate random weights and biases.  We don't care about the values, since they
    // don't affect speed.

    vector<float> w1, b1, w2, b2;
    std::default_random_engine generator(0);
    std::normal_distribution<double> distribution(0, 1);
    for (int i = 0; i < width; i++) {
        b1.push_back(distribution(generator));
        b2.push_back(distribution(generator));
        for (int j = 0; j < numGaussians; j++)
            w1.push_back(distribution(generator));
        for (int j = 0; j < width; j++)
            w2.push_back(distribution(generator));
    }

    // Allocate all the memory we will need.

    CudaCFConvNeighbors neighbors(numAtoms, cutoff, periodicBoxVectors.size() > 0);
    CudaCFConv cfconv(numAtoms, width, numGaussians, cutoff, periodicBoxVectors.size() > 0, 0.2, CFConv::ShiftedSoftplus, w1.data(), b1.data(), w2.data(), b2.data());
    float *positionsData, *vectorsData, *input, *output, *inputDerivs, *outputDerivs, *positionDerivs;
    cudaMalloc(&positionsData, positions.size()*sizeof(float));
    cudaMalloc(&vectorsData, 9*sizeof(float));
    cudaMallocManaged(&input, numAtoms*width*sizeof(float));
    cudaMalloc(&output, numAtoms*width*sizeof(float));
    cudaMalloc(&inputDerivs, numAtoms*width*sizeof(float));
    cudaMalloc(&outputDerivs, numAtoms*width*sizeof(float));
    cudaMalloc(&positionDerivs, positions.size()*sizeof(float));
    cudaMemcpy(positionsData, positions.data(), positions.size()*sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(vectorsData, periodicBoxVectors.data(), periodicBoxVectors.size()*sizeof(float), cudaMemcpyDefault);
    for (int i = 0; i < width; i++)
        for (int j = 0; j < numAtoms; j++)
            input[j*width+i] = distribution(generator);

    // Run the benchmark.

    clock_t start = clock();
    for (int i = 0; i < iterations; i++) {
        neighbors.build(positionsData, vectorsData);
        for (int j = 0; j < 6; j++) {
            cfconv.compute(neighbors, positionsData, vectorsData, input, output);
            cfconv.backprop(neighbors, positionsData, vectorsData, input, outputDerivs, inputDerivs, positionDerivs);
        }
    }
    cudaDeviceSynchronize();
    clock_t finish = clock();
    double duration = (double) (finish-start)/CLOCKS_PER_SEC;
    printf("  %f sec\n", duration);
    printf("  %f ms/iteration\n", duration/iterations*1000);

    // Release device memory.

    cudaFree(positionsData);
    cudaFree(vectorsData);
    cudaFree(input);
    cudaFree(output);
    cudaFree(inputDerivs);
    cudaFree(outputDerivs);
    cudaFree(positionDerivs);
}

int main(int argc, char* argv[]) {
    try {
        if (argc != 3)
            throw runtime_error("Expected two command line arguments");
        vector<float> positions, periodicBoxVectors;
        loadPdb(argv[1], positions, periodicBoxVectors);
        runBenchmark(stoi(argv[2]), positions, periodicBoxVectors);
    }
    catch (const exception& e) {
        cout << e.what() << endl;
        return 1;
    }
    return 0;
}
