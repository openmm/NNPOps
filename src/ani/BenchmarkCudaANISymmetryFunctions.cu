#include "CudaANISymmetryFunctions.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include <stdio.h>
#include <time.h>

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

void loadPdb(string filename, vector<float>& positions, vector<int>& species, vector<float>& periodicBoxVectors) {
    ifstream file(filename);
    if (!file.is_open())
        throw runtime_error("Failed to open PDB file");
    string line;
    vector<string> elements = {" H", " C", " N", " O", " S", " F", "Cl"};
    map<string, int> elementIndex;
    for (int i = 0; i < elements.size(); i++)
        elementIndex[elements[i]] = i;
    while (getline(file, line)) {
        if (line.rfind("ATOM", 0) == 0 || line.rfind("HETATM", 0) == 0) {
            positions.push_back(stof(line.substr(30, 8)));
            positions.push_back(stof(line.substr(38, 8)));
            positions.push_back(stof(line.substr(46, 8)));
            string element = line.substr(76, 2);
            if (elementIndex.find(element) == elementIndex.end())
                throw runtime_error("Unsupported element: "+element);
            species.push_back(elementIndex[element]);
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

void runBenchmark(CudaANISymmetryFunctions& ani, int iterations, vector<float>& positions, vector<int>& species, vector<float>& periodicBoxVectors) {
    float* positionsData;
    float* vectorsData;
    float* radial;
    float* angular;
    float* derivs;
    cudaMalloc(&positionsData, positions.size()*sizeof(float));
    cudaMalloc(&vectorsData, 9*sizeof(float));
    cudaMalloc(&radial, ani.getNumAtoms()*ani.getNumSpecies()*ani.getRadialFunctions().size()*sizeof(float));
    cudaMalloc(&angular, ani.getNumAtoms()*(ani.getNumSpecies()*(ani.getNumSpecies()+1)/2)*ani.getAngularFunctions().size()*sizeof(float));
    cudaMalloc(&derivs, positions.size()*sizeof(float));
    cudaMemcpy(positionsData, positions.data(), positions.size()*sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(vectorsData, periodicBoxVectors.data(), periodicBoxVectors.size()*sizeof(float), cudaMemcpyDefault);
    for (int i = 0; i < iterations; i++) {
        ani.computeSymmetryFunctions(positionsData, vectorsData, radial, angular);
        ani.backprop(radial, angular, derivs);
    }
    cudaFree(positionsData);
    cudaFree(vectorsData);
    cudaFree(radial);
    cudaFree(angular);
    cudaFree(derivs);
}

int main(int argc, char* argv[]) {
    try {
        if (argc != 3)
            throw runtime_error("Expected two command line arguments");
        vector<float> positions, periodicBoxVectors;
        vector<int> species;
        loadPdb(argv[1], positions, species, periodicBoxVectors);
        vector<RadialFunction> radialFunctions = {
            {19.7, 0.8},
            {19.7, 1.06875},
            {19.7, 1.3375},
            {19.7, 1.60625},
            {19.7, 1.875},
            {19.7, 2.14375},
            {19.7, 2.4125},
            {19.7, 2.68125},
            {19.7, 2.95},
            {19.7, 3.21875},
            {19.7, 3.4875},
            {19.7, 3.75625},
            {19.7, 4.025},
            {19.7, 4.29375},
            {19.7, 4.5625},
            {19.7, 4.83125}
        };
        vector<AngularFunction> angularFunctions = {
            {12.5, 0.8, 14.1, 0.392699},
            {12.5, 0.8, 14.1, 1.1781},
            {12.5, 0.8, 14.1, 1.9635},
            {12.5, 0.8, 14.1, 2.74889},
            {12.5, 1.1375, 14.1, 0.392699},
            {12.5, 1.1375, 14.1, 1.1781},
            {12.5, 1.1375, 14.1, 1.9635},
            {12.5, 1.1375, 14.1, 2.74889},
            {12.5, 1.475, 14.1, 0.392699},
            {12.5, 1.475, 14.1, 1.1781},
            {12.5, 1.475, 14.1, 1.9635},
            {12.5, 1.475, 14.1, 2.74889},
            {12.5, 1.8125, 14.1, 0.392699},
            {12.5, 1.8125, 14.1, 1.1781},
            {12.5, 1.8125, 14.1, 1.9635},
            {12.5, 1.8125, 14.1, 2.74889},
            {12.5, 2.15, 14.1, 0.392699},
            {12.5, 2.15, 14.1, 1.1781},
            {12.5, 2.15, 14.1, 1.9635},
            {12.5, 2.15, 14.1, 2.74889},
            {12.5, 2.4875, 14.1, 0.392699},
            {12.5, 2.4875, 14.1, 1.1781},
            {12.5, 2.4875, 14.1, 1.9635},
            {12.5, 2.4875, 14.1, 2.74889},
            {12.5, 2.825, 14.1, 0.392699},
            {12.5, 2.825, 14.1, 1.1781},
            {12.5, 2.825, 14.1, 1.9635},
            {12.5, 2.825, 14.1, 2.74889},
            {12.5, 3.1625, 14.1, 0.392699},
            {12.5, 3.1625, 14.1, 1.1781},
            {12.5, 3.1625, 14.1, 1.9635},
            {12.5, 3.1625, 14.1, 2.74889}
        };
        CudaANISymmetryFunctions ani(species.size(), 7, 5.1, 3.5, periodicBoxVectors.size() > 0, species, radialFunctions, angularFunctions, true);
        clock_t start, finish;
        double duration;
        start = clock();
        runBenchmark(ani, stoi(argv[2]), positions, species, periodicBoxVectors);
        finish = clock();
        duration = (double)(finish - start) / CLOCKS_PER_SEC;
        printf("  %f s\n", duration);
        printf("  %f ms/it\n", duration/stoi(argv[2])*1000);
    }
    catch (const exception& e) {
        cout << e.what() << endl;
        return 1;
    }
    return 0;
}
