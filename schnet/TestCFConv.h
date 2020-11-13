#include "CFConv.h"
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <cstdio>

using namespace std;

void assertEqual(float v1, float v2, float atol, float rtol) {
    float diff = fabs(v1-v2);
    if (diff > atol && diff/v1 > rtol)
        throw runtime_error(string("Assertion failure: expected ")+to_string(v1)+" found "+to_string(v2));
}

void validateDerivatives(CFConvNeighbors& neighbors, CFConv& conv, float* positions, float* periodicVectors, vector<float>& x) {
    int numAtoms = conv.getNumAtoms();
    int width = conv.getWidth();
    vector<float> y(numAtoms*width);
    vector<float> inputDeriv(numAtoms*width), positionDeriv(numAtoms*3), outputDeriv(numAtoms*width, 0);
    vector<float> offsetx(numAtoms*width), offsetPositions(numAtoms*3);
    float step = 1e-3;
    for (int i = 0; i < y.size(); i++) {
        // Use backprop to compute the gradient of one symmetry function.

        neighbors.build(positions, periodicVectors);
        conv.compute(neighbors, positions, periodicVectors, x.data(), y.data());
        outputDeriv[i] = 1;
        conv.backprop(neighbors, positions, periodicVectors, x.data(), outputDeriv.data(), inputDeriv.data(), positionDeriv.data());
        outputDeriv[i] = 0;

        // Displace the inputs along the gradient direction, compute the symmetry functions,
        // and calculate a finite difference approximation to the gradient magnitude from them.

        float norm = 0;
        for (int j = 0; j < inputDeriv.size(); j++)
            norm += inputDeriv[j]*inputDeriv[j];
        norm = sqrt(norm);
        float delta = step/norm;
        for (int j = 0; j < offsetx.size(); j++)
            offsetx[j] = x[j] - delta*inputDeriv[j];
        conv.compute(neighbors, positions, periodicVectors, offsetx.data(), y.data());
        float value1 = y[i];
        for (int j = 0; j < offsetx.size(); j++)
            offsetx[j] = x[j] + delta*inputDeriv[j];
        conv.compute(neighbors, positions, periodicVectors, offsetx.data(), y.data());
        float value2 = y[i];
        float estimate = (value2-value1)/(2*step);

        // Verify that they match.

        assertEqual(norm, estimate, 1e-5, 5e-3);

        // Displace the atom positions along the gradient direction, compute the symmetry functions,
        // and calculate a finite difference approximation to the gradient magnitude from them.

        norm = 0;
        for (int j = 0; j < positionDeriv.size(); j++)
            norm += positionDeriv[j]*positionDeriv[j];
        norm = sqrt(norm);
        delta = step/norm;
        for (int j = 0; j < offsetPositions.size(); j++)
            offsetPositions[j] = positions[j] - delta*positionDeriv[j];
        neighbors.build(offsetPositions.data(), periodicVectors);
        conv.compute(neighbors, offsetPositions.data(), periodicVectors, x.data(), y.data());
        value1 = y[i];
        for (int j = 0; j < offsetPositions.size(); j++)
            offsetPositions[j] = positions[j] + delta*positionDeriv[j];
        neighbors.build(offsetPositions.data(), periodicVectors);
        conv.compute(neighbors, offsetPositions.data(), periodicVectors, x.data(), y.data());
        value2 = y[i];
        estimate = (value2-value1)/(2*step);

        // Verify that they match.

        assertEqual(norm, estimate, 1e-5, 5e-3);
    }
}

void testWater(float* periodicVectors, float* expectedOutput) {
    int numAtoms = 18;
    float positions[18][3] = {
        { 0.726, -1.384, -0.376},
        {-0.025, -0.828, -0.611},
        { 1.456, -1.011, -0.923},
        {-1.324,  0.387, -0.826},
        {-1.923,  0.698, -1.548},
        {-1.173,  1.184, -0.295},
        { 0.837, -1.041,  2.428},
        { 1.024, -1.240,  1.461},
        { 1.410, -1.677,  2.827},
        { 2.765,  0.339, -1.505},
        { 2.834,  0.809, -0.685},
        { 3.582, -0.190, -1.593},
        {-0.916,  2.705,  0.799},
        {-0.227,  2.580,  1.426},
        {-0.874,  3.618,  0.468},
        {-2.843, -1.749,  0.001},
        {-2.928, -2.324, -0.815},
        {-2.402, -0.876, -0.235}
    };
    float w1[8][5] = {
        { 0.2463,  0.4514, -0.0814,  0.4989, -0.0181},
        { 0.1861, -0.0959,  0.5731, -0.4886,  0.2752},
        { 0.1224,  0.0544, -0.4167,  0.0564,  0.4693},
        {-0.3454,  0.2254,  0.1369,  0.1254, -0.0839},
        { 0.6725, -0.2192, -0.2956,  0.1557, -0.4014},
        { 0.2698,  0.1810, -0.0427, -0.2896, -0.1737},
        { 0.2868, -0.0575,  0.1532, -0.3876, -0.3673},
        {-0.1480,  0.4084, -0.1911, -0.5908, -0.3344}
    };
    float w2[8][8] = {
        { 0.0330,  0.0760, -0.0510,  0.1475, -0.0068,  0.3645,  0.2031,  0.1880},
        {-0.5510,  0.5577, -0.5649,  0.0140,  0.3258, -0.0632, -0.0646,  0.5246},
        { 0.3532,  0.1317,  0.2234,  0.2135, -0.0733,  0.3791, -0.4857,  0.2957},
        {-0.1780, -0.4993, -0.4461, -0.2980,  0.4579,  0.4065, -0.3442, -0.1189},
        {-0.3317,  0.4657,  0.5751, -0.5237,  0.1118, -0.2340, -0.1010, -0.1860},
        {-0.2565,  0.0528, -0.5955,  0.4339, -0.5280, -0.1271, -0.3940,  0.5210},
        {-0.4741,  0.5324,  0.1273, -0.1002,  0.4240, -0.0653, -0.5595,  0.3056},
        {-0.3968,  0.5980, -0.5542,  0.3235,  0.5160,  0.1541,  0.5051, -0.3144}
    };
    vector<float> b1 = {1, 2, 3, 4, 5, 6, 7, 8};
    vector<float> b2 = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    vector<float> x(8*18);
    for (int i = 0; i < x.size(); i++)
        x[i] = 0.1*i;
    vector<float> y(8*18);
    CFConvNeighbors* neighbors = createNeighbors(numAtoms, 2.0, (periodicVectors != NULL));
    neighbors->build((float*) positions, periodicVectors);
    CFConv* conv = createConv(numAtoms, 8, 5, 2.0, (periodicVectors != NULL), 0.5, (float*) w1, b1.data(), (float*) w2, b2.data());
    conv->compute(*neighbors, (float*) positions, periodicVectors, x.data(), y.data());
    for (int i = 0; i < y.size(); i++)
        assertEqual(expectedOutput[i], y[i], 1e-4, 1e-3);
    validateDerivatives(*neighbors, *conv, (float*) positions, periodicVectors, x);
    delete neighbors;
    delete conv;
}

void testWaterNonperiodic() {
    // Values computed with SchNetPack.
    float expectedOutput[] = {
        6.6657829, 5.1362805, 4.3362718, -2.6427372, -4.1542015, -1.5942615, 2.0310063, 10.610411,
        1.3191403, 1.0992877, 1.174861, -0.76823407, -1.1996126, -0.66166562, 0.74301827, 3.7644627,
        0.52987105, 0.57404131, 0.6803357, -0.48603082, -0.83363354, -0.43371251, 0.52734971, 2.8404183,
        21.448631, 15.574216, 12.700784, -7.4042816, -11.032789, -4.3033171, 5.1564302, 25.685184,
        8.6628437, 6.2632074, 5.2727389, -3.0848112, -4.5920477, -1.9519548, 2.2185543, 11.037331,
        8.7150316, 6.3248448, 5.3149056, -3.114675, -4.6293163, -1.9557781, 2.2522798, 11.153768,
        32.172924, 23.424303, 18.550533, -10.736898, -16.264935, -5.7996726, 7.192831, 36.542068,
        16.937481, 12.04541, 9.9613571, -5.7430658, -8.4591007, -3.5382349, 3.9525094, 19.534882,
        17.61989, 12.638445, 10.343842, -5.9593883, -8.8567104, -3.559639, 4.1038408, 20.33786,
        45.743965, 33.181282, 26.131493, -15.055626, -22.725782, -8.0308104, 9.9640656, 50.404785,
        25.268753, 18.019264, 14.644307, -8.3915596, -12.369637, -4.9310017, 5.6836424, 27.934893,
        23.857389, 16.993416, 13.829982, -7.940237, -11.693309, -4.6838045, 5.3686848, 26.462799,
        59.573944, 43.115246, 33.862885, -19.452675, -29.289074, -10.321277, 12.795625, 64.493713,
        33.42009, 23.799131, 19.233534, -10.987288, -16.175241, -6.3719406, 7.3756623, 36.168938,
        31.60857, 22.469316, 18.195433, -10.415167, -15.306055, -6.0727119, 6.9771519, 34.30328,
        67.125504, 48.290691, 38.041954, -21.869686, -32.762165, -11.755814, 14.26789, 72.072632,
        35.483078, 25.192368, 20.326399, -11.645652, -17.048292, -6.7168756, 7.7418537, 38.036129,
        35.362617, 25.096624, 20.284662, -11.627875, -16.997498, -6.7423472, 7.7480173, 38.007557
    };
    testWater(NULL, expectedOutput);
}

void testWaterPeriodic() {
    // Values computed with SchNetPack.
    float expectedOutput[] = {
        18.987059, 13.167937, 11.771148, -6.8072405, -9.4687815, -4.9209137, 5.0537095, 23.428143,
        10.606168, 7.2291617, 6.7242117, -3.8749375, -5.3526406, -3.0690796, 2.9493177, 13.598877,
        47.210438, 31.992928, 28.139683, -15.919709, -22.448139, -11.650467, 11.166942, 52.899227,
        81.724319, 59.101315, 47.216019, -27.291693, -40.958469, -15.187178, 18.343681, 92.38488,
        87.742668, 63.669167, 50.596939, -27.877956, -43.177094, -17.64563, 19.995712, 97.293167,
        25.624153, 18.281868, 15.086675, -8.7424355, -12.883776, -5.3536925, 6.0267334, 29.912189,
        32.435249, 23.602295, 18.728577, -10.842152, -16.393995, -5.893826, 7.2826471, 36.905506,
        19.521622, 13.713362, 11.546396, -6.637351, -9.5248575, -4.2738609, 4.6259727, 22.199139,
        23.766819, 16.700838, 14.142832, -8.1113482, -11.664352, -5.3069444, 5.7081265, 27.309629,
        67.918587, 49.29739, 39.438034, -22.224222, -33.916054, -13.497845, 15.671877, 77.092125,
        51.731312, 37.289658, 30.424944, -17.598099, -26.194244, -10.355038, 12.153141, 60.363403,
        39.829742, 28.671593, 23.424986, -13.577568, -20.137529, -8.0050716, 9.3628139, 46.48629,
        59.573944, 43.115246, 33.862888, -19.452673, -29.289074, -10.321277, 12.795625, 64.493713,
        34.680882, 24.618969, 20.0191, -11.433189, -16.7152, -6.7448702, 7.7174506, 37.542442,
        34.539341, 24.418261, 20.073877, -11.498013, -16.661949, -6.9871426, 7.8319163, 37.864975,
        68.965775, 49.63401, 39.307724, -22.631813, -33.887375, -12.384632, 14.897013, 75.146507,
        39.130844, 27.649551, 22.657299, -12.986568, -18.833965, -7.8369241, 8.7835398, 42.641678,
        42.567787, 29.885326, 24.786411, -14.207994, -20.254772, -8.8278933, 9.7198906, 46.303577
    };
    float periodicVectors[] = {
        5.0, 0.0, 0.0,
        0.0, 5.0, 0.0,
        0.0, 0.0, 5.0
    };
    testWater(periodicVectors, expectedOutput);
}

void testWaterTriclinic() {
    // Values computed with SchNetPack.
    float expectedOutput[] = {
        17.745617, 12.366516, 11.006595, -6.3729196, -8.9715538, -4.5642729, 4.7206745, 22.16106,
        2.1477644, 1.6353126, 1.6874884, -1.0601712, -1.5335996, -0.9023248, 0.96841896, 4.6215644,
        48.149979, 32.666721, 28.666414, -16.224579, -22.909214, -11.816319, 11.362852, 53.923328,
        81.724319, 59.101315, 47.216019, -27.291695, -40.958469, -15.18718, 18.343681, 92.38488,
        87.898651, 63.771194, 50.694328, -27.934128, -43.23925, -17.691732, 20.040503, 97.457832,
        31.421341, 22.045143, 18.586432, -10.696246, -15.383665, -6.9288845, 7.4406862, 35.924221,
        32.862415, 23.872673, 18.981148, -10.983425, -16.547239, -6.0044589, 7.3860188, 37.276985,
        17.575203, 12.459368, 10.354189, -5.9679904, -8.7071857, -3.7207754, 4.1282072, 20.17907,
        25.453362, 17.910492, 15.153757, -8.6952667, -12.591169, -5.7042055, 6.1155634, 29.505075,
        73.083969, 52.727303, 42.568474, -23.987827, -36.284313, -14.885297, 16.949322, 82.807732,
        51.731312, 37.289658, 30.424944, -17.598099, -26.194244, -10.355038, 12.153141, 60.363403,
        40.332741, 28.998619, 23.738894, -13.757333, -20.342489, -8.1541634, 9.5037212, 47.020203,
        93.104004, 66.689079, 53.167786, -30.515736, -45.219173, -16.985079, 20.209469, 100.40714,
        33.590511, 23.91007, 19.338394, -11.047472, -16.240902, -6.4204984, 7.4229531, 36.340359,
        121.82475, 87.80098, 69.5644, -39.282513, -58.987659, -21.967787, 26.487518, 129.77406,
        131.88364, 95.230721, 75.166237, -42.842003, -64.655472, -23.57815, 28.533545, 142.7018,
        44.382286, 31.053387, 25.881222, -14.801838, -21.071901, -9.3320808, 10.13959, 48.193203,
        77.567245, 55.379211, 44.659477, -25.511921, -37.458195, -14.745174, 17.227661, 83.865067
    };
    float periodicVectors[] = {
        5.0, 0.0, 0.0,
        1.5, 5.0, 0.0,
        -0.5, -1.0, 5.0
    };
    testWater(periodicVectors, expectedOutput);
}

int main() {
    testWaterNonperiodic();
    testWaterPeriodic();
    testWaterTriclinic();
    return 0;
}
