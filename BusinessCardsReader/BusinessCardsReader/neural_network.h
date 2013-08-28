#ifndef NNetwork
#define NNetwork

#include "stdafx.h"

class neuralNetwork {

private: 
	// ilosc neuronow w danej warstwie
	int nInput, nHidden, nOutput;

	// tablica wartosci neuronow
	double* inputNeurons;
	double* hiddenNeurons;
	double* outputNeurons; 

	//wagi pomiedzy warstwami
	double** wInputHidden;
	double** wHiddenOutput;

public:
	neuralNetwork(int numInput, int numHidden, int numOutput, char* fileName);
	~neuralNetwork();
	
	int* feedForwardPatternAndGetResult(double* pattern);
	string parseResultToString (int* result);

private:
	void loadWeights(char* inputFilename);
	inline double activationFunction(double x);
	int* getResults(double* outputNeurnos); 
	void feedForward(double* pattern);
};

#endif