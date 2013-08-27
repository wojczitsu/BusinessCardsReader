#pragma once

#include "dataReader.h"

class trainer ;

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

	friend trainer;

public:
	neuralNetwork(int numInput, int numHidden, int numOutput);
	~neuralNetwork();

	double getSetAccuracy (std::vector<dataEntry*>& set);
	double getSetMSE (std::vector<dataEntry*>& set);

	bool saveWeights(char* fileName);

private:
	void initializeWeights();
	inline double activationFunction( double x );
	inline int clampOutput( double x );
	void feedForward(double* pattern);
};
