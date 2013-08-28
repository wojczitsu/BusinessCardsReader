#pragma once

#include "neuralNetwork.h"

class trainer {

private:
	// siec do nauczenia
	neuralNetwork* nn;

	//parametry uczenia
	double learningRate;
	double momentum;

	// licznik epochow
	long epoch;
	long maxEpochs; 

	// oczekiwana min dokladnosc / max blad MSE 
	double desiredAccuracy;
	double desiredMSE;
	
	//parametr delta dla wag
	double** deltaInputHidden;
	double** deltaHiddenOutput;

	//blad gradientu
	double* hiddenErrorGradients;
	double* outputErrorGradients;

	//statystyki dokladnosci w danym epochow podczas uczenia
	double trainingSetAccuracy;
	double generalizationSetAccuracy;
	double validationSetAccuracy;
	double trainingSetMSE;
	double generalizationSetMSE;
	double validationSetMSE;

public:
	trainer( neuralNetwork* untrainedNN );
	~trainer();

	// ustawianie parametrow zatrzymania sie uczenia
	void setTrainingParameters(double learningRate, double momentum);
	void setStoppingConditions(int mEpochs, double dAccuracy, double dMSE);
	void trainNetwork( trainingDataSet* tSet ); // wywolanie fukncji runTrainingEpoch

private:
	void runTrainingEpoch( std::vector<dataEntry*> trainingSetAccuracy); // wywolanie funkcji backpropagate
	void backPropagate (double* desiredOutputs); // wywolanie funkcji getOutputErrorGradient, getHiddenErrorGradient, updateWeights
	inline double getOutputErrorGradient( double desiredValue, double outputValue );
	double getHiddenErrorGradient( int j );
	void updateWeights();	
};