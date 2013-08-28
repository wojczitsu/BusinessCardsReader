#include <iostream>
#include <ctime>
#include <string>

#include "neuralNetwork.h"
#include "trainer.h"

using namespace std;

int main () {

	srand( (unsigned int) time(0) );

	dataReader d;
	d.loadDataFile("../Data/entry-character-recognition2.csv",400,64);
	
	neuralNetwork nn(400,250,64);

	trainer nT( &nn );
	nT.setTrainingParameters(0.1, 0.8);	// learningRate, Momentum  
	nT.setStoppingConditions(100, 90, 0.001); // max Epochs , min Accuracy, desired_mse
	nT.trainNetwork( d.getTrainingDataSet() );

	nn.saveWeights("../Data/weights.csv");

	char c; cin >> c;

	return 0;
}