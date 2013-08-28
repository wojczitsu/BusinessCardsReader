#include <iostream>
#include <math.h>

#include "trainer.h"

using namespace std;

//constructor 
trainer::trainer(neuralNetwork* nN) :	nn(nN), epoch(0),
										trainingSetAccuracy(0), validationSetAccuracy(0), generalizationSetAccuracy(0), 
										trainingSetMSE(0), validationSetMSE(0), generalizationSetMSE(0) {

	// create delta lists
	deltaInputHidden = new( double*[nn->nInput + 1] );
	for ( int i=0; i <= nn->nInput; i++ ) {

		deltaInputHidden[i] = new (double[nn->nHidden]);
		for ( int j=0; j < nn->nHidden; j++ ) {
			deltaInputHidden[i][j] = 0;		
		}
	}

	deltaHiddenOutput = new( double*[nn->nHidden + 1] );
	for ( int i=0; i <= nn->nHidden; i++ ) {

		deltaHiddenOutput[i] = new (double[nn->nOutput]);			
		for ( int j=0; j < nn->nOutput; j++ ) {
			deltaHiddenOutput[i][j] = 0;	
		}
	}

	// error gradient storage
	hiddenErrorGradients = new( double[nn->nHidden + 1] );
	for ( int i=0; i <= nn->nHidden; i++ ) {
		hiddenErrorGradients[i] = 0;
	}
	
	outputErrorGradients = new( double[nn->nOutput + 1] );
	for ( int i=0; i <= nn->nOutput; i++ ) {
		outputErrorGradients[i] = 0;
	}
}

//destructor
trainer::~trainer(){
	for (int i=0; i<=nn->nInput; i++) {
		delete[] deltaInputHidden[i];
	}
	delete[] deltaInputHidden;

	for (int i=0; i<=nn->nHidden; i++) {
		delete[] deltaHiddenOutput[i];
	}
	delete[] deltaHiddenOutput;

	delete[] hiddenErrorGradients;
	delete[] outputErrorGradients;
}

//set training parameters
void trainer::setTrainingParameters(double lR, double m) {
	learningRate = lR;
	momentum = m;
}

//set Stopping Conditions
void trainer::setStoppingConditions(int mEpochs, double dAccuracy, double dMSE){
	maxEpochs = mEpochs;
	desiredAccuracy = dAccuracy;
	desiredMSE = dMSE;
}

// trenowanie sieci
void trainer::trainNetwork(trainingDataSet* tSet) {
	cout	<< endl << " Zaczynamy trenowanie: " << endl
			<< " Learnin reate: " << learningRate << ", Momentum: " << momentum << ", Max Epochs: " << maxEpochs << ", Min MSE: " << desiredMSE << endl
			<< " " << nn->nInput << " Input Neurons, " << nn->nHidden << " Hidden Neurons, " << nn->nOutput << " Output Neurons" << endl
			<< endl << endl ;

	epoch = 0;
	// warunki aby przeszla petla while w pierwszym przebiegu / pozniej juz te wartosci sa liczone poprawnie
	trainingSetMSE = desiredMSE + 0.5;
	generalizationSetMSE = desiredMSE + 0.5;

	while ( (trainingSetAccuracy < desiredAccuracy || generalizationSetAccuracy < desiredAccuracy) 
			&& epoch < maxEpochs && (trainingSetMSE > desiredMSE || generalizationSetMSE > desiredMSE ) ) {

		double previousTAccuracy = trainingSetAccuracy;
		double previousGAccuracy = generalizationSetAccuracy;

		runTrainingEpoch( tSet->trainingSet ); // tutaj ustawiane trainingSetAccuracy i trainingSetMSE / wlasciwe trenowanie 

		generalizationSetAccuracy = nn->getSetAccuracy( tSet->generalizationSet );
		generalizationSetMSE = nn->getSetMSE( tSet->generalizationSet );

		// wypisywanie na ekran tylko wtedy gdy zmiana dokladnosci wieksza niz 1%
		if (ceil(previousTAccuracy) != ceil(trainingSetAccuracy) || ceil(previousGAccuracy) !=generalizationSetAccuracy) {
			cout << "Learning rate: " << learningRate << endl;
			cout << "Epoch :" << epoch;
			cout << " TSet Acc:" << trainingSetAccuracy << "%, MSE: " << trainingSetMSE ;
			cout << " GSet Acc:" << generalizationSetAccuracy << "%, MSE: " << generalizationSetMSE << endl;
		}

		epoch++; 
		if( epoch % 3 == 0) {
			learningRate = learningRate / 1.3 ;
		}		
	}	

	// po nauczeniu od razu walidacja
	validationSetAccuracy = nn->getSetAccuracy(tSet->validationSet);
	validationSetMSE = nn->getSetMSE(tSet->validationSet);

	cout << endl << "Nauka skonczona! - > Przeszlo epochow: " << epoch << endl;
	cout << " Dokladnosc walidacji: " << validationSetAccuracy <<"%" << endl;
	cout << " MSE zbioru walidujacego: " << validationSetMSE << endl << endl;
}

// trenowanie w danym epochu
void trainer::runTrainingEpoch( vector<dataEntry*> trainingSet ){
	
	double incorrectPatterns = 0;
	double mse = 0;

	for (int i = 0; i < (int) trainingSet.size(); i++){

		// nakarmienie sieci i policzenie bledow propagacji
		nn->feedForward( trainingSet[i]->pattern );
		backPropagate ( trainingSet[i]->target );

		// do obliczenia skutecznosci
		bool patternCorrect = true; 

		for (int j = 0; j < nn->nOutput; j++ ){

			// sprawdzanie poprawnosci
			if ( nn->clampOutput( nn->outputNeurons[j] ) != trainingSet[i]->target[j] ) {
				patternCorrect = false; 
			}

			// liczenie bledu MSE
			mse += pow( (nn->outputNeurons[j] - trainingSet[i]->target[j]), 2);
		}

		// liczenie niepoprawnych wynikow
		if (!patternCorrect) {
			incorrectPatterns++;
		}
	}

	// obliczenie skutecznosci w procentach i bledu mse
	trainingSetAccuracy = 100 - (incorrectPatterns/trainingSet.size() * 100);
	trainingSetMSE = mse / (nn->nOutput * trainingSet.size() );
}

// od warstwy koncowej do poczatkowej liczenie wag 
void trainer::backPropagate(double* desiredOutputs){

	for (int k = 0; k < nn->nOutput; k++) {
		outputErrorGradients[k] = getOutputErrorGradient ( desiredOutputs[k], nn->outputNeurons[k] );

		for (int j = 0; j <= nn->nHidden; j++) {
			// liczenie nowych wag 
			deltaHiddenOutput[j][k] = learningRate * nn->hiddenNeurons[j] * outputErrorGradients[k] + momentum * deltaHiddenOutput[j][k] ;
		}
	}

	for (int j=0; j < nn->nHidden; j++) {
		hiddenErrorGradients[j] = getHiddenErrorGradient( j );

		for (int i = 0; i <= nn->nInput; i++) {
			deltaInputHidden[i][j] = learningRate * nn->inputNeurons[i] * hiddenErrorGradients[j] + momentum * deltaInputHidden[i][j];
		}
	}

	// zmiana wag przy kazdym epochu 
	updateWeights();
}

// blad gradientu na wyjsciu
inline double trainer::getOutputErrorGradient( double desiredValue, double outputValue ) {
	// blad gradientu
	return outputValue * ( 1 - outputValue ) * (desiredValue - outputValue);
}

// blad gradientu na wejsciu
double trainer::getHiddenErrorGradient( int j ) {

	double weightSum = 0;
	for (int k=0; k < nn->nOutput; k++) {
		weightSum += nn->wHiddenOutput[j][k] * outputErrorGradients[k];
	}

	return nn->hiddenNeurons[j] * ( 1 - nn->hiddenNeurons[j] ) * weightSum; 
}

// aktualizowanie wag
void trainer::updateWeights() {

	//wagi pomiedzy warstwa wejsciowa a ukryta
	for (int i = 0; i <= nn->nInput; i++) {
		for (int j = 0; j < nn->nHidden; j++) {
			nn->wInputHidden[i][j] += deltaInputHidden[i][j];
		}
	}

	//wagi pomiedzy warstwa ukryta a wyjsciowa
	for (int j=0; j <=nn->nHidden; j++) {
		for (int k=0; k< nn->nOutput; k++) {
			nn->wHiddenOutput[j][k] += deltaHiddenOutput[j][k];
		}
	}
}
