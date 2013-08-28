#include <iostream>
#include <fstream>
#include <vector>

#include "neuralNetwork.h"

using namespace std;

//konstruktor
neuralNetwork::neuralNetwork(int nI, int nH, int nO) : nInput(nI), nHidden(nH), nOutput(nO) {
	
	//utworzenie listy neuronow wraz z bias dla warstwy wejsciowej i ukrytej
	inputNeurons = new (double[nInput + 1]);	
	for (int i = 0; i < nInput; i++) 
		inputNeurons[i] = 0;
	
	inputNeurons[nInput] = -1;

	hiddenNeurons = new (double[nHidden + 1]);
	for (int i = 0; i < nHidden; i++)
		hiddenNeurons[i] = 0;
	
	hiddenNeurons[nHidden] = -1;

	outputNeurons = new (double[nOutput]);
	for (int i = 0; i < nOutput; i++)
		outputNeurons[i] = 0;

	// utworzenie listy wag uwzgledniajac tez bias 
	wInputHidden = new (double*[nInput + 1]);
	for (int i = 0; i <= nInput; i++) {
		wInputHidden[i] = new (double[nHidden]);
		
		for (int j = 0; j < nHidden; j++)
			wInputHidden[i][j] = 0;
	}

	wHiddenOutput = new (double*[nHidden + 1]);
	for (int i = 0; i <= nHidden; i++) {
		wHiddenOutput[i] = new (double[nOutput]);

		for (int j = 0; j < nOutput; j++)
			wHiddenOutput[i][j] = 0;
	}

	initializeWeights();
}

//destruktor
neuralNetwork::~neuralNetwork() {

	delete[] inputNeurons;
	delete[] hiddenNeurons;
	delete[] outputNeurons;

	for(int i = 0; i <= nInput; i ++) {
		delete[] wInputHidden[i];
	}
	delete[] wInputHidden;

	for(int i = 0; i <= nHidden; i++) {
		delete[] wHiddenOutput[i];
	}
	delete[] wHiddenOutput;
}

// losowe nadanie wag dla uczenia 
void neuralNetwork::initializeWeights() {

	double rH = 1/sqrt( (double) nInput);
	double rO = 1/sqrt( (double) nHidden);
	
	//wagi pomiedzy neuronami wejsciowymi i ukrytymi
	for(int i = 0; i <= nInput; i++) {		
		for(int j = 0; j < nHidden; j++) {
			wInputHidden[i][j] = ( ( (double)(rand()%100)+1)/100  * 2 * rH ) - rH;			
		}
	}

	//wagi pomiedzy neuronami ukrytymi a wyjsciowymi
	for(int i = 0; i <= nHidden; i++) {		
		for(int j = 0; j < nOutput; j++) {			
			wHiddenOutput[i][j] = ( ( (double)(rand()%100)+1)/100 * 2 * rO ) - rO;
		}
	}
}

// funkcja aktywacji
inline double neuralNetwork::activationFunction( double x ) {
	return 1/(1+exp(-x));
}

// 
inline int neuralNetwork::clampOutput( double x ) {
	if ( x < 0.1 ) return 0;
	else if ( x > 0.9 ) return 1;
	else return -1;
}
// policzenie bledu dokladnosci
double neuralNetwork::getSetAccuracy( vector<dataEntry*>& set ) {
	double incorrectResults = 0;
		
	//dla kazdej danej wejsciowej
	for ( int tp = 0; tp < (int) set.size(); tp++) {						
		//nakramienie sieci
		feedForward( set[tp]->pattern );
		
		//flaga na poprawnosc wzorca
		bool correctResult = true;

		//sprawdzenie wyjsc do oczekiwanego rezultatu
		for ( int k = 0; k < nOutput; k++ )	{					
			
			//ustawienie flagi na false jesli oczekiwana wartosc jest bledna
			if ( clampOutput(outputNeurons[k]) != set[tp]->target[k] ) {
				correctResult = false;
			}
		}
		
		// zwiekszanie bledu skutecznosci jesli bledne rezultaty
		if ( !correctResult ) incorrectResults++;			
	}	

	//policzenie bledu i zwrocenie w procentach
	return 100 - (incorrectResults/set.size() * 100);
}

// policzenie bledu MSE
double neuralNetwork::getSetMSE( vector<dataEntry*>& set ) {
	double mse = 0;
		
	//dla kazdej danej wejsciowej
	for ( int tp = 0; tp < (int) set.size(); tp++) {						
		// nakarmienie sieci 
		feedForward( set[tp]->pattern );
		
		// sprawdzenie wszystkich wyjsc do oczekiwanego rezultatu
		for ( int k = 0; k < nOutput; k++ )	{					
			// zsumowanie MSE
			mse += pow((outputNeurons[k] - set[tp]->target[k]), 2);
		}			
	}
	
	//policzenie bledu i zwrocenie w procentach
	return mse/(nOutput * set.size());
}

//karmienei sieci 
void neuralNetwork::feedForward(double* pattern) {
	//ustawienie neuronow wejsciowych zadanym wzorcem
	for(int i = 0; i < nInput; i++) {
		inputNeurons[i] = pattern[i];
	}
	
	// policzenie wartosci neuronow warstyw ukrytej uwzgledniajac wagi i bias
	for(int j=0; j < nHidden; j++) {
		//wyczysczenei starej wartosci
		hiddenNeurons[j] = 0;				
		
		//policzenie sumy wag 
		for( int i=0; i <= nInput; i++ ){
			hiddenNeurons[j] += inputNeurons[i] * wInputHidden[i][j];
		}
		
		//pobranie rezultatu z fukncji aktywacyjnej
		hiddenNeurons[j] = activationFunction( hiddenNeurons[j] );			
	}
	
	// To samo co wyzej dla warstwy wyjsciowej neurnowo
	for(int k=0; k < nOutput; k++) {
	
		outputNeurons[k] = 0;				
		
		for( int j=0; j <= nHidden; j++ ) {
			outputNeurons[k] += hiddenNeurons[j] * wHiddenOutput[j][k];
		}
		
		outputNeurons[k] = activationFunction( outputNeurons[k] );
	}
}

//zapisanie wag do pliku dla potomnych
bool neuralNetwork::saveWeights(char* fileName) {
	
	fstream outputFile;
	outputFile.open(fileName, ios::out);

	if ( outputFile.is_open() )
	{
		outputFile.precision(50);		

		//wagi pomiedzy warstwa wejsciowa a ukryta
		for ( int i=0; i <= nInput; i++ ) 
		{
			for ( int j=0; j < nHidden; j++ ) 
			{
				outputFile << wInputHidden[i][j] << ",";				
			}
		}
		
		//wagi pomiedzy warstwa ukryta a wyjsciowa
		for ( int i=0; i <= nHidden; i++ ) 
		{		
			for ( int j=0; j < nOutput; j++ ) 
			{
				outputFile << wHiddenOutput[i][j];					
				if ( i * nOutput + j + 1 != (nHidden + 1) * nOutput ) outputFile << ",";
			}
		}

		cout << endl << "Wagi zostaly zapisane do: '" << fileName << "'" << endl;

		outputFile.close();
		return true;
	}
	else 
	{
		cout << endl << "Blad. Plik'" << fileName << "' nie moze zostac utworzoy: " << endl;
		return false;
	}
}