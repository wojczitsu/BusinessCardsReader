#include "stdafx.h"

//konstruktor
neuralNetwork::neuralNetwork(int nI, int nH, int nO, char* fileName) throw(string) : nInput(nI), nHidden(nH), nOutput(nO) {
	
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

	try {
		loadWeights(fileName);
	}
	catch(string ex){
		throw ex;
	}		
}

//destruktor
neuralNetwork::~neuralNetwork() {

	delete[] inputNeurons;
	delete[] hiddenNeurons;
	delete[] outputNeurons;

	for(int i = 0; i <= nInput; i ++)
		delete[] wInputHidden[i];
	delete[] wInputHidden;

	for(int i = 0; i <= nHidden; i++)
		delete[] wHiddenOutput[i];
	delete[] wHiddenOutput;
}

//ladowanie wag z pliku .csv
void neuralNetwork::loadWeights(char * fileName) {

	fstream inputFile;
	inputFile.open(fileName, ios::in);

	if(inputFile.is_open()){
		vector<double> weights;
		string line = "";

		while(!inputFile.eof()){
			getline(inputFile, line);

			if(line.length() > 2) {
				char* cstr = new char[line.size()+1];
				char* t;
				strcpy(cstr, line.c_str());

				// split danych z csv - dzielimy miedzy przecinkami
				int i = 0;	
				t=strtok(cstr,",");

				while( t!=NULL) {
					weights.push_back(atof(t));
					t=strtok(NULL, ",");
					i++;
				}

				delete[] cstr;
			}
		}

		if( weights.size() != ((nInput + 1)*nHidden + (nHidden + 1)*nOutput)) {
			
			inputFile.close();

			string exception = "sufficient weights weren't loaded";
			throw exception;			
		}
		else {
			int pos = 0; 

			for(int i = 0; i <= nInput; i++) {
				for (int j=0; j < nHidden; j++) {
					wInputHidden[i][j] = weights[pos++];					
				}
			}

			for(int i = 0; i <= nHidden; i++) {
				for(int j = 0; j < nOutput; j++) {
					wHiddenOutput[i][j] = weights[pos++];
				}
			}

			inputFile.close();
		}

	}
	else {
		string exception = "cannot open file";
		throw exception;
	}

}

// funkcja aktywacji
inline double neuralNetwork::activationFunction(double x) {
	//funkcja sigmoidalna
	return 1/(1+exp(-x));
}

// karmienie sieci danymi na wejscie wedlug wzoru i pobranie rezultatu
int* neuralNetwork::feedForwardPatternAndGetResult(double* pattern) {

	feedForward(pattern);
	return getResults(outputNeurons);
}

// karmienie sieci
void neuralNetwork::feedForward(double* pattern) {
	
	for(int i = 0; i < nInput; i++)
		inputNeurons[i] = pattern[i];

	for(int j = 0; j < nHidden; j++) {
		hiddenNeurons[j] = 0;

		for(int i = 0; i <= nInput; i++)
			hiddenNeurons[j] += inputNeurons[i] * wInputHidden[i][j];

		hiddenNeurons[j] = activationFunction ( hiddenNeurons[j] ) ;
	}

	for(int k = 0; k < nOutput; k++){
		outputNeurons[k] = 0 ;

		for(int j = 0; j <= nHidden; j++) 
			outputNeurons[k] += hiddenNeurons[j] * wHiddenOutput[j][k];

		outputNeurons[k] = activationFunction(outputNeurons[k]);
	}
}

// pobranie rezultatu - szukanie najbardziej prawdopodobnego wyniku
int * neuralNetwork::getResults(double * outputNeurons){
	
	int* results = new int[nOutput];

	int posOne = 0;	
	results[0] = 0;

	for (int i=1; i < nOutput; i++ ) {
		if(outputNeurons[posOne] < outputNeurons[i]) {
			posOne = i ;
		}			
		results[i] = 0; 
	}

	results[posOne] = 1;
	return results;
}

// analiza neuronow wyjsciowych, zwrocenie konkretnego stringa dla przypadku tej sieci
string neuralNetwork::parseResultToString(int* result) throw(string) {
	int position;

	for(int i=0 ; i<nOutput; i++){	// szukanie 1
		if(result[i]){
			position = i;
		}	
	}

	switch(position) {
		case 63 : return "A"; break; 
		case 62 : return "B"; break; 
		case 61 : return "C"; break; 
		case 60 : return "D"; break; 
		case 59 : return "E"; break; 
		case 58 : return "F"; break; 
		case 57 : return "G"; break; 
		case 56 : return "H"; break; 
		case 55 : return "I"; break; 
		case 54 : return "J"; break; 
		case 53 : return "K"; break; 
		case 52 : return "L"; break; 
		case 51 : return "M"; break; 
		case 50 : return "N"; break; 
		case 49 : return "O"; break;
		case 48 : return "P"; break;
		case 47 : return "Q"; break;
		case 46 : return "R"; break;
		case 45 : return "S"; break;
		case 44 : return "T"; break;
		case 43 : return "U"; break;
		case 42 : return "V"; break;
		case 41 : return "W"; break;
		case 40 : return "X"; break;
		case 39 : return "Y"; break;
		case 38 : return "Z"; break;

		case 37 : return "a"; break;
		case 36 : return "b"; break;
		case 35 : return "c"; break;
		case 34 : return "d"; break;
		case 33 : return "e"; break;
		case 32 : return "f"; break;
		case 31 : return "g"; break;
		case 30 : return "h"; break;
		case 29 : return "i"; break;
		case 28 : return "j"; break;
		case 27 : return "k"; break;
		case 26 : return "l"; break;
		case 25 : return "m"; break;
		case 24 : return "n"; break;
		case 23 : return "o"; break;
		case 22 : return "p"; break;
		case 21 : return "q"; break;
		case 20 : return "r"; break;
		case 19 : return "s"; break;
		case 18 : return "t"; break;
		case 17 : return "u"; break;
		case 16 : return "v"; break;
		case 15 : return "w"; break;
		case 14 : return "x"; break;
		case 13 : return "y"; break;
		case 12 : return "z"; break;

		case 11 : return "9"; break;
		case 10 : return "8"; break;
		case 9 : return "7"; break;
		case 8 : return "6"; break;
		case 7 : return "5"; break;
		case 6 : return "4"; break;
		case 5 : return "3"; break;
		case 4 : return "2"; break;
		case 3 : return "1"; break;
		case 2 : return "0"; break;
		case 1 : return "."; break;
		case 0 : return "@"; break;
		default:
			string exception = "Pozycja poza zakresem" ;
			throw exception;
		break;
	}
	
	string exception = "Nieoczekiwany blad" ;
	throw exception;
}