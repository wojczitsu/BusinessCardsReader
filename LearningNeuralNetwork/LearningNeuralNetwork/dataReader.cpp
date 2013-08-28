#include "dataReader.h"

#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <algorithm>

using namespace std;

//destruktor
dataReader::~dataReader() {

	for (int i=0; i < (int) data.size(); i++) {
		delete data[i];
	}
	data.clear();
}

//ladowanie danych z pliku .csv
bool dataReader::loadDataFile(const char* fileName, int nI, int nT) {
	
	// wyczyszczenie ewentualnie wczesniejszych danych
	for (int i=0; i < (int) data.size(); i++) {
		delete data[i];
	}
	data.clear();
	tSet.clear();

	nInputs = nI;
	nTargets = nT;

	fstream inputFile;
	inputFile.open(fileName, ios::in);

	if (inputFile.is_open() ) {

		string line = "";

		while( !inputFile.eof() ) {
			getline(inputFile, line); 

			if(line.length() > 2) 
				processLine(line);	// w tym miejscu pole data jest napelnianie danymi
		}

		random_shuffle(data.begin(), data.end());

		trainingDataEndIndex = (int) ( 0.8 * data.size() );
		int gSize = (int) ( ceil(0.1 * data.size()) );
		int vSize = (int) ( data.size() - trainingDataEndIndex - gSize );

		// podzielenie danych na zbior treningowy, do generalizacji w trakcie uczenia i walidacji po skonczonym uczeniu
		for ( int i=0; i<trainingDataEndIndex; i++ ) {
			tSet.trainingSet.push_back( data [i] );
		}
		for ( int i=trainingDataEndIndex; i < trainingDataEndIndex + gSize; i++ ) {
			tSet.generalizationSet.push_back( data[i] );
		}
		for ( int i=trainingDataEndIndex + gSize; i < (int) data.size(); i++ ) {
			tSet.validationSet.push_back( data[i] );
		}

		cout << "Plik wejsciowy: " << fileName << "\nWczytywanie zakonczone: " << data.size() << " wzorow zaladowane" << endl; 
		return true;
	}
	else {
		cout << "Nie moge zaladowac pliku: " << fileName << endl;
		return false;
	}

}

//procesowanie pojedynczej linii
void dataReader::processLine(string &line) {

	double* pattern = new double[nInputs];
	double* target = new double[nTargets]; 

	char* cstr = new char[line.size()+1];
	char* t;
	strcpy(cstr, line.c_str());

	int i=0;
	t=strtok(cstr,",");

	while (t!=NULL && i < (nInputs + nTargets) ) {
		if (i<nInputs) {
			pattern[i] = atof(t);
		}
		else {
			target[i - nInputs] = atof(t);
		}

		t=strtok(NULL, ",");
		i++;
	}

	data.push_back( new dataEntry(pattern,target) );
}

// pobranie zbioru treningowego
trainingDataSet* dataReader::getTrainingDataSet() {
	return &tSet;
}