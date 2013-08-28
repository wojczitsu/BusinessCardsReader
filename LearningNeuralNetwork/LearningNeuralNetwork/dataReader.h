#pragma once

#include <vector>
#include <string>

class dataEntry {

public:	
	double* pattern;	//dane wejsciowe - wzor
	double* target;		//dane wyjsciowe - odpowiedz

	dataEntry(double* p, double* t): pattern(p), target(t) {}
		
	~dataEntry() {				
		delete[] pattern;
		delete[] target;
	}
};

class trainingDataSet {

public:
	std::vector<dataEntry*> trainingSet;
	std::vector<dataEntry*> generalizationSet;
	std::vector<dataEntry*> validationSet;

	trainingDataSet() {};
	~trainingDataSet() {};

	void clear() {
		trainingSet.clear();
		generalizationSet.clear();
		validationSet.clear();
	}
};

class dataReader {

private:
	std::vector<dataEntry*> data;	// dane z pliku
	int nInputs;
	int nTargets;

	trainingDataSet tSet; // dane do trenowania
	int trainingDataEndIndex; 

public: 
	dataReader() {} ;
	~dataReader() ;

	bool loadDataFile(const char* fileName, int nI, int nT );
	trainingDataSet* getTrainingDataSet(); 

private:
	void processLine( std::string &line );
};

