#include <iostream>
#include <fstream>
#include <ostream>
#include <stdio.h>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <time.h>

#include "AdaBoost.h"
using namespace std;

int main(int argc, char *argv[])
{

	float x[10][2];

	short y[]={1,1,-1,-1,1,-1,1,1,-1,-1};

	std::vector<float> X;

	x[0][0] = 0.1;
	x[1][0] = 0.15;
	x[2][0] = 0.2;
	x[3][0] = 0.25;
	x[4][0] = 0.3;
	x[5][0] = 0.6;
	x[6][0] = 0.6;
	x[7][0] = 0.8;
	x[8][0] = 0.9;
	x[9][0] = 0.9;

	x[0][1] = 0.5;
	x[1][1] = 0.2;
	x[2][1] = 0.1;
	x[3][1] = 0.55;
	x[4][1] = 0.8;
	x[5][1] = 0.45;
	x[6][1] = 0.9;
	x[7][1] = 0.75;
	x[8][1] = 0.2;
	x[9][1] = 0.82;


	int totalSample = 10;
	int NFeature = 2;

	for(int i=0;i<totalSample;i++)
		for(int j=0;j<NFeature;j++)
			X.push_back(x[i][j]);
	vector<double>::size_type size=X.size();
	for (int i=0;i<size; i++)
	{
		cout<<"X["<<i<<"] = "<<X[i]<<endl;
	}
	
	cout<<"the size of vector is "<<size<<endl;


	int iterN = 5;
	char fileName[]="./classifier.txt";

	AdaBoostTrain(&X[0],&y[0],totalSample,NFeature,iterN,fileName);


	return 0;
}
