//
/*
*/
//

#include "RandomForest.h"

#include <vector>
#include <string>
#include <iostream>
#include <fstream>

using namespace std;

int main(int argc, char* argv[])
{
	if (argc != 4)
	{
		cout<<"Error Parameters, Please Check! "<<endl;
		return -1;
	}

	SampleSet TrainSet;
	const char *TrainDatafile = argv[1];
	const char *forestFold = argv[2];

//	ForestParams params;
	Forest *forest_1 = new Forest();

	int FeatureDim = 150;
	double precision = 0.2;
	ForestParams Params;
	
	cout<<"Get the Train DataSet: "<<endl;
	forest_1->GetDataSet(TrainDatafile, FeatureDim, TrainSet);
	int TrainSampleNum = TrainSet.numSamples;

	//for 2 class posRate < 0.03, Entropy(0.03) = 0.2,Entropy(0.05) = 0.28 .
	forest_1->SetParams(TrainSampleNum, FeatureDim, precision);

	forest_1->Forest_folder = forestFold;
	
	cout<<"Training Parameters: "<<endl;

	cout<<"Number of trees: "<<forest_1->Params.numOfTrees<<endl;
	cout<<"Minimum Node size: "<<forest_1->Params.minSize<<endl;
	cout<<"Maximum Tree Depth: "<<forest_1->Params.maxDepth<<endl;
	cout<<"The precision: "<<forest_1->Params.Precision<<endl;
	cout<<"Number of Training Dimension: "<<forest_1->Params.numFeatures<<endl;
	cout<<"Number of Training Samples: "<<forest_1->Params.numOfForestSamples<<endl;
	cout<<"The Result will be stored in: "<<forest_1->Forest_folder<<endl;
	cout<<endl;
	cout<<"Number of Features per Tree: "<<forest_1->Params.numOfFeaturesPerTree<<endl;
	cout<<"Number of Train Samples per Tree: "<<forest_1->Params.numOfSamplesPerTree<<endl;
	cout<<endl;

	forest_1->TrainForest(&TrainSet);

	const char *TestDataFile = argv[3];
	vector<double> testResult;
	forest_1->TestForest(TestDataFile, testResult );
	
	return 0;
}

//	int SampleNum = 0;

/*	ifstream fin;
	fin.open(argv[1]);
	string t;
	while(!fin.eof())
	{
		getline(fin, t);
		if(t.length()>0)
			SampleNum++;
	}
	fin.clear();
	fin.seekg(0, ios::beg);

	SampleSet TrainSet;
	TrainSet.numDims = FeatureDim;
	TrainSet.numSamples = SampleNum;

//	TrainSet = new SampleSet;
	TrainSet.Data = new double *[SampleNum];
	for (int n=0; n<SampleNum; ++n)
	{
		TrainSet.Data[n] = new double[FeatureDim];
	}

	double *Lab = new double[SampleNum];
	TrainSet.Label = new bool[SampleNum];

	if (!fin)
	{
		cout<<"Cannot Open the file. "<<endl;
		return -1;
	}
	for (int n=0; n<SampleNum; ++n)
	{
		fin>>Lab[n];
//		if(n%1000 == 0)
//			cout<<n<<"---->"<<Lab[n]<<endl;
		for (int k=0; k<FeatureDim; ++k)
		{
			fin>>TrainSet.Data[n][k];
		}
	}

	for (int n=0; n<SampleNum; ++n)
	{
		TrainSet.Label[n] = (bool)Lab[n];
		TrainSet.SampleSetIndex.push_back(n);
	}
	delete []Lab;

//	fin.close();
	
	cout<<"The Number of Training Samples :"<<TrainSet.numSamples<<endl;
	cout<<"The Dimension of Samples :"<<TrainSet.numDims<<endl;

*/