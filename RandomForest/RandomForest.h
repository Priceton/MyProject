// Author: Benwei Gong
/* Date: 2014.5.15
*/ 
// Instruction: This Program is used for 2-Classification.

#ifndef _RandomForest_H
#define _RandomForest_H

#include <vector>
#include <time.h>
#include <string>
#include <iostream>

using namespace std;

struct SampleSet
{
	double **Data;
	bool *Label;
	int numSamples;
	int numDims;
	vector<int> SampleSetIndex;
};

struct BestResult
{
	int FEAIndex;
	double FEAThreshold;
	vector<int> LEFTIndex;
	vector<int> RIGHTIndex;

	double INFOGain;

	void EMPTY()
	{
		FEAIndex = -1;
		FEAThreshold = 0.0;
		INFOGain = 0.0;
		LEFTIndex.clear();
		RIGHTIndex.clear();
	}
};

class Node
{
public:
	int numPoints;
	bool IsLeaf;
	int feaIndex;
	double feaThreshold;
	double Entropy;

	vector<int> SampleIndexNode;

	void Empty();
	void Initialization();
	void CopyIndex(vector<int> sampleIndex);

//	double GetRate(SampleSet *NodeSet);

	double GetInfoGain(SampleSet *NodeSet, vector<int> leftIndex, vector<int> rightIndex);

	void GetMaxMin(SampleSet *NodeSet, int feaTemp, double &MaxValue, double &MinValue);

	void SplitNode(SampleSet *NodeSet, int feaTemp,
		double &Threshold, vector<int> &LeftIndex, vector<int> &RightIndex);
};

struct TestNode
{
	char state;
	int numIndex;
	int feaIndex;
	double feaThreshold;
	double rate;

	int leftIndex;
	int rightIndex;
};

struct ForestParams
{
	int numOfTrees;
	int maxDepth;
	int minSize;

	int numFeatures;
	double Precision;

	int numOfSamplesPerTree;
	int numOfFeaturesPerTree;

	int numOfForestSamples;
	vector<int> SampleForestIndex;
};

class Forest
{
public:
	Node *Tree;
	ForestParams Params;
	const char* Forest_folder;

	void GetDataSet(const char* fileName, int feaDim, SampleSet &DataSet);
	
	void SetParams(int numSamples,int Dim, double precision);

	void GetForestSet(SampleSet *ForestSet);

	void TrainForest(SampleSet *ForestSet);
 
	void TestForest(const char* TestFile, vector<double> &TestResult);

private:
	void FeatureSelection(vector<int> &randFeaIndexPerTree);
	void SampleSeletcion(vector<int> &randSampleIndexPerTree);
	void TrainTree(SampleSet *TreeSet, int numOfNodes, vector<int> TreeSampleIndex, vector<int> TreeFeatureIndex);
	void WriteTree(SampleSet *TreeSet, int treeID, int numOfNodes);
};



double GetRate(SampleSet *NodeSet, vector<int> sampleIndexVec);

double GetEntropy(SampleSet *NodeSet, vector<int> sampleIndexVec);

bool TrainNode(SampleSet *NodeSet, Node Nd, ForestParams params, vector<int> randomFeatures, BestResult *result);



#endif
