// Author: Benwei Gong
/* Date: 2014.6.12
*/
// Instruction: This Program is used for 2-Classification.

#include "RandomForest.h"

#include <fstream>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>


using namespace std;

void Node::Initialization()
{
	feaIndex = -1;
	feaThreshold = 0.0;
	IsLeaf = true;
	numPoints = 0;
	Entropy = 0.0;
}

void Node::Empty()
{
	SampleIndexNode.clear();
	feaIndex = -1;
	feaThreshold = 0.0;
	Entropy = 0.0;
	IsLeaf = false;
	numPoints = 0;
}

void Node::CopyIndex(vector<int> sampleIndex)
{
	SampleIndexNode = sampleIndex;
	numPoints = (int)SampleIndexNode.size();
}

double Node::GetInfoGain(SampleSet *NodeSet, vector<int> leftIndex, vector<int> rightIndex)
{
	double infoGain = 0.0;
	double Entropy = GetEntropy(NodeSet, SampleIndexNode);
	double LeftEntropy = GetEntropy(NodeSet, leftIndex);
	double RightEntropy = GetEntropy(NodeSet, rightIndex);

	double leftRate = double(leftIndex.size())/numPoints;
	double rightRate = 1 - leftRate;

	infoGain = Entropy - leftRate*LeftEntropy - rightRate*RightEntropy;

	return infoGain;
}

void Node::GetMaxMin(SampleSet *NodeSet, int feaTemp, double &MaxValue, double &MinValue)
{
	MaxValue = -10000.0;
	MinValue = 10000.0;

	for (int n=0; n<numPoints; ++n)
	{
		int row = SampleIndexNode[n];
		if(NodeSet->Data[row][feaTemp] >= MaxValue)
			MaxValue = NodeSet->Data[row][feaTemp];
		if (NodeSet->Data[row][feaTemp] < MinValue)
			MinValue = NodeSet->Data[row][feaTemp];
	}
}

void Node::SplitNode(SampleSet *NodeSet, int feaTemp,
					 double &Threshold, vector<int> &LeftIndex, vector<int> &RightIndex)
{
	double maxValue = 0.0;
	double minValue = 0.0;
	double BestInfoGain = 0.0;
	vector<int> leftIndex;
	vector<int> rightIndex;

	GetMaxMin(NodeSet,feaTemp, maxValue, minValue);

	double tempThreshold = minValue;
	double step = 2.0*(maxValue - minValue)/numPoints;

	while(tempThreshold < maxValue)
	{
		tempThreshold += step;
		for (int i=0; i<SampleIndexNode.size(); ++i)
		{
			int row = SampleIndexNode[i];

			if (NodeSet->Data[row][feaTemp] < tempThreshold)
			{
				leftIndex.push_back(row);
			}else
			{
				rightIndex.push_back(row);
			}
		}

		double infoGain = GetInfoGain(NodeSet, leftIndex, rightIndex);
		if (infoGain > BestInfoGain)
		{
			BestInfoGain = infoGain;
			Threshold = tempThreshold;
			LeftIndex = leftIndex;
			RightIndex = rightIndex;
			IsLeaf = false;
		}

		leftIndex.clear();
		rightIndex.clear();
	}
}



double GetRate(SampleSet *NodeSet, vector<int> sampleIndexVec)
{
	int posNum = 0;
	for(vector<int>::iterator sampleIter = sampleIndexVec.begin(); sampleIter != sampleIndexVec.end(); ++sampleIter)
	{
		int s = *sampleIter;
		if (NodeSet->Label[s]){
			posNum++;
		}
	}
	double Rate =(double)(posNum)/sampleIndexVec.size();

	return Rate;
}

double GetEntropy(SampleSet *NodeSet, vector<int> sampleIndexVec)
{
	int pos = 0;
	for (vector<int>::iterator iter = sampleIndexVec.begin(); iter != sampleIndexVec.end(); ++iter)
	{
		int row = *iter;
		if (NodeSet->Label[row] != 0)
			pos++;
	}
	double posRate = (double)pos/sampleIndexVec.size();
	double negRate = 1 - posRate;

	double result = (-posRate*log(posRate) - negRate*log(negRate))/log(2.0);
	
	return result;
}

bool TrainNode(SampleSet *NodeSet, Node Nd, ForestParams params, vector<int> randomFeatures, BestResult *result)
{
	int numPoint = Nd.numPoints;
	int feaIndexBest = -1;
	double InfoGainBest = 0.0;
	double ThresholdBest = 0.0;
	bool ISLEAF;

	vector<int> LeftIndex_Best, LeftIndex;
	vector<int> RightIndex_Best, RightIndex;

	Nd.Entropy = GetEntropy(NodeSet, Nd.SampleIndexNode);

	if (Nd.Entropy < params.Precision)
		ISLEAF = true;

	else
	{
		int feaIndex = -1;
		double feaThreshold = 0.0;
		double infoGain = 0.0;

		// Choose the best features for splitting.
		for (int i=0; i< params.numOfFeaturesPerTree; ++i)
		{
			feaIndex = randomFeatures[i];
			Nd.SplitNode(NodeSet, feaIndex, feaThreshold, LeftIndex, RightIndex);
			infoGain = Nd.GetInfoGain(NodeSet, LeftIndex, RightIndex);

			if (infoGain > InfoGainBest)
			{
				InfoGainBest = infoGain;
				feaIndexBest = feaIndex;
				ThresholdBest = feaThreshold;
				ISLEAF = false; 
				LeftIndex_Best = LeftIndex;
				RightIndex_Best = RightIndex;
			}

			LeftIndex.clear();
			RightIndex.clear();
		}
		result->FEAIndex = feaIndexBest;
		result->FEAThreshold = ThresholdBest;
		result->LEFTIndex = LeftIndex_Best;
		result->RIGHTIndex = RightIndex_Best;

		LeftIndex.clear();
		RightIndex.clear();
		LeftIndex_Best.clear();
		RightIndex_Best.clear();
	}

	return ISLEAF;
}


// Definition for the Forest.
void Forest::GetDataSet(const char* fileName, int FeatureDim, SampleSet &DataSet)
{
	int SampleNum = 0;

	ifstream dataIn;
	dataIn.open(fileName);
	string t;
	while(!dataIn.eof())
	{
		getline(dataIn, t);
		if(t.length()>0)
			SampleNum++;
	}
	dataIn.clear();
	dataIn.seekg(0, ios::beg);

//	SampleSet TestSet;
	DataSet.numDims = FeatureDim;
	DataSet.numSamples = SampleNum;

	//	TrainSet = new SampleSet;
	DataSet.Data = new double *[SampleNum];
	for (int n=0; n<SampleNum; ++n)
	{
		DataSet.Data[n] = new double[FeatureDim];
	}

	double *Lab = new double[SampleNum];
	DataSet.Label = new bool[SampleNum];

	if (!dataIn)
	{
		cout<<"Cannot Open the file. "<<endl;
		return;
	}
	for (int n=0; n<SampleNum; ++n)
	{
		dataIn>>Lab[n];
		for (int k=0; k<FeatureDim; ++k)
		{
			dataIn>>DataSet.Data[n][k];
		}
	}
	for (int n=0; n<SampleNum; ++n)
	{
		DataSet.Label[n] = (bool)Lab[n];
		DataSet.SampleSetIndex.push_back(n);
	}
	delete []Lab;

	//	fin.close();

	cout<<"The Number of Data Samples :"<<DataSet.numSamples<<endl;
	cout<<"The Dimension of Samples :"<<DataSet.numDims<<endl;
}

void Forest::GetForestSet(SampleSet *ForestSet)
{
	vector<int>	SampleIndex;

	int TotalSample = ForestSet->numSamples;
	for (int i=0; i<TotalSample; ++i){
		SampleIndex.push_back(i);
	}
	random_shuffle(SampleIndex.begin(), SampleIndex.end());
	for (int k=0; k<(int)TotalSample/2; ++k)
	{
		int r = SampleIndex[k];
		Params.SampleForestIndex.push_back(r);
	}
}

void Forest::SampleSeletcion(vector<int> &randSampleIndexPerTree)
{
	random_shuffle(Params.SampleForestIndex.begin(), Params.SampleForestIndex.end());
	for (int k=0; k<Params.numOfSamplesPerTree; ++k)
	{
		int row = Params.SampleForestIndex[k];
		randSampleIndexPerTree.push_back(row);
	}
}

void Forest::FeatureSelection(vector<int> &randFeaIndexPerTree)
{
	vector<int>	TotalFeatures;
	for (int f=0; f<Params.numFeatures; ++f){
		TotalFeatures.push_back(f);
	}
	random_shuffle(TotalFeatures.begin(), TotalFeatures.end());
	for (int i=0; i<Params.numOfFeaturesPerTree; ++i)
	{
		int fea = TotalFeatures[i];
		randFeaIndexPerTree.push_back(fea);
	}
}

void Forest::SetParams(int numSamples, int Dim, double precision)
{
	Params.maxDepth = 11;
	Params.minSize = 30;
	Params.numOfTrees = 100;
	Params.numFeatures = Dim;
	Params.Precision = precision;
	Params.numOfForestSamples = numSamples;

	Params.numOfFeaturesPerTree = 12;
	Params.numOfSamplesPerTree = 8000;
}

void Forest::TrainTree(SampleSet *TreeSet, int numOfNodes, vector<int> TreeSampleIndex, vector<int> TreeFeatureIndex)
{
	for (int i=0; i<numOfNodes; ++i)
	{
		Tree[i].Initialization();
	}
	bool *BeTrained;
	BeTrained = new bool[(int)pow(2.0, Params.maxDepth)-1];
	for (int i=0; i<(int)pow(2.0, Params.maxDepth)-1; ++i)
	{
		BeTrained[i] = false;
	}
	BeTrained[0] = true;
	int numOfPointTemp = 0;
	bool ISLEAF;
	BestResult result;

	Tree[0].CopyIndex(TreeSampleIndex);

	for (int treeLevel = 0; treeLevel<Params.maxDepth; ++treeLevel)
	{
		for (int nodeIndex = (int)pow(2.0, treeLevel) -1; 
			nodeIndex<(int)pow(2.0, treeLevel+1) -1; nodeIndex++)
		{
			if(!BeTrained[nodeIndex])
				continue;
			numOfPointTemp = (int)Tree[nodeIndex].SampleIndexNode.size();

			if (numOfPointTemp < Params.minSize || treeLevel == Params.maxDepth -1)
			{
				ISLEAF = true;
				Tree[nodeIndex].IsLeaf = ISLEAF;
				Tree[nodeIndex].feaIndex = -1;
				Tree[nodeIndex].feaThreshold = 0.0;
				Tree[nodeIndex].numPoints = numOfPointTemp;
			}
			else
			{
				result.EMPTY();
				ISLEAF = TrainNode(TreeSet, Tree[nodeIndex], Params, TreeFeatureIndex, &result);

				Tree[nodeIndex].feaIndex = result.FEAIndex;
				Tree[nodeIndex].feaThreshold = result.FEAThreshold;
				Tree[nodeIndex].IsLeaf = ISLEAF;
				
				if(!ISLEAF)
				{
					BeTrained[(nodeIndex+1) *2 -1] = true;
					BeTrained[(nodeIndex+1) *2 ] = true;
					Tree[(nodeIndex+1) *2 -1].CopyIndex(result.LEFTIndex);
					Tree[(nodeIndex+1) *2].CopyIndex(result.RIGHTIndex);
				}
				else
				{
					BeTrained[(nodeIndex+1) *2 -1] = false;
					BeTrained[(nodeIndex+1) *2 ] = false;
				}
			}
		}
	}
}

void Forest::WriteTree(SampleSet *TreeSet, int TreeID, int numOfNodes)
{
	char file_name[1024];
	sprintf(file_name, "%s/Tree_%d.txt", Forest_folder, TreeID);

	ofstream filesOut(file_name, ios::out);
	//fileOut the number of nodes and training samples of each tree.
	filesOut<<numOfNodes<<" "<<Params.numOfSamplesPerTree<<endl;

	for (int nodeIndex=0; nodeIndex<numOfNodes; nodeIndex++)
	{
		if (Tree[nodeIndex].numPoints == 0)
		{
//			continue;
			//NodeID, 中间节点， 特征选择， 特征阈值， 正样本比例，总样本数， LeftId, RightId.
			//Tree[nodeIndex] is not used.
			filesOut<<nodeIndex<<" "<<'N'<<" "<<-1<<" "<<0<<" "<<-1.0<<" "<<0<<" "<<-1<<" "<<-1<<endl;
		}
		else
		{
			double classRate = GetRate(TreeSet, Tree[nodeIndex].SampleIndexNode);
			if (!Tree[nodeIndex].IsLeaf)
			{
				//NodeID, I/L, feaIndex, feaThreshold, Rate, LeftId, RightId.
				filesOut<<nodeIndex<<" "<<'I'<<" "<<Tree[nodeIndex].feaIndex<<" "<<Tree[nodeIndex].feaThreshold<<" "<<
					classRate<<" "<<Tree[nodeIndex].numPoints<<" "<<(nodeIndex+1)*2-1<<" "<<(nodeIndex+1)*2<<endl;	
			}
			else
			{
				//NodeID, Middle Node, featureNum, feaThreshold, Rate, LeftId, RightId.
				filesOut<<nodeIndex<<" "<<'L'<<" "<<-1<<" "<<0<<" "<<classRate<<" "<<Tree[nodeIndex].numPoints<<" "<<-1<<" "<<-1<<endl;
			}
		}

	}

	filesOut.close();

}

void Forest::TrainForest(SampleSet *ForestSet)
{
	//Get the Forest Training set.
	GetForestSet(ForestSet);
//	Forest_folder = forest_File;

	vector<int> randSampleIndexPerTree;
	vector<int> randFeatureIndexPerTree;
	
	for (int TreeIter=0; TreeIter<Params.numOfTrees; ++TreeIter)
	{
		SampleSeletcion(randSampleIndexPerTree);
		FeatureSelection(randFeatureIndexPerTree);

		int numOfNodes = (int)pow(2.0, Params.maxDepth) - 1;
		Tree = new Node[numOfNodes];
		TrainTree(ForestSet, numOfNodes, randSampleIndexPerTree, randFeatureIndexPerTree);
		WriteTree(ForestSet, TreeIter, numOfNodes);

		randSampleIndexPerTree.clear();
		randFeatureIndexPerTree.clear();

		/* clear the tree - memory management */
		for (int k=0;k<numOfNodes;k++)
			Tree[k].Empty();
		if (TreeIter % 50 == 0){
			cout<<"Training tree "<<TreeIter<<" / "<<Params.numOfTrees<<" done."<<endl;
		}

	}
	
}

void Forest::TestForest(const char* TestFile, vector<double> &TestResult)
{
	SampleSet TestSet;
	cout<<"Getting the Testing DataSet: "<<endl;
	GetDataSet(TestFile, Params.numFeatures, TestSet);
	int TestSampleNum = TestSet.numSamples;
	
	//Input for Forest.
	int NumNode = 0;
	int NumTree = Params.numOfTrees;

	vector<char> StateVec;
	vector<int>	feaIndexVec;
	vector<double> feaThresholdVec;
	vector<double> rateVec;
	vector<int>	leftVec;
	vector<int> rightVec;

	vector<vector<double> > rateResult;
	vector<double> rateTemp;

	char forestFile[512];
	for (int t=0; t<NumTree; ++t)
	{
		sprintf(forestFile, "%s\\Tree_%d.txt", Forest_folder, t);
//		cout<<forestFile<<endl;

		char chVal;
		int intVal;
		double fVal;
		ifstream fin;
		fin.open(forestFile);
		if (fin.good())
		{
			fin>>intVal;
//			cout<<"The number of Node is: "<<intVal<<endl;
			fin>>intVal;
		}

		while (fin.good())
		{
			fin>>intVal;
			if(!fin.good())
				break;
			fin>>chVal;
			StateVec.push_back(chVal);
			fin>>intVal;
			feaIndexVec.push_back(intVal);
			fin>>fVal;
			feaThresholdVec.push_back(fVal);
			fin>>fVal;
			rateVec.push_back(fVal);
			fin>>intVal;
			fin>>intVal;
			leftVec.push_back(intVal);
			fin>>intVal;
			rightVec.push_back(intVal);
		}
		fin.close();
		NumNode = StateVec.size();

		TestNode* Tree = new TestNode[NumNode];
		for (int i=0; i<NumNode; ++i)
		{
			Tree[i].state = StateVec[i];
			Tree[i].numIndex = i;
			Tree[i].feaIndex = feaIndexVec[i];
			Tree[i].feaThreshold = feaThresholdVec[i];
			Tree[i].rate = rateVec[i];
			Tree[i].leftIndex = leftVec[i];
			Tree[i].rightIndex = rightVec[i];
		}

		for (int n=0; n<TestSampleNum; ++n)
		{
			int k = 0;
			while (k < NumNode)
			{	
				if (Tree[k].state == 'N')
					break;
				else if (Tree[k].state == 'L')
				{
					rateTemp.push_back(Tree[k].rate);
					break;
				}
				else if (Tree[k].state == 'I')
				{
					int feaTemp = Tree[k].feaIndex;
					if (TestSet.Data[n][feaTemp] <= Tree[k].feaThreshold)
						k = Tree[k].leftIndex;
					else
						k = Tree[k].rightIndex;
				}

			}
		}
		rateResult.push_back(rateTemp);

		rateTemp.clear();
		StateVec.clear();
		feaIndexVec.clear();
		feaThresholdVec.clear();
		leftVec.clear();
		rightVec.clear();
		rateVec.clear();
	}

	vector<bool>	LabResult;
	vector<double>	sumRate;
	sumRate.resize(TestSampleNum);

	int errCount = 0;
	for (int n=0; n<sumRate.size(); ++n)
	{
		for (int k=0; k<NumTree; ++k)
			sumRate[n] += rateResult[k][n];

		sumRate[n] /= NumTree;
		if(sumRate[n] > 0.5)
			LabResult.push_back(1);
		else
			LabResult.push_back(0);

		if(LabResult[n] != TestSet.Label[n]) 
			errCount++;
	}

	cout<<"The errorCount is: "<<errCount<<endl;

}
