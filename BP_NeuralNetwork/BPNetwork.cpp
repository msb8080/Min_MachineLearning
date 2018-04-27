#include "BPNetwork.h"


//BP神经网络类构造函数
BpNet::BpNet()
{
	srand((unsigned)time(NULL));        // 随机数种子    
	error = 100.f;                      // error初始值，极大值即可

	// 初始化输入层
	for (int i = 0; i < innode; i++)
	{
		inputLayer[i] = new inputNode(); //新建输入层节点
		for (int j = 0; j < hidenode; j++)
		{
			inputLayer[i]->weight.push_back(get_11Random());  //随机生成节点之间的权值
			inputLayer[i]->wDeltaSum.push_back(0.f);	//
		}
	}

	// 初始化隐藏层
	for (int i = 0; i < hidelayer; i++)
	{
		if (i == hidelayer - 1) //最后一个隐含层
		{
			for (int j = 0; j < hidenode; j++)
			{
				hiddenLayer[i][j] = new hiddenNode();
				hiddenLayer[i][j]->bias = get_11Random();
				for (int k = 0; k < outnode; k++)
				{
					hiddenLayer[i][j]->weight.push_back(get_11Random());
					hiddenLayer[i][j]->wDeltaSum.push_back(0.f);
				}
			}
		}
		else	//除了最后一个隐含层外，其他隐含层不设置偏移量的累积和
		{
			for (int j = 0; j < hidenode; j++)
			{
				hiddenLayer[i][j] = new hiddenNode();
				hiddenLayer[i][j]->bias = get_11Random();
				for (int k = 0; k < hidenode; k++) { 
					hiddenLayer[i][j]->weight.push_back(get_11Random()); 
				}
			}
		}
	}

	// 初始化输出层
	for (int i = 0; i < outnode; i++)
	{
		outputLayer[i] = new outputNode();
		outputLayer[i]->bias = get_11Random();
	}
}

//正向传播
void BpNet::forwardPropagationEpoc()
{
	// forward propagation on hidden layer  隐含层上的正向传播
	for (int i = 0; i < hidelayer; i++)		 //使用状态函数和激活函数更新隐含层节点值
	{
		if (i == 0)  //第一层隐含层处理
		{
			for (int j = 0; j < hidenode; j++)
			{
				double sum = 0.f;  //记录第j个隐含层节点的状态
				for (int k = 0; k < innode; k++)
				{
					sum += inputLayer[k]->value * inputLayer[k]->weight[j];  
				}
				sum += hiddenLayer[i][j]->bias;
				hiddenLayer[i][j]->value = sigmoid(sum);  //第i个隐含层第j个节点的值更新
			}
		}
		else //除了第一层隐含层外，其他隐含层以前一层隐含层作为输入层
		{
			for (int j = 0; j < hidenode; j++)
			{
				double sum = 0.f;
				for (int k = 0; k < hidenode; k++)
				{
					sum += hiddenLayer[i - 1][k]->value * hiddenLayer[i - 1][k]->weight[j];
				}
				sum += hiddenLayer[i][j]->bias;
				hiddenLayer[i][j]->value = sigmoid(sum);
			}
		}
	}

	// forward propagation on output layer  输出层上的正向传播
	for (int i = 0; i < outnode; i++)  //使用状态函数和激活函数更新输出层节点值
	{
		double sum = 0.f;
		for (int j = 0; j < hidenode; j++)
		{
			sum += hiddenLayer[hidelayer - 1][j]->value * hiddenLayer[hidelayer - 1][j]->weight[i];
		}
		sum += outputLayer[i]->bias;
		outputLayer[i]->value = sigmoid(sum);
	}
}

void BpNet::backPropagationEpoc()  //反向传播
{
	// backward propagation on output layer
	// -- compute delta		计算误差△
	for (int i = 0; i < outnode; i++)
	{
		//计算训练指标函数E
		double tmpe = fabs(outputLayer[i]->value - outputLayer[i]->rightout);	//求浮点数x的绝对值
		error += tmpe * tmpe / 2; 

		//计算输出层节点反向传播误差
		outputLayer[i]->delta	
			= (outputLayer[i]->value - outputLayer[i]->rightout)*(1 - outputLayer[i]->value)*outputLayer[i]->value;
		//修改后
	/*	outputLayer[i]->delta
			= (outputLayer[i]->rightout - outputLayer[i]->value)*(1 - outputLayer[i]->value)*outputLayer[i]->value;*/
	}

	// backward propagation on hidden layer
	// -- compute delta
	for (int i = hidelayer - 1; i >= 0; i--)    // 隐藏层反向传播误差计算
	{
		if (i == hidelayer - 1)   //最后一个隐含层，其下一层为输出层
		{
			for (int j = 0; j < hidenode; j++)
			{
				double sum = 0.f;
				for (int k = 0; k < outnode; k++){ sum += outputLayer[k]->delta * hiddenLayer[i][j]->weight[k]; }
				hiddenLayer[i][j]->delta = sum * (1 - hiddenLayer[i][j]->value) * hiddenLayer[i][j]->value;
			}
		}
		else  //不是最后一个隐含层，其下一层也为隐含层
		{
			for (int j = 0; j < hidenode; j++)
			{
				double sum = 0.f;
				for (int k = 0; k < hidenode; k++){ sum += hiddenLayer[i + 1][k]->delta * hiddenLayer[i][j]->weight[k]; }
				hiddenLayer[i][j]->delta = sum * (1 - hiddenLayer[i][j]->value) * hiddenLayer[i][j]->value;
			}
		}
	}

	// backward propagation on input layer
	// -- update weight delta sum		
	for (int i = 0; i < innode; i++)
	{
		for (int j = 0; j < hidenode; j++)
		{
			inputLayer[i]->wDeltaSum[j] += inputLayer[i]->value * hiddenLayer[0][j]->delta;
			//更新权值？？？
		}
	}

	// backward propagation on hidden layer
	// -- update weight delta sum & bias delta sum
	for (int i = 0; i < hidelayer; i++)
	{
		if (i == hidelayer - 1)
		{
			for (int j = 0; j < hidenode; j++)
			{
				hiddenLayer[i][j]->bDeltaSum += hiddenLayer[i][j]->delta;
				for (int k = 0; k < outnode; k++)
				{
					//更新权值？？？
					hiddenLayer[i][j]->wDeltaSum[k] += hiddenLayer[i][j]->value * outputLayer[k]->delta;
				}
			}
		}
		else
		{
			for (int j = 0; j < hidenode; j++)
			{
				hiddenLayer[i][j]->bDeltaSum += hiddenLayer[i][j]->delta;
				for (int k = 0; k < hidenode; k++)
				{
					hiddenLayer[i][j]->wDeltaSum[k] += hiddenLayer[i][j]->value * hiddenLayer[i + 1][k]->delta;
				}
			}
		}
	}

	// backward propagation on output layer
	// -- update bias delta sum
	for (int i = 0; i < outnode; i++) outputLayer[i]->bDeltaSum += outputLayer[i]->delta;
}


//从正确的样本中训练出精确的BP神经网络
void BpNet::training(static vector<sample> sampleGroup, double threshold)
{
	int sampleNum = sampleGroup.size(); //获取训练样本总个数

	while (error > threshold)  //当实际误差和不小于目标误差，则继续使用样本训练BP神经网络
		//for (int curTrainingTime = 0; curTrainingTime < trainingTime; curTrainingTime++)
	{
		cout << "training error: " << error << endl; //打印出当前神经网络总体误差值
		error = 0.f;	//用于记录每次训练网络的总误差值的变量清零
		// initialize delta sum
		//初始化输入层的wDeltaSum
		for (int i = 0; i < innode; i++) inputLayer[i]->wDeltaSum.assign(inputLayer[i]->wDeltaSum.size(), 0.f);
		//初始化隐含层的wDeltaSum
		for (int i = 0; i < hidelayer; i++){
			for (int j = 0; j < hidenode; j++)
			{
				hiddenLayer[i][j]->wDeltaSum.assign(hiddenLayer[i][j]->wDeltaSum.size(), 0.f);
				hiddenLayer[i][j]->bDeltaSum = 0.f;
			}
		}
		//初始化输出层的bias偏移量误差值的总和
		for (int i = 0; i < outnode; i++) outputLayer[i]->bDeltaSum = 0.f;

		//
		for (int iter = 0; iter < sampleNum; iter++)  //一个样本训练一次网络，则一组训练为sampleNum次训练
		{
			setInput(sampleGroup[iter].in); //使用样本初始化输入层节点值
			setOutput(sampleGroup[iter].out);	//使用样本初始化输出层的期望输出值
			forwardPropagationEpoc();
			backPropagationEpoc();	
		}

		// backward propagation on input layer
		// -- update weight     更新输入层到隐含层权值
		for (int i = 0; i < innode; i++)
		{
			for (int j = 0; j < hidenode; j++)
			{
				inputLayer[i]->weight[j] -= learningRate * inputLayer[i]->wDeltaSum[j] / sampleNum;

			}
		}

		// backward propagation on hidden layer
		// -- update weight & bias   更新隐含层的权值和偏移量
		for (int i = 0; i < hidelayer; i++)
		{
			if (i == hidelayer - 1)
			{
				for (int j = 0; j < hidenode; j++)
				{
					// bias
					hiddenLayer[i][j]->bias -= learningRate * hiddenLayer[i][j]->bDeltaSum / sampleNum;

					// weight
					for (int k = 0; k < outnode; k++)
					{
						hiddenLayer[i][j]->weight[k] -= learningRate * hiddenLayer[i][j]->wDeltaSum[k] / sampleNum;
					}
				}
			}
			else
			{
				for (int j = 0; j < hidenode; j++)
				{
					// bias
					hiddenLayer[i][j]->bias -= learningRate * hiddenLayer[i][j]->bDeltaSum / sampleNum;

					// weight
					for (int k = 0; k < hidenode; k++)
					{
						hiddenLayer[i][j]->weight[k] -= learningRate * hiddenLayer[i][j]->wDeltaSum[k] / sampleNum;
					}
				}
			}
		}

		// backward propagation on output layer
		// -- update bias
		for (int i = 0; i < outnode; i++)
		{
			outputLayer[i]->bias -= learningRate * outputLayer[i]->bDeltaSum / sampleNum;
		}
	}
}

//测试样本
void BpNet::predict(vector<sample>& testGroup)
{
	int testNum = testGroup.size();  //获取样本总个数

	for (int iter = 0; iter < testNum; iter++)	
	{
		testGroup[iter].out.clear();
		setInput(testGroup[iter].in);

		// forward propagation on hidden layer
		for (int i = 0; i < hidelayer; i++)
		{
			if (i == 0)
			{
				for (int j = 0; j < hidenode; j++)
				{
					double sum = 0.f;
					for (int k = 0; k < innode; k++)
					{
						sum += inputLayer[k]->value * inputLayer[k]->weight[j];
					}
					sum += hiddenLayer[i][j]->bias;
					hiddenLayer[i][j]->value = sigmoid(sum);
				}
			}
			else
			{
				for (int j = 0; j < hidenode; j++)
				{
					double sum = 0.f;
					for (int k = 0; k < hidenode; k++)
					{
						sum += hiddenLayer[i - 1][k]->value * hiddenLayer[i - 1][k]->weight[j];
					}
					sum += hiddenLayer[i][j]->bias;
					hiddenLayer[i][j]->value = sigmoid(sum);
				}
			}
		}

		// forward propagation on output layer
		for (int i = 0; i < outnode; i++)
		{
			double sum = 0.f;
			for (int j = 0; j < hidenode; j++)
			{
				sum += hiddenLayer[hidelayer - 1][j]->value * hiddenLayer[hidelayer - 1][j]->weight[i];
			}
			sum += outputLayer[i]->bias;
			outputLayer[i]->value = sigmoid(sum);
			testGroup[iter].out.push_back(outputLayer[i]->value);
		}
	}
}


//将样本中的输入值更新到输入层节点上
void BpNet::setInput(static vector<double> sampleIn)
{
	for (int i = 0; i < innode; i++) inputLayer[i]->value = sampleIn[i];
}


//将样本中的正确输出更新到输出层节点上
void BpNet::setOutput(static vector<double> sampleOut)
{
	for (int i = 0; i < outnode; i++) outputLayer[i]->rightout = sampleOut[i];
}