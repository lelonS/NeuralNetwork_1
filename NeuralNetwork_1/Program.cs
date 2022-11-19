// See https://aka.ms/new-console-template for more information


using NeuralNetwork_1;






// TODO BATCH
static void TrainNetwork(NeuralNetwork neuralNetwork, DataPoint[] trainingData, DataPoint[] testData, double learningRate, double targetCost, int batchSize = 0)
{
    double cost = 100;
    int iteration = 0;
    DataPoint[] data = trainingData;
    Random rnd = new();
    Console.WriteLine("Start");
    while (cost > targetCost)
    {
        if (batchSize > 0)
        {
            data = new DataPoint[batchSize];
            for (int i = 0; i < batchSize; i++)
            {
                data[i] = trainingData[rnd.Next(trainingData.Length)];
            }
        }
        cost = neuralNetwork.Learn(data, learningRate);

        if (iteration % 1000 == 0)
        {
            Console.WriteLine("Training cost: " + iteration + " ::: " + cost);
        }
        iteration++;
    }
}

void TestInputs(NeuralNetwork network)
{
    while (true)
    {
        Console.WriteLine();
        double[] input = new double[network.amountInputs];
        for (int i = 0; i < network.amountInputs; i++)
        {
            Console.Write("Input " + i + ": ");
            input[i] = double.Parse(Console.ReadLine());
            Console.WriteLine();
        }
        double[] prediction = network.Prediction(input);
        for (int i = 0; i < prediction.Length; i++)
        {
            Console.WriteLine(i + " : " + prediction[i]);
        }
    }
}






int[] layers = new int[] { 64, 64 };
NeuralNetwork neuralNetwok = new(layers);
DataPoint[] trainingData = new DataPoint[10000];

Random rnd = new Random(1);

string[] seq = new string[trainingData.Length + 1];
for (int i = 0; i < seq.Length; i++)
{
    double r = rnd.NextDouble();
    byte[] t = BitConverter.GetBytes(r);
    string s = "";
    for(int j = 0; j < t.Length; j++)
    {
        s += Convert.ToString(t[j], 2).PadLeft(8, '0');
    }
    seq[i] = s;
    //Console.WriteLine(seq[i]);
}
Console.WriteLine(seq[0] + " " + seq[0].Length + " ");
Console.WriteLine(seq[1] + " " + seq[1].Length + " ");
Console.WriteLine(seq[2] + " " + seq[2].Length + " ");


for (int i = 0; i < trainingData.Length; i++)
{
    double[] inputs = new double[64];

    double[] outputs = new double[64];

    for (int j = 0; j < 64; j++)
    {
        // Console.WriteLine(seq[i].Length);
        inputs[j] = double.Parse(seq[i][j].ToString());
        outputs[j] = double.Parse(seq[i][j].ToString());
    }

    trainingData[i] = new DataPoint(inputs, outputs);
}

Console.WriteLine(trainingData[0].inputs[0].ToString() + " " + seq[0]);
TrainNetwork(neuralNetwok, trainingData, trainingData, 0.01, 0.001, 20);
TestInputs(neuralNetwok);


//int[] order = new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

//DataPoint[] dataPoints = new DataPoint[10*10];


//// Generate DataPoints
//string[] orderStr = new string[order.Length];
//for (int i = 0; i < order.Length; i++)
//{
//    string str = Convert.ToString(order[i], 2);
//    str = str.PadLeft(4, '0');
//    orderStr[i] = str;
//    Console.WriteLine(i + " " + str);
//}

//for (int k = 0; k < order.Length; k++)
//{
//    for (int i = 0; i < orderStr.Length; i++)
//    {
//        double[] input = new double[4];
//        for (int j = 0; j < 4; j++)
//        {
//            input[j] = double.Parse(orderStr[i][j].ToString());
//        }
//        double[] expectedOutput = new double[10];
//        expectedOutput[i] = 1;
//        DataPoint newDataPoint = new(input, expectedOutput);
//        Console.WriteLine(newDataPoint.ToString());
//        dataPoints[i+k*10] = newDataPoint;
//    }
//}


//int[] layers = new int[] { 4, 4, 4, 10};
//NeuralNetwork neuralNetwok = new(layers);
//double[] prediction = neuralNetwok.Prediction(new double[] { 0, 0, 0, 0 });
//for (int i = 0; i < prediction.Length; i++)
//{
//    Console.WriteLine(i + " " + prediction[i]);
//}
//// Console.WriteLine(neuralNetwok.Prediction(new double[] { 1,1,1,1 })[0]);

//double lr = 0.2;
//Random rnd = new();
//double cost = 10;
//int iter = 0;
//while (cost > 0.0001)
//{

//    iter++;
//    if (iter % 1000 == 0)
//    {
//        Console.WriteLine(cost + " " + iter);
//    }
//    /*
//    DataPoint[] selection = new DataPoint[100];
//    for (int j = 0; j < selection.Length; j++)
//    {
//        selection[j] = dataPoints[rnd.Next(dataPoints.Length)];
//    }
//    */
//    cost = neuralNetwok.Learn(dataPoints, lr);
//}


//Console.WriteLine();
//prediction = neuralNetwok.Prediction(new double[] { 0, 0, 0, 0 });
//for (int i = 0; i < prediction.Length; i++)
//{
//    Console.WriteLine(i + " " + prediction[i]);
//}

//Console.WriteLine();

//prediction = neuralNetwok.Prediction(new double[] { 0, 0, 1, 0 });
//for (int i = 0; i < prediction.Length; i++)
//{
//    Console.WriteLine(i + " " + prediction[i]);
//}
