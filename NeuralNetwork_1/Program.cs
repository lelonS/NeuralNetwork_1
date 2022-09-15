// See https://aka.ms/new-console-template for more information


using NeuralNetwork_1;


int[] order = new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

DataPoint[] dataPoints = new DataPoint[10*10];


// Generate DataPoints
string[] orderStr = new string[order.Length];
for (int i = 0; i < order.Length; i++)
{
    string str = Convert.ToString(order[i], 2);
    str = str.PadLeft(4, '0');
    orderStr[i] = str;
    Console.WriteLine(i + " " + str);
}

for (int k = 0; k < order.Length; k++)
{
    for (int i = 0; i < orderStr.Length; i++)
    {
        double[] input = new double[4];
        for (int j = 0; j < 4; j++)
        {
            input[j] = double.Parse(orderStr[i][j].ToString());
        }
        double[] expectedOutput = new double[10];
        expectedOutput[i] = 1;
        DataPoint newDataPoint = new(input, expectedOutput);
        Console.WriteLine(newDataPoint.ToString());
        dataPoints[i+k*10] = newDataPoint;
    }
}


int[] layers = new int[] { 4, 4, 4, 10};
NeuralNetwork neuralNetwok = new(layers);
double[] prediction = neuralNetwok.Prediction(new double[] { 0, 0, 0, 0 });
for (int i = 0; i < prediction.Length; i++)
{
    Console.WriteLine(i + " " + prediction[i]);
}
// Console.WriteLine(neuralNetwok.Prediction(new double[] { 1,1,1,1 })[0]);

double lr = 0.2;
Random rnd = new();
for (int i = 0; i < 10000; i++)
{
    DataPoint[] selection = new DataPoint[20];
    for (int j = 0; j < selection.Length; j++)
    {
        selection[j] = dataPoints[rnd.Next(dataPoints.Length)];
    }
    neuralNetwok.Learn(selection, lr);
}


Console.WriteLine();
prediction = neuralNetwok.Prediction(new double[] { 0, 0, 0, 0 });
for (int i = 0; i < prediction.Length; i++)
{
    Console.WriteLine(i + " " + prediction[i]);
}

Console.WriteLine();

prediction = neuralNetwok.Prediction(new double[] { 0, 0, 1, 0 });
for (int i = 0; i < prediction.Length; i++)
{
    Console.WriteLine(i + " " + prediction[i]);
}
