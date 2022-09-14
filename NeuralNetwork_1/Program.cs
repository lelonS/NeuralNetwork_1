// See https://aka.ms/new-console-template for more information


using NeuralNetwork_1;

int[] layers = new int[] { 1, 2};
NeuralNetwork neuralNetwok = new(layers);
Console.WriteLine(neuralNetwok.Prediction(new double[] { 1 })[0]);
