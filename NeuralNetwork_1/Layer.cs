using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork_1
{
    internal class Layer
    {
        private readonly int amountInputNodes;
        private readonly int amountOutputNodes;
        private double[,] inputWeights; // [inputNode, outputNode]
        private double[] biases; // [outputNode]

        public Layer(int _amountInputNodes, int _amountOutputNodes, bool randomize = true)
        {
            this.amountInputNodes = _amountInputNodes;
            this.amountOutputNodes = _amountOutputNodes;

            inputWeights = new double[amountInputNodes, amountOutputNodes];
            biases = new double[amountOutputNodes];
            if (randomize)
            {
                Random random = new();
                for (int i = 0; i < inputWeights.GetLength(0); i++)
                {
                    for (int j = 0; j < inputWeights.GetLength(1); j++)
                    {
                        inputWeights[i, j] = random.NextDouble() * 2 - 1;
                    }
                }
                for (int i = 0; i < biases.Length; i++)
                {
                    biases[i] = random.NextDouble() * 2 - 1;
                }
            }
        }

        public double[] GetOutput(double[] inputs)
        {
            double[] output = new double[amountOutputNodes];
            for (int i = 0; i < amountOutputNodes; i++)
            {
                for (int j = 0; j < amountInputNodes; j++)
                {
                    output[i] += inputs[j] * inputWeights[j, i];
                }
                output[i] += biases[i];
                output[i] = ActivationFuncion(output[i]);
            }
            return output;
        }

        private static double ActivationFuncion(double x)
        {
            return 1/(1 + Math.Exp(-x));
        }
    }
}
