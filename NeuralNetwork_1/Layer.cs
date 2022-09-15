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
        public double[,] inputWeights; // [inputNode, outputNode]
        private double[] biases; // [outputNode]

        private double[] inputs;
        private double[] weightedInputs;
        private double[] outputs;

        private double[] nodeValues;
        private double[,] weightGradients;
        private double[] biasGradients;

        public Layer(int _amountInputNodes, int _amountOutputNodes, bool randomize = true)
        {
            this.amountInputNodes = _amountInputNodes;
            this.amountOutputNodes = _amountOutputNodes;
            this.weightedInputs = new double[amountOutputNodes];
            this.ClearGradients();

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
            this.inputs = inputs;
            double[] output = new double[amountOutputNodes];
            for (int i = 0; i < amountOutputNodes; i++)
            {
                for (int j = 0; j < amountInputNodes; j++)
                {
                    output[i] += inputs[j] * inputWeights[j, i];
                }
                output[i] += biases[i];
                weightedInputs[i] = output[i];
                output[i] = ActivationFuncion(output[i]);
            }
            this.outputs = output;
            return output;
        }

        private static double ActivationFuncion(double x)
        {
            return 1/(1 + Math.Exp(-x));
        }

        private static double ActivationDerivative(double x)
        {
            double activation = ActivationFuncion(x);
            return activation * (1 - activation);
        }

        private static double NodeCostDerivative(double ouput, double expectedOutput)
        {
            return 2 * (ouput - expectedOutput);
        }

        public double[] CalcOuputNodeValues(double[] expectedOutput)
        { 
            this.nodeValues = new double[expectedOutput.Length];
            for (int i = 0; i < this.nodeValues.Length; i++)
            {
                double costDeriv = NodeCostDerivative(this.outputs[i], expectedOutput[i]);
                double activationDeriv = ActivationDerivative(weightedInputs[i]);
                this.nodeValues[i] = costDeriv * activationDeriv;
            }
            return this.nodeValues;
        }

        public double[] CalcHiddenNodeValues(Layer nextLayer)
        {
            this.nodeValues = new double[amountOutputNodes];
            for (int nodeIndex = 0; nodeIndex < this.nodeValues.Length; nodeIndex++)
            {
                double nodeValue = 0;
                for (int nextNodeIndex = 0; nextNodeIndex < nextLayer.nodeValues.Length; nextNodeIndex++)
                {
                    double weightedInputDeriv = nextLayer.inputWeights[nodeIndex, nextNodeIndex];
                    nodeValue += weightedInputDeriv * nextLayer.nodeValues[nextNodeIndex];
                }
                nodeValue *= ActivationDerivative(weightedInputs[nodeIndex]);
                this.nodeValues[nodeIndex] = nodeValue;
            }
            return this.nodeValues;
        }

        public void UpdateGradients()
        {
            for (int i = 0; i < amountOutputNodes; i++)
            {
                for (int j = 0; j < amountInputNodes; j++)
                {
                    double wDeriv = this.inputs[j] * this.nodeValues[i];
                    weightGradients[j, i] += wDeriv;
                }
                double bDeriv = 1* this.nodeValues[i];
                biasGradients[i] += bDeriv;
            }
        }

        public void ApplyGradients(double learnRate)
        {
            for (int i = 0; i < amountOutputNodes; i++)
            {
                this.biases[i] -= this.biasGradients[i] * learnRate;
                for (int j = 0; j < amountInputNodes; j++)
                {
                    this.inputWeights[j, i] -= weightGradients[j, i] * learnRate;
                }
            }
        }

        public void ClearGradients()
        {
            this.nodeValues = new double[amountOutputNodes];
            this.biasGradients = new double[amountOutputNodes];
            this.weightGradients = new double[amountInputNodes, amountOutputNodes];
        }
    }
}
