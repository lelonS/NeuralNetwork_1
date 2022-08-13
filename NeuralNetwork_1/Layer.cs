using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork_1
{
    internal class Layer
    {
        private int amountInputNodes;
        private int amountOutputNodes;
        private double[,] inputWeights; // [inputNode, outputNode]
        private double[] biases; // [outputNode]

        public Layer(int _amountInputNodes, int _amountOutputNodes)
        {
            this.amountInputNodes = _amountInputNodes;
            this.amountOutputNodes = _amountOutputNodes;

            inputWeights = new double[amountInputNodes, amountOutputNodes];
            biases = new double[amountOutputNodes];
        }
    }
}
