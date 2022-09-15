using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork_1
{
    internal class DataPoint
    {
        public double[] inputs;
        public double[] expectedOutputs;

        public DataPoint(double[] inputs, double[] expectedOutputs)
        {
            this.inputs = inputs;
            this.expectedOutputs = expectedOutputs;
        }

        public override string ToString()
        {
            string inputStr = "";
            foreach (double input in this.inputs)
            {
                inputStr += input.ToString() + " ";
            }
            string outputStr = "";
            foreach (double output in this.expectedOutputs)
            {
                outputStr += output.ToString() + " ";
            }
            return inputStr + " -> " + outputStr;
        }
    }
}

