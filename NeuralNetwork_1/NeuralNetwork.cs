using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork_1
{
    internal class NeuralNetwork
    {
        private readonly Layer[] layers;
        public int amountLayers;
        public int amountInputs;
        public int amountOutputs;
        public NeuralNetwork(int[] layers)
        {
            // Layer 0, input layer is simply the input, no layer representation
            this.layers = new Layer[layers.Length - 1];
            this.amountLayers = layers.Length;
            this.amountInputs = layers[0];
            this.amountOutputs = layers[^1];

            for (int i = 0; i < layers.Length - 1; i++)
            {
                this.layers[i] = new Layer(layers[i], layers[i + 1]);
            }
        }

        public double[] Prediction(double[] input)
        {
            double[] layer_output = input;
            for (int i = 0; i < this.layers.Length; i++)
            {
                layer_output = this.layers[i].GetOutput(layer_output);
            }
            return layer_output;
        }


        public void UpdateAllGradients(DataPoint dataPoint)
        {
            // Run input through network
            Prediction(dataPoint.inputs);

            // Update output layer
            Layer outputLayer = this.layers[^1];
            double[] _ = outputLayer.CalcOuputNodeValues(dataPoint.expectedOutputs);
            outputLayer.UpdateGradients();

            for (int i = this.layers.Length - 2; i >= 0; i--)
            {
                Layer hiddenLayer = this.layers[i];
                _ = hiddenLayer.CalcHiddenNodeValues(this.layers[i + 1]);
                hiddenLayer.UpdateGradients();
            }
        }

        public void ApplyAllGradients(double learnRate)
        {
            for (int i = 0; i < this.layers.Length; i++)
            {
                this.layers[i].ApplyGradients(learnRate);
            }
        }
        public void ClearAllGradients()
        {
            for (int i = 0; i < this.layers.Length; i++)
            {
                this.layers[i].ClearGradients();
            }
        }
        public void Learn(DataPoint[] dataPoints, double learnRate)
        {
            for (int i = 0; i < dataPoints.Length; i++)
            {
                UpdateAllGradients(dataPoints[i]);
            }
            ApplyAllGradients(learnRate / dataPoints.Length);
            ClearAllGradients();
        }
    }
}
