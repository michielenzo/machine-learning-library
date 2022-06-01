using System;
using System.Collections.Generic;
using System.Text;

namespace Machine_Learning_Library.DeepLearning.NeuralNetwork
{
    public class Neuron
    {
        public double bias;

        public double Activation = 0;

        public readonly List<Axon> Axons = new List<Axon>();

        public readonly List<Axon> Synapses = new List<Axon>();

        public Neuron(double bias)
        {
            this.bias = bias;
        }
        
    }
}
