using System;
using System.Collections.Generic;
using System.Text;

namespace Machine_Learning_Library.DeepLearning.NeuralNetwork
{
    public class Neuron
    {
        public float bias = 0.5f;

        public float activation = 0;

        /*
        public List<Axon> axons;

        public List<Axon> synapses;
        */

        public readonly Dictionary<Neuron, float> axis = new Dictionary<Neuron, float>();

        public readonly Dictionary<Neuron, float> Synapses = new Dictionary<Neuron, float>();

    }
}
