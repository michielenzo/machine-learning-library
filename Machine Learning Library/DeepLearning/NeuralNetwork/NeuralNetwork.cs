using System;
using System.Collections.Generic;
using System.Linq;

namespace Machine_Learning_Library.DeepLearning.NeuralNetwork
{
    class NeuralNetwork {
        
        private readonly List<List<Neuron>> _layers = new List<List<Neuron>>();
        
        private readonly Random _random = new Random();

        public NeuralNetwork(int[] layerSpecification, string[] outPutLabels)
        {
            CreateNeurons(layerSpecification, outPutLabels);
            CreateAxons();
        }

        public void Train(double[] inputActivations, string label)
        {
            RunEpoch(inputActivations);
            double cost = CalculateCost(label);
        }

        private double CalculateCost(string label)
        {
            List<Neuron> outputLayer = _layers.Last();
            List<double> costs = new List<double>();
            
            foreach (Neuron neuron in outputLayer)
            {
                double wantedActivation = 0;
                if (((OutputNeuron) neuron).label == label) wantedActivation = 1;
                
                costs.Add(Math.Pow(neuron.Activation - wantedActivation, 2));
            }

            return costs.Sum();
        }


        public void RunEpoch(double[] inputActivations)
        {
            for (int i = 0; i < inputActivations.Length; i++)
            {
                _layers[0][i].Activation = inputActivations[i];
            }
            
            List<List<Neuron>> allLayersExceptTheFirst = _layers.GetRange(1, _layers.Count - 1);
            
            foreach (var neuron in allLayersExceptTheFirst.SelectMany(layer => layer))
            {
                neuron.Activation = CalculateActivation(neuron);
            }
        }

        private double CalculateActivation(Neuron neuron)
        {
            double weightedSum = neuron.Synapses.Sum(synapse => synapse.Weight * synapse.Owner.Activation);
            return ReLu(weightedSum + neuron.bias);
        }
        
        /*
         * ReLu is an activation function alternative for the sigmoid function. The sigmoid activation function is
         * representative of the working of biological neural networks.
         */
        private static double ReLu(double val)
        {
            return Math.Max(0, val);
        }

        private void CreateNeurons(int[] layerSpecification, string[] outPutLabels)
        {
            
            foreach (int nNeurons in layerSpecification[.. ^1])
            {
                List<Neuron> layer = new List<Neuron>();
                
                for (int j = 0; j < nNeurons; j++) layer.Add(new Neuron( -_random.NextDouble()));
                
                _layers.Add(layer);
            }
            
            List<Neuron> outPutLayer = new List<Neuron>();
            for (int j = 0; j < layerSpecification[^1]; j++)
            {
                outPutLayer.Add(new OutputNeuron(_random.NextDouble(), outPutLabels[j]));
            }
            
            _layers.Add(outPutLayer);
        }

        private void CreateAxons()
        {
            // For every layer except the last one.
            for(int i = 0; i < _layers.Count - 1; i++)
            {
                List <Neuron> layer = _layers[i];
                
                // For each neuron in layer i
                foreach (var neuron in layer)
                {
                    // For each neuron in the next layer
                    for (int k = 0; k < _layers[i + 1].Count; k++)
                    {
                        List<Neuron> nextLayer = _layers[i + 1];
                        Neuron neuronInNextLayer = nextLayer[k];
                        
                        Axon axon = new Axon {
                            Owner = neuron, 
                            Receiver = neuronInNextLayer, 
                            Weight = (float) _random.NextDouble()
                        };

                        neuron.Axons.Add(axon);
                        neuronInNextLayer.Synapses.Add(axon);
                    }
                }
            }
        }
        
    }
}
