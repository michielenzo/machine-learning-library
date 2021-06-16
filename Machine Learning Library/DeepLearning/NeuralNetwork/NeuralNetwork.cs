using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Text;

namespace Machine_Learning_Library.DeepLearning.NeuralNetwork
{
    class NeuralNetwork {
        
        private readonly List<List<Neuron>> _layers = new List<List<Neuron>>();
        
        private Random _random = new Random();

        public NeuralNetwork(int[] layerSpecification) 
        {
            CreateNeurons(layerSpecification);
            CreateAxons();
        }

        public void RunEpoch()
        {
            foreach (List<Neuron> layer in _layers)
            {
                foreach (Neuron neuron in layer)
                {
                    neuron.activation = CalculateActivation(neuron);
                }                
            }
        }

        private float CalculateActivation(Neuron neuron)
        {
            float weightedSum = neuron.Synapses.Sum(synapse => synapse.Key.activation * synapse.Value);
            return ReLU(weightedSum + neuron.bias);
        }
        
        private static float ReLU(double val)
        {
            return (float) Math.Max(0, val);
        }

        private void Create(int[] layerSpecification)
        {
            
        }

        private void CreateNeurons(int[] layerSpecification)
        {
            foreach (int nNeurons in layerSpecification)
            {
                List<Neuron> layer = new List<Neuron>();

                for (int j = 0; j < nNeurons; j++)
                {
                    layer.Add(new Neuron());
                }

                _layers.Add(layer);
            }
        }

        private void CreateAxons()
        {
            for (int i = 0; i < _layers.Count; i++) 
            {
                foreach (Neuron neuron in _layers[i])
                {
                    if (i < _layers.Count - 1)
                    {
                        foreach (Neuron neuronInNextLayer in _layers[i+1]) 
                        {
                            neuron.axis.Add(neuronInNextLayer , 0);
                        }                        
                    }

                    if (i < 1) continue;
                    
                    foreach (Neuron neuronInPrevioustLayer in _layers[i-1]) 
                    {
                        neuron.Synapses.Add(neuronInPrevioustLayer , 0);
                    }
                }
            }
        }
    }
}
