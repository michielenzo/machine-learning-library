namespace Machine_Learning_Library.DeepLearning.NeuralNetwork
{
    public class OutputNeuron: Neuron
    {
        public string label;
        
        public OutputNeuron(double bias, string label) : base(bias)
        {
            this.label = label;
        }
    }
}