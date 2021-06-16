using System;
using Machine_Learning_Library.DeepLearning.NeuralNetwork;

namespace Machine_Learning_Library
{
    public class Program
    {
        public static void Main()
        {
            Random random = new Random();
            Console.WriteLine(random.NextDouble());
            Console.WriteLine(random.NextDouble());
            Console.WriteLine(random.NextDouble());
            Console.WriteLine(random.NextDouble());
            Console.WriteLine(random.NextDouble());
            Console.WriteLine(random.NextDouble());
            /*int[] arr = {3,3,3};
            NeuralNetwork neuralNetwork = new NeuralNetwork(arr);
            neuralNetwork.RunEpoch();*/
        }
        
        public static float Sigmoid(double value) {
            return 1.0f / (1.0f + (float) Math.Exp(-value));
        }
    }
    
    
}