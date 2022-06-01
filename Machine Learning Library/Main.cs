using System;
using Machine_Learning_Library.DeepLearning.NeuralNetwork;

namespace Machine_Learning_Library
{
    public class Program
    {
        public static void Main()
        {

            /*Random random = new Random();

            Console.WriteLine((random.NextDouble() * 100000) % 5);*/

            /*for (int i = 10 - 1; i >= 0; i--)
            {
                Console.WriteLine(i);
            }*/

            int[] arr = {3,3,3,3,3};
            string[] outputLabels = {"a","b","c"};
            NeuralNetwork neuralNetwork = new NeuralNetwork(arr, outputLabels);
            
            double[] inputActivations = { 4, 0.1, 3 };
            neuralNetwork.RunEpoch(inputActivations);
            
            Console.WriteLine(Sigmoid(2));
        }
        
        public static float Sigmoid(double value) {
            return 1.0f / (1.0f + (float) Math.Exp(-value));
        }
    }
    
    
}