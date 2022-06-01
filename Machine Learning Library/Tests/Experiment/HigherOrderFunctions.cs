using System;
using NUnit.Framework;

namespace Machine_Learning_Library.experiment
{
    [TestFixture]
    public class HigherOrderFunctions
    {

        [Test]
        public void HighActionFunctionExperiment1()
        {
            Action<int, int> addFunction = (num1, num2) => { Console.WriteLine(num1 + num2); };
            MultiplyB4Add(addFunction, 5);
        }
        
        [Test]
        public void HighFuncFunctionExperiment()
        {
            Func<double, double, double> addfunction = (num1, num2) => { return num1 + num2; };
            
            Console.WriteLine(MultiplyB4Add(addfunction, 3));
        }

        private void MultiplyB4Add(Action<int, int> addFunction, int factor)
        {
            addFunction(4 * factor,5 * factor);
        }
        
        private double MultiplyB4Add(Func<double, double, double> addFunction, int factor)
        {
            return addFunction(4 * factor,5 * factor);
        }
    }
}