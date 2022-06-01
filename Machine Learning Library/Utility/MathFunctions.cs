using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using NUnit.Framework.Internal;

namespace Machine_Learning_Library
{
    public class MathFunctions
    {
        
        /// <summary>
        /// This function calculates the slope of a point in a mathematical function using another point very close to
        /// the original.
        /// It is calculated by the following formula: slope = (y2 - y1) / (x2 - x1) = deltaY / deltaX
        /// </summary>
        /// <param name="func"></param>
        /// <param name="x1"></param>
        /// <returns></returns>
        public static double GetSlope(Func<double, double> func, double x1)
        {
            double x2 = x1 + 0.001;
            double y1 = func(x1);
            double y2 = func(x2);
            return (y2 - y1) / (x2 - x1);
        }

        public double Sigmoid(double value) {
            return 1.0 / (1.0 + Math.Exp(-value));
        }
        
    }
}