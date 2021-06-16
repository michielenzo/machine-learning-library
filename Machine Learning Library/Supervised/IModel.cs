namespace Machine_Learning_Library
{
    public interface IModel
    {
        void Fit(float independent, float dependent);
        
        double Predict(float dependent);
    }
}    