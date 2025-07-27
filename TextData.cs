using Microsoft.ML.Data;

public class TextData
{
    [LoadColumn(0)]  // Maps the first column to the Text property
    public string Text { get; set; }

    [LoadColumn(1)]  // Maps the second column to the Label property
    public bool Label { get; set; }  // Change Label to Boolean for binary classification
}
