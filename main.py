from pipeline import AutoMLPipeline
from config import Config
import sys

def main():
    print("Multi-Agent AutoML Team")
    print("Three AI Agents Working Together")
    
    try:
        Config.validate()
        print("Azure OpenAI configuration validated\n")
    except ValueError as e:
        print(f"Configuration error: {e}")
        return
    
    if len(sys.argv) > 2:
        input_csv = sys.argv[1]
        target_column = sys.argv[2]
    else:
        input_csv = "outputs/sample_data.csv"
        target_column = "target"
        print(f"Using sample data: {input_csv}")
        print(f"Target column: {target_column}\n")
    
    pipeline = AutoMLPipeline()
    
    try:
        print("Starting the Multi-Agent Pipeline...\n")
        final_metrics = pipeline.run(input_csv, target_column)
        
        print("PIPELINE COMPLETE!")
        print(f"\nFinal Model Metrics:")
        for metric, value in final_metrics.items():
            print(f"  - {metric}: {value:.4f}")
        
        print("\nGenerated Files:")
        print("  outputs/clean_data.csv - Cleaned dataset")
        print("  outputs/engineered_data.csv - With new features")
        print("  outputs/final_model_code.py - Best model code")
        print("  outputs/final_report.md - Complete execution report")
        print("  outputs/agent_execution.log - Detailed logs")
        
    except Exception as e:
        print(f"\nPipeline error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("\nUsage:")
    print("  python main.py                          # use sample data")
    print("  python main.py data.csv target_column   # use your own data")
    print()
    main()