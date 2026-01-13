def main():
    """
    Unified entry point for the manuscript workflow:

    OMNI -> Feature construction -> LightGBM -> Residual -> Transformer -> Evaluation
    Baselines: RF / CNN / LSTM (same tabular feature set)
    """
    print("OMNI → Feature → LightGBM → Residual → Transformer → Evaluation")
    print("Baselines: RF / CNN / LSTM (tabular)")

if __name__ == "__main__":
    main()
