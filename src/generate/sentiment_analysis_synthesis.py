from inference.batch_inference import SentimentAnalysisSynthesisPipeline
import argparse
from pathlib import Path
from src.generate.utils import get_model_name


def main(args):
    pipeline = SentimentAnalysisSynthesisPipeline(
        model_path=args.model_path,
        prompt_template_path=args.prompt_template_path,
        num_shots=args.num_shots,
        num_tickers=args.num_tickers,
        num_gpus=args.num_gpus,
        batch_size=args.batch_size,
        max_context_window=args.max_context_window,
        max_generate_tokens=args.max_generate_tokens,
        max_num_seqs=args.max_num_seqs,
        verbose=args.verbose
    )
    results = pipeline.generate(
        num_examples=args.num_examples
    )

    model = get_model_name(args.model_path)
    output_path = Path(args.output_folder) / (model + "_" + str(args.num_examples) + 
                                              "_prompt=" + Path(args.prompt_template_path).stem + ".json")
    metadata_path = Path(args.output_folder) / Path("metadata") / (model + "_" + str(args.num_examples) + 
                                                "_prompt=" + Path(args.prompt_template_path).stem + ".json")

    if not metadata_path.parent.exists():
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    pipeline.save_results(
        examples=results,
        output_path=output_path,
        metadata_path=metadata_path,
        metadata={**args.__dict__}
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data generation args
    parser.add_argument("--num_examples", type=int, required=True, help="Number of examples to generate")
    parser.add_argument("--num_shots", type=int, default=3, help="Number of shots to use")
    parser.add_argument("--num_tickers", type=int, default=1, help="Number of tickers to use")

    # model args
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--prompt_template_path", type=str, required=True, help="Path to the prompt template")
    parser.add_argument("--output_folder", type=str, default="./data/sentiment_analysis/data_manual_prompt/synthetic_data_full", help="Path to save the results")

    # inference args
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
    parser.add_argument("--max_context_window", type=int, default=1024, help="Max context window")
    parser.add_argument("--max_generate_tokens", type=int, default=128, help="Max generate tokens")
    parser.add_argument("--max_num_seqs", type=int, default=30, help="Max number of sequences")
    parser.add_argument("--verbose", action="store_true", help="Whether to print verbose output")
    
    args = parser.parse_args()
    main(args)
