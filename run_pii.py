import os
from pii import (
    DatasetProcessor, 
    DatasetAnalyzer,
    DatasetVisualizer,
    TorchDataset,
    TorchInitializer,
    TorchCollator,
    TorchEvaluator,
    PIIDetector,
)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PII Detection and Analysis")
    parser.add_argument('--data_in', type=str, default='build/pii/echr', help='Directory containing dataset JSON files')
    parser.add_argument('--data_out', type=str, default='build/pii/outputs', help='Output directory for processed data')
    parser.add_argument('--analyze', action='store_true', help='Run dataset analysis')
    parser.add_argument('--model_train', action='store_true', help='Train the model')
    parser.add_argument('--model_eval', action='store_true', help='Evaluate the model')
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased', help='Pre-trained model name')
    parser.add_argument('--model_epochs', type=int, default=50, help='Number of training epochs')
    
    args = parser.parse_args()
    processor = DatasetProcessor(args.data_in)
    datasets, labels = processor.process()
    
    os.makedirs(args.data_out, exist_ok=True)

    if args.analyze:
        report_path = os.path.join(args.data_out, 'tab_analysis.md')
        entity_path = os.path.join(args.data_out, 'per_entity.png')
        identifier_path = os.path.join(args.data_out, 'per_identifier.png')
        confidentiality_path = os.path.join(args.data_out, 'per_confidentiality.png')

        output = ""
        analyzer = DatasetAnalyzer(datasets)
        output += analyzer.examine_datasets()
        output += analyzer.examine_entity_mentions()
        output += analyzer.print_split_statistics()

        visualizer = DatasetVisualizer(datasets)
        output += visualizer.plot_entity_type_distribution(path=entity_path)
        output += visualizer.plot_identifier_type_distribution(path=identifier_path)
        output += visualizer.plot_confidential_status_distribution(path=confidentiality_path)

        analysis_result = analyzer.run_analysis()
        output += analysis_result.examine_document_level()
        output += analysis_result.examine_entity_level()
        output += analysis_result.examine_meta_level()
        output += analysis_result.print_final_summary()

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(output)

    if args.model_train or args.model_eval:
        with TorchInitializer(args.model_name, labels) as (model, tokenizer):
            train_dataset = TorchDataset(datasets['train'], tokenizer, labels)
            train_dataset.sample_one(tokenizer)
            
            val_dataset = TorchDataset(datasets['dev'], tokenizer, labels)
            test_dataset = TorchDataset(datasets['test'], tokenizer, labels)

            collator = TorchCollator(tokenizer)
            evaluator = TorchEvaluator(labels)
            evaluator.sample_one()

            trainer = PIIDetector(args.data_out, model, tokenizer, labels)
            trainer.sample_one()

            if args.model_train:
                trainer.train(train_dataset, val_dataset, collator, evaluator, epochs=args.model_epochs)
            if args.model_eval:
                trainer.evaluate(test_dataset)