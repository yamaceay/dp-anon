#region ###################################### Imports ######################################
import sys
import os
import json
import gc
import re
import ntpath
import time
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO) # Configure logging

from tqdm.autonotebook import tqdm
import numpy as np
import pandas as pd
import en_core_web_lg # This model is leveraged for every spaCy usage (https://spacy.io/models/en#en_core_web_lg)

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from transformers import pipeline, Pipeline
import datasets
import shap
import concurrent


#endregion

#region ###################################### Configuration file argument ######################################

#region ################### Arguments parsing ###################
def argument_parsing():
    if (args_count := len(sys.argv)) > 2:
        raise Exception(f"One argument expected, got {args_count - 1}")
    elif args_count < 2:
        raise Exception("You must specify the JSON configuration filepath as first argument")

    target_dir = sys.argv[1]
    return target_dir

#endregion

#region ################### JSON file loading ###################
def get_config_from_file(target_dir):
    if not target_dir.endswith(".json"):
        raise Exception(f"The configuration file {target_dir} needs to have json format (end with .json)")
    elif not os.path.isfile(target_dir):
        raise Exception(f"The JSON configuration file {target_dir} doesn't exist")

    with open(target_dir, "r") as f:
        config = json.load(f)
    return config
#endregion

#endregion

#region ###################################### PETRE class ######################################

class PETRE():
    #region ################### Properties ###################

    #region ########## Mandatory configs ##########

    mandatory_configs_names = ["output_base_folder_path", "data_file_path",
                            "individual_name_column", "original_text_column",
                            "starting_anonymization_path", "tri_pipeline_path", "ks"]
    output_base_folder_path = None
    data_file_path = None
    individual_name_column = None
    original_text_column = None
    starting_anonymization_path = None
    tri_pipeline_path = None
    ks = None

    #endregion

    #region ########## Optional configs with default values ##########

    optional_configs_names = ["mask_text", "use_mask_all_instances", "explainability_mode", "use_chunking"]
    mask_text:str = ""
    use_mask_all_instances:bool=True
    explainability_mode = "SHAP" # Options: "SHAP" orf "Greedy"
    use_chunking:bool=True

    #endregion

    #region ########## Derived configs ##########

    starting_annon_name:str = None
    output_folder_path:str = None

    #endregion

    #region ########## Functional properties ##########
    
    data_df:pd.DataFrame = None    
    label_to_name:dict = None
    name_to_label:dict = None
    annotated_individuals:set = None
    non_annotated_individuals:set = None

    tri_pipeline:Pipeline = None
    dataset:Dataset = None
    shap_explainer = None
    explainability_method = None
    device = None

    #endregion

    #endregion


    #region ################### Constructor and configurations ###################

    def __init__(self, **kwargs):
        self.set_configs(**kwargs, are_mandatory_configs_required=True)

    def set_configs(self, are_mandatory_configs_required=False, **kwargs):
        arguments = kwargs.copy()

        # Mandatory configs
        for setting_name in self.mandatory_configs_names:
            value = arguments.get(setting_name, None)
            if isinstance(value, str) or isinstance(value, list):
                self.__dict__[setting_name] = arguments[setting_name]
                del arguments[setting_name]
            elif are_mandatory_configs_required:
                raise AttributeError(f"Mandatory argument {setting_name} is not defined or it is not a string or list")
        
        # Check list of ks
        if (not isinstance(self.ks, list)) or len(self.ks)==0 or len([k for k in self.ks if isinstance(k, int)])<len(self.ks):
           raise AttributeError(f"Setting \"ks\" must be a list of integers", isinstance(self.ks, list), len(self.ks), len([k for k in self.ks if isinstance(k, int)]))
        # Sort ks in ascending order
        else:
           self.ks.sort()
        
        # Store remaining optional configs
        for (opt_setting_name, opt_setting_value) in arguments.items():
            if opt_setting_name in self.optional_configs_names:                
                if isinstance(opt_setting_value, str) or isinstance(opt_setting_value, int) or \
                isinstance(opt_setting_value, float) or isinstance(opt_setting_value, bool):
                    self.__dict__[opt_setting_name] = opt_setting_value
                else:
                    raise AttributeError(f"Optional argument {opt_setting_name} is not a string, integer, float or boolean.")
            else:
                logging.warning(f"Unrecognized setting name {opt_setting_name}")

        # Generate derived configs
        head, tail = ntpath.split(self.starting_anonymization_path)
        filename = tail or ntpath.basename(head)
        self.starting_annon_name = os.path.splitext(filename)[0]
        self.output_folder_path =  os.path.join(self.output_base_folder_path, self.starting_annon_name)

        # Check for GPU with CUDA
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        else:
            self.device = torch.device("cpu")

    #endregion


    #region ################### Run all blocks ###################

    def run(self, verbose=True):
        self.initialization(verbose=verbose)
        self.incremental_execution(verbose=verbose)

    #endregion


    #region ################### Initialization ###################

    def initialization(self, verbose=True):
        if verbose: logging.info("######### START: INITIALIZATION #########")

        if verbose: logging.info("#### START: LOADING DATA ####")

        if os.path.exists(self.data_file_path):
            if self.data_file_path.endswith(".json"): # JSON
                complete_df = pd.read_json(self.data_file_path)
            else: # Or CSV
               complete_df = pd.read_csv(self.data_file_path)
            self.data_df = complete_df[[self.individual_name_column, self.original_text_column]]

            self.names = sorted([name for name in self.data_df[self.individual_name_column]])
            self.name_to_label = {name:idx for idx, name in enumerate(self.names)}
            self.label_to_name = {label:name for name, label in self.name_to_label.items()}
            self.num_labels = len(self.label_to_name)
        else:
            raise Exception(f"Data file at {self.data_file_path} not found")

        if verbose: logging.info("#### END: LOADING DATA ####")

        if verbose: logging.info("#### START: LOADING TRI PIPELINE AND EXPLAINER ####")
        
        # Load re-identification pipeline
        
        self.tri_pipeline = pipeline("text-classification", model=self.tri_pipeline_path,
                                    tokenizer=self.tri_pipeline_path,
                                    device=self.device,
                                    top_k=self.num_labels,
                                    max_length=512,
                                    truncation=True)

        # Define explainability_method
        if self.explainability_mode == "SHAP":
            self.shap_explainer = shap.Explainer(self.tri_pipeline, silent=True)
            self.explainability_method = self.shap_explainability
        elif self.explainability_mode == "Greedy":
            self.explainability_method = self.greedy_explainability
        else:
            raise Exception(f"Unrecognized explainability mode \"{self.explainability_mode}\". Select \"SHAP\" or \"Greddy\"")

        if verbose: logging.info("#### END: LOADING TRI PIPELINE AND EXPLAINER ####")

        if verbose: logging.info("#### START: CREATING DATASET WITH STARTING ANONYMIZATION ####")

        self.dataset = PETREDataset(self.data_df, self.tri_pipeline.tokenizer, self.name_to_label, self.mask_text, use_chunking=self.use_chunking)
        
        # Add starting anonymization
        if os.path.exists(self.starting_anonymization_path):
            with open(self.starting_anonymization_path, "r", encoding="utf-8") as f:
                starting_annotations = json.load(f)
            self.dataset.add_annotations(starting_annotations)
        else:
            raise Exception(f"Starting anonymization file at {self.starting_anonymization_path} not found")
        
        if verbose: logging.info("#### END: CREATING DATASET WITH STARTING ANONYMIZATION ####")

        if verbose: logging.info("#### START: COMPARING DATA WITH ANONYMIZATION ####")

        individuals = set(self.names)
        if verbose: logging.info(f"There are {len(individuals)} individuals to protect")
        annotation_names = set(starting_annotations.keys())
        if verbose: logging.info(f"There are {len(annotation_names)} annotations")
        self.annotated_individuals = individuals.intersection(annotation_names)
        self.non_annotated_individuals = individuals - self.annotated_individuals
        if len(self.non_annotated_individuals) == 0:
           if verbose: logging.info(f"All individuals have annotations")
        else:
           if verbose: logging.warning(f"There are {len(self.non_annotated_individuals)} individuals without annotations: {self.non_annotated_individuals}")

        if verbose: logging.info("#### END: COMPARING DATA WITH ANONYMIZATION ####")

        if verbose: logging.info("#### START: CREATING OUTPUT FOLDER ####")
        
        if not os.path.isdir(self.output_folder_path):
            os.makedirs(self.output_folder_path, exist_ok=True)

        if verbose: logging.info("#### END: CREATING OUTPUT FOLDER ####")
        

        if verbose: logging.info("######### END: INITIALIZATION #########")

    #endregion


    #region ################### Incremental execution ###################

    def incremental_execution(self, verbose:bool=True):
        if verbose: logging.info("######### START: EXECUTION #########")

        if verbose: logging.info("#### START: STARTING POINT ####")
        
        # Compute and save initial ranks
        accuracy, ranks, docs_probs = self.evaluate(max_rank=1)
        ranks_file_path = os.path.join(self.output_folder_path, f'ranks_k=0.csv')
        ranks.tofile(ranks_file_path, sep=",")
        logging.info(f"Initial rank==1 rate = {accuracy*100:.2f}%") # TODO: Remove?

        if verbose: logging.info("#### END: STARTING POINT ####")

        # Incrementing k
        for current_k in self.ks:
            if verbose: logging.info(F"#### START: PETRE WITH K={current_k} ####")
            
            self.petre(current_k, plot_explanations=False, verbose=True) # TODO: Check this verbose            

            if verbose: logging.info(F"#### END: PETRE WITH K={current_k} ####")

        if verbose: logging.info("######### END: EXECUTION #########")

    #region ########## Re-identification risk evaluation ##########

    def evaluate(self, max_rank:int=1, use_annotated:bool=True, batch_size:int=128):
        docs_probs = []
        n_correct_preds = 0
        n_individuals = len(self.dataset)
        
        # Generate all inputs        
        input_texts, doc_to_texts_idxs = self.dataset.get_all_texts(use_annotated)        

        # Gather results per document
        docs_probs, ranks = self.pipeline_results_to_docs_probs(input_texts,
                                                                doc_to_texts_idxs,
                                                                batch_size=batch_size)

        # Cumpute number of correct predictions and accuracy
        n_correct_preds = np.count_nonzero(ranks<=max_rank)
        accuracy = n_correct_preds/n_individuals        

        return accuracy, ranks, docs_probs 
    
    def pipeline_results_to_docs_probs(self, input_texts:list, doc_to_input_idxs:dict, batch_size:int=128):
        docs_probs = []
        ranks = []

        # Create dataset and input into pipeline
        inputs_dataset = datasets.Dataset.from_dict({"text":input_texts})["text"]
        results = self.tri_pipeline(inputs_dataset, batch_size=batch_size)

        # Gather results per document
        for document in tqdm(self.dataset, desc="Evaluating all documents"):
            label = document["label"]
            splits = document["splits"]
            splits_probs = torch.zeros((len(splits), len(self.names))) # Splits * Individuals
            doc_results = [results[idx] for idx in doc_to_input_idxs[label]]

            # Get probabilities from each split prediction
            for split_idx, split_preds in enumerate(doc_results):
                for pred in split_preds:
                    pred_label, pred_score = self.pipeline_pred_to_label_score(pred)
                    splits_probs[split_idx, pred_label] = pred_score
            
            # Store into docs_probs
            docs_probs.append(splits_probs)

            # Check rank position of aggregated probabilities
            rank, prob = self.get_doc_rank(splits_probs, label)
            ranks.append(rank)
        
        # Transform ranks into NumPy array
        ranks = np.array(ranks)        
        
        return docs_probs, ranks

    def evaluate_doc(self, document:dict, use_annotated:bool=True):
        complete_text = document["text"]
        label = document["label"]
        splits = document["splits"]
        splits_probs = torch.zeros((len(splits), len(self.names))) # Splits * Individuals

        # Evaluate each split
        for split_idx, split in enumerate(splits):
            splits_probs[split_idx, :] = self.evaluate_split(complete_text, split, use_annotated=use_annotated)

        # Check rank position of aggregated probabilities
        rank, prob = self.get_doc_rank(splits_probs, label)

        return splits_probs, rank

    def evaluate_split(self, complete_text:str, split:dict, use_annotated:bool=True):
        split_span = split["text_span"]
        split_text = complete_text[split_span[0]:split_span[1]]

        # Annotate text if required
        if use_annotated:
            split_text = self.dataset.annotate_text(split_text, split)

        # Get predictions
        split_probs = self.evaluate_text(split_text)

        return split_probs

    def evaluate_text(self, text:str):
        results = self.tri_pipeline([text])[0]
        probs = torch.empty((len(results)))

        # Get score (probability) for each individual
        for pred in results:
            pred_label, pred_score = self.pipeline_pred_to_label_score(pred)
            probs[pred_label] = pred_score
        
        return probs

    def pipeline_pred_to_label_score(self, pred:dict):
        label = int(pred["label"].split("_")[1])
        score = float(pred["score"])
        return label, score
        
    def get_doc_rank(self, splits_probs:np.array, label:int):
        # Aggregate probabilities by averaging
        aggregated_probs = splits_probs.sum(dim=0) / splits_probs.shape[0]
        prob = aggregated_probs[label] # Probability of the individual

        # Get rank position
        sorted_idxs = torch.argsort(aggregated_probs, descending=True)
        idx = torch.where(sorted_idxs == label)[0].item()
        rank = idx + 1 # +1 To start rank at 1        

        return rank, prob

    #endregion

    #region ########## Explainability ##########

    def shap_explainability(self, text:str, label:int, split:dict, plot_explanations:bool=False):
        terms_to_tokens = split["terms_to_tokens"]
        terms_weights = np.zeros(len(terms_to_tokens))
        masked_terms_idxs = split["masked_terms_idxs"]

        # Use SHAP to compute weight of each token
        tokens_weights = self.get_tokens_weights(text, label, plot_explanations=plot_explanations)

        # Aggregate tokens weights into terms weights
        for idx, term_tokens in enumerate(terms_to_tokens):
            # If term is masked, minus infinte
            if idx in masked_terms_idxs:
                term_weight = float("-inf")
            # Otherwise, compute term weight
            else:
                term_weight = 0
                for token_idx in term_tokens:
                    term_weight += tokens_weights[token_idx]

            # Set term weight
            terms_weights[idx] = term_weight

        return terms_weights

    def get_tokens_weights(self, text:str, label:int, plot_explanations:bool=False):
        shap_values = self.shap_explainer([text], batch_size=1)
        tokens_weights = shap_values.values[0, :, label]
        if plot_explanations:
            shap.plots.text(shap_values[0, :, label])

        return tokens_weights

    def greedy_explainability(self, text:str, label:int, split:dict, batch_size:int=128, plot_explanations:bool=False):
        terms_spans = split["terms_spans"]
        terms_weights = np.zeros(len(terms_spans))
        masked_terms_idxs = split["masked_terms_idxs"]

        # Get texts for current masking and masking of any remaining term
        input_texts = []
        annotated_text = self.dataset.annotate_text(text, split)
        input_texts.append(annotated_text)
        terms_idxs_to_assign = []
        for term_idx, (start, end) in enumerate(terms_spans):
            # If masked, weight of -Infinity
            if term_idx in masked_terms_idxs:
                terms_weights[term_idx] = float("-inf")
            # Otherwise, create a text version with it masked
            else:
                annotated_text = text[:start] + self.mask_text + text[end:]
                input_texts.append(annotated_text)
                terms_idxs_to_assign.append(term_idx)
        
        # Create dataset and input texts into pipeline
        inputs_dataset = datasets.Dataset.from_dict({"text":input_texts})["text"]
        results = self.tri_pipeline(inputs_dataset, batch_size=batch_size)

        # Get score (probability) for input
        probs = np.empty(len(results))
        for idx, res in enumerate(results):
            for pred in res:
                pred_label, pred_score = self.pipeline_pred_to_label_score(pred)
                if pred_label == label:
                    probs[idx] = pred_score
        
        # First probability corresponds to base probability to reduce
        base_prob = probs[0]

        # Ther rest of probabilities correspond to each masked term
        terms_weights[terms_idxs_to_assign] = base_prob - probs[1:] # The difference to the base_prob is their weight

        return terms_weights

    #endregion

    #region ########## Method ##########

    def petre(self, k:int, plot_explanations:bool=False, verbose:bool=True):
        annotated_terms = {} # TODO: Remove?
        total_n_steps = 0
        annotations_file_path = os.path.join(self.output_folder_path, f'petre_k={k}.json')

        # Load already existing annotations if they exist
        if os.path.exists(annotations_file_path):
            if verbose: logging.info(f"Loading already existing annotations for k={k}")
            with open(annotations_file_path, "r", encoding="utf-8") as f:
                annotations = json.load(f)
                self.dataset.add_annotations(annotations)
        
        # Compute individuals that keep requiring protection # TODO: Check if use this
        accuracy, ranks, docs_probs = self.evaluate(max_rank=k)
        n_individuals_to_protect = np.count_nonzero(ranks<k)
        if verbose: logging.info(f"Number of individuals requiring protection = {n_individuals_to_protect}")

        with tqdm(range(n_individuals_to_protect), total=n_individuals_to_protect) as pbar:
            # For each document in the dataset
            for idx, document in enumerate(self.dataset):
                # If document requires protection
                rank = ranks[idx]
                if rank < k:
                    label = document["label"]
                    name = self.label_to_name[label]
                    doc_processed = False
                    splits_probs = docs_probs[idx]
                    rank, prob = self.get_doc_rank(splits_probs, label)
                    
                    while not doc_processed:
                        message = f"Individual [{name}] obtained a rank of {rank} with a probability of {prob*100:.2f}%"
                        pbar.set_description(message)
                        if verbose: logging.info(message)

                        # While top position is not great enough and not in an end state
                        while rank < k:
                            # Mask the most disclosive term of the most disclosive split possible
                            most_disclosive_term, n_masked_terms, n_steps = self.mask_most_disclosive_term(document, splits_probs,
                                                                                                            annotated_terms,
                                                                                                            plot_explanations=plot_explanations)
                            total_n_steps += n_steps
                            
                            # If no term is masked, there are no more terms to mask in this document (avoid infinite loops)
                            if n_masked_terms == 0:                                
                                if verbose: logging.info("All meaningful terms already have been masked")
                                break # Exit lopp
                            # If at least one term has been masked
                            else:
                                rank, prob = self.get_doc_rank(splits_probs, label) # Recompute the rank
                                if verbose: logging.info(f"Term [{most_disclosive_term}] masked with {n_masked_terms} instance/s | Rank = {rank} | Prob = {prob*100:.2f}%")
                        
                        # Document has been processed
                        doc_processed = True
                        pbar.update()

                        # Store updated annotations
                        annotations = self.dataset.get_annotations()
                        with open(annotations_file_path, 'w') as f:
                            json.dump(annotations, f)           
        
        # Compute and store new ranks
        accuracy, ranks, docs_probs = self.evaluate(max_rank=1)
        ranks_file_path = os.path.join(self.output_folder_path, f'Ranks_k={k}.csv')
        ranks.tofile(ranks_file_path, sep=",")

        if verbose: logging.info(f"Total number of steps = {total_n_steps}")

        return annotations, total_n_steps

    def mask_most_disclosive_term(self, document:dict, splits_probs:torch.tensor, annotated_terms:dict, plot_explanations:bool=False):
        n_masked_terms = 0
        n_steps = 0
        complete_text = document["text"]
        splits = document["splits"]
        label = document["label"]
        name = self.label_to_name[label]           
        annotated_terms[name] = annotated_terms.get(name, set())
        
        # Get splits sorted by disclosiveness
        individual_probs = splits_probs[:, label]
        sorted_splits_idxs = torch.argsort(individual_probs, descending=True)

        # Following disclosiveness order, search to mask the most disclosive term of the most disclosive split
        most_disclosive_term = None
        for split_idx in sorted_splits_idxs:
            split = splits[split_idx]

            # If not all terms in the split are already masked (avoid infinite loops)
            if len(split["masked_terms_idxs"]) < len(split["terms_spans"]):
                # Get split's most disclosive term
                most_disclosive_term_idx, most_disclosive_term = self.get_most_disclosive(complete_text, split,
                                                                                          label,
                                                                                          plot_explanations=plot_explanations)
                n_steps += 1

                # If there is a term to mask, end of process
                if most_disclosive_term is not None:
                    n_masked_terms = self.mask(most_disclosive_term, most_disclosive_term_idx,
                                                split_idx, splits, splits_probs, complete_text)
                    annotated_terms[name].update(most_disclosive_term)
                    break # Exit loop

            # Next split
            split_idx += 1
        
        return most_disclosive_term, n_masked_terms, n_steps

    def get_most_disclosive(self, complete_text:str, split:dict, label:int, plot_explanations:bool=False)->tuple:
        text_span = split["text_span"]
        split_text = complete_text[text_span[0]:text_span[1]]
        terms_spans = split["terms_spans"]
        terms_to_tokens = split["terms_to_tokens"]
        masked_terms_idxs = split["masked_terms_idxs"]

        # Get explanation terms' weights
        annotated_text = self.dataset.annotate_text(split_text, split)
        terms_weights = self.explainability_method(split_text, label, split, plot_explanations=plot_explanations)

        # Get index of most disclosive term index
        most_disclosive_term_idx = self.get_most_disclosive_term_idx(terms_weights)

        # Get text of the most disclosive term
        if most_disclosive_term_idx >= 0:
           most_disclosive_term_span = terms_spans[most_disclosive_term_idx]
           most_disclosive_term = split_text[most_disclosive_term_span[0]:most_disclosive_term_span[1]]
        else:
           most_disclosive_term = None        

        return most_disclosive_term_idx, most_disclosive_term
    
    def get_most_disclosive_term_idx(self, terms_weights:np.array)->int:
        most_disclosive_term_idx = -1

        # Sort weights from maximum to minimum
        sorted_terms_idxs = np.argsort(terms_weights)[::-1]

        # For each term idx
        for term_idx in sorted_terms_idxs:
            term_weight = terms_weights[term_idx]
            # If term already masked, exit loop
            if term_weight == float("-inf"):
                break
            # Otherwise, most disclosive term found
            else:                
                most_disclosive_term_idx = term_idx
                break

        return most_disclosive_term_idx

    def mask(self, most_disclosive_term:str, most_disclosive_term_idx:int, split_idx:int, splits:list, splits_probs:np.array, complete_text:str)->int:
        n_masked_terms = 0

        # If enabled, mask all instances of the most_disclosive_term
        if self.use_mask_all_instances:
            n_masked_terms += self.mask_all_instances(complete_text, splits, splits_probs, most_disclosive_term)
        # Otherwise, only mask the term within the most disclosive split
        else:
            split = splits[split_idx]
            self.dataset.mask_terms(split, [most_disclosive_term_idx])
            # Revaluate/update the split probabilities
            splits_probs[split_idx, :] = self.evaluate_split(complete_text, split)
            n_masked_terms += 1
        
        return n_masked_terms

    def mask_all_instances(self, complete_text:str, splits:list, splits_probs:np.array, most_disclosive_term:str) -> int:
        n_masked_terms = 0

        for split_idx, split in enumerate(splits):
            text_span = split["text_span"]
            terms_spans = split["terms_spans"]            

            # Search other instances of the most_disclosive_term
            terms_idxs_to_mask = []
            for term_idx, span in enumerate(terms_spans):
                span_len = span[1]-span[0]
                if span_len == len(most_disclosive_term):
                    span_text = complete_text[text_span[0]+span[0]:text_span[0]+span[1]]
                    if span_text == most_disclosive_term:
                        terms_idxs_to_mask.append(term_idx)
            
            # If at least one new term is masked
            if len(terms_idxs_to_mask) > 0:
                self.dataset.mask_terms(split, terms_idxs_to_mask)
                # Revaluate/update the split probabilities
                splits_probs[split_idx, :] = self.evaluate_split(complete_text, split)
            
            n_masked_terms += len(terms_idxs_to_mask)
        
        return n_masked_terms
    
    #endregion

    #endregion

#endregion

#region ###################################### PETRE dataset ######################################

class PETREDataset(Dataset):
  def __init__(self, df:pd.DataFrame, tokenizer, name_to_label:dict, mask_text:str, use_chunking:bool):
      # Dataframe must have two columns: name and text
      assert len(df.columns) == 2
      self.df = df

      # Set general attributes
      self.tokenizer = tokenizer
      self.name_to_label = name_to_label
      self.label_to_name = {value:key for key, value in self.name_to_label.items()}
      self.spacy_nlp = en_core_web_lg.load()
      self.use_chunking = use_chunking
      self.mask_text = mask_text
      self.tokenized_mask = self.tokenizer.encode(self.mask_text, add_special_tokens=False)
      self.terms_to_ignore = self.get_terms_to_ignore()

      # Compute inputs and labels
      self.generate()

  def get_terms_to_ignore(self):
    stopwords = self.spacy_nlp.Defaults.stop_words
    terms_to_ignore = set()
    terms_to_ignore.update(stopwords) # Stopwords as base of terms to ignore
    terms_to_ignore.update({self.mask_text, "[CLS]", "[SEP]", "[PAD]", "", " ", "\t", "\n"}) # Add special tokens

    return terms_to_ignore

  def generate(self, gc_freq=5):
        texts_column = list(self.df[self.df.columns[1]])
        names_column = list(self.df[self.df.columns[0]])
        labels_idxs = list(map(lambda x: self.name_to_label[x], names_column))   # Compute labels, translated to the identity index

        # Sentence splitting
        self.documents = [None] * len(labels_idxs)
        # For each document
        for idx, (text, label) in tqdm(enumerate(zip(texts_column, labels_idxs)), total=len(texts_column),
                                                desc="Processing sentence splitting"):
            # Split into sentences
            splits = []
            document = {"text": text, "label": label, "splits": splits}
            doc = self.spacy_nlp(text)
            for sentence in doc.sents:
                sentence_txt = text[sentence.start_char:sentence.end_char]                
                sent_token_count = len(self.tokenizer.encode(sentence_txt, add_special_tokens=True))
                # If sentence token length is greater than model's maximum, error
                if sent_token_count > self.tokenizer.model_max_length:                    
                    error_text = f"WARNING: Sentence with length {sent_token_count} > {self.tokenizer.model_max_length} at index {idx} with label {label}"
                    error_text += "\nSentence will be truncated to fit model's maximum length."
                    logging.warning(error_text)
                # Otherwise, compute sentence/split information
                else:
                    terms_spans = self.get_terms_spans(sentence, use_chunking=self.use_chunking)
                    terms_to_tokens = self.get_terms_to_tokens(terms_spans, sentence_txt)
                    splits.append({"text_span": (sentence.start_char, sentence.end_char),
                                    "terms_spans": terms_spans,
                                    "terms_to_tokens": terms_to_tokens,
                                    "masked_terms_idxs": []})

            # Store document in self.documents
            self.documents[label] = document

            # Delete document for reducing memory consumption
            del doc

            # Periodically use GarbageCollector for reducing memory consumption
            if idx % gc_freq == 0:
                gc.collect()

  def get_terms_spans(self, sentence_span, use_chunking:bool)->list:
        text_spans = []
        added_tokens = np.zeros(len(sentence_span), dtype=bool)
        start_char_idx = sentence_span.start_char
        start_token_idx = sentence_span.start
        special_chars_pattern = re.compile(r"[^\nA-Za-z0-9À-ÖØ-öø-ÿЀ-ӿ/]+")

        if use_chunking:
            for chunk in sentence_span.noun_chunks:
                chunk_start = chunk.start - start_token_idx # Adjust indexes within sentence_span
                chunk_end = chunk.end - start_token_idx
                start = sentence_span[chunk_start].idx
                last_token = sentence_span[chunk_end - 1]
                end = last_token.idx + len(last_token)
                text_spans.append((start, end))
                added_tokens[chunk_start:chunk_end] = True
            
            for chunk in sentence_span.ents:
                chunk_start = chunk.start - start_token_idx # Adjust indexes within sentence_span
                chunk_end = chunk.end - start_token_idx
                # If is it not already added
                if not added_tokens[chunk_start:chunk_end].any():
                    start = sentence_span[chunk_start].idx
                    last_token = sentence_span[chunk_end - 1]
                    end = last_token.idx + len(last_token)
                    text_spans.append((start, end))
                    added_tokens[chunk_start:chunk_end] = True            

        # Add text spans after last chunks (or all spans, if chunks are ignored)
        for token_idx in range(len(sentence_span)):
            if not added_tokens[token_idx]:
                token = sentence_span[token_idx]
                clean_token_text = re.sub(special_chars_pattern, '', token.text).strip()
                if clean_token_text not in self.terms_to_ignore:  # Avoiding undesired terms
                    start = token.idx
                    end = start + len(token)
                    text_spans.append((start, end))

        # Sort text spans by starting position
        text_spans = sorted(text_spans, key=lambda span: span[0], reverse=False)

        # Move all spans values considering sentence_span offset
        text_spans = list(map(lambda x:(x[0]-start_char_idx,x[1]-start_char_idx), text_spans))

        return text_spans

  def get_terms_to_tokens(self, terms_spans, text):
    terms_to_tokens = []
    results = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = results["offset_mapping"]

    last_token_idx = 0
    for (term_start, term_end) in terms_spans:
      term_tokens = []
      for token_idx in range(last_token_idx, len(offsets)):
        (token_start, token_end) = offsets[token_idx]
        if term_start <= token_start < term_end:
          term_tokens.append(token_idx+1) #+1 Due to the [CLS] token
        elif len(term_tokens) > 0: # If first token not contained, is end of term
          break
      last_token_idx = token_idx # Store last token to continue search
      terms_to_tokens.append(term_tokens)

    return terms_to_tokens

  def add_annotations(self, annotations, disable_tqdm:bool=True):
    # For annotations from multiple documents
    for name, annots in tqdm(annotations.items(), total=len(annotations), desc="Adding annotations", disable=disable_tqdm):
      if name not in self.name_to_label:
        continue
      label = self.name_to_label[name]
      document = self.documents[label]
      # For each annotation span
      for annotation_span in annots:
        splits_spans = [split["text_span"] for split in document["splits"]]
        intersecting_splits_idxs = self.get_intersecting_spans(annotation_span, splits_spans)        
        # For each split where the span intersects
        for split_idx in intersecting_splits_idxs:
          split = document["splits"][split_idx]
          split_span = split["text_span"]
          terms_spans = [(split_span[0]+span[0], split_span[0]+span[1]) for span in split["terms_spans"]]
          annotated_terms_idxs = self.get_intersecting_spans(annotation_span, terms_spans)
          self.mask_terms(split, annotated_terms_idxs)

  def get_intersecting_spans(self, ref_span, spans_list):
    intersecting_spans_idxs = []
    sorted_spans = sorted(spans_list, key=lambda span: span[0], reverse=False)
    for span_idx, span in enumerate(sorted_spans):
      # If intersect
      if span[0] <= ref_span[0] < span[1] or \
        span[0] <= ref_span[1] < span[1] or \
        ref_span[0] <= span[0] < ref_span[1]:
        intersecting_spans_idxs.append(span_idx)
      else:
        if span[1] > ref_span[0]:
          break
    return intersecting_spans_idxs

  def mask_terms(self, split, terms_idx_to_mask):
    terms_to_tokens = split["terms_to_tokens"]
    masked_terms_idxs = split["masked_terms_idxs"]

    # Sort terms index from first to last
    sorted_terms_idx_to_mask = sorted(terms_idx_to_mask, reverse=False)

    # Apply tokens offset to following tokens in terms_to_tokens
    for term_idx in sorted_terms_idx_to_mask:
      if term_idx not in masked_terms_idxs: # Check to avoid re-masking
        n_term_tokens = len(terms_to_tokens[term_idx])
        n_mask_tokens = len(self.tokenized_mask)
        tokens_offset = n_mask_tokens - n_term_tokens
        # For all the following terms
        for next_term_idx in range(term_idx+1, len(terms_to_tokens)):
          next_term_tokens = terms_to_tokens[next_term_idx]
          # For all the tokens of that term
          for idx, token_idx in enumerate(next_term_tokens):
            next_term_tokens[idx] = token_idx + tokens_offset

        # Add masked term to list
        masked_terms_idxs.append(term_idx)

  def annotate_text(self, text, split):
    terms_spans = split["terms_spans"]
    masked_terms_idxs = split["masked_terms_idxs"]
    sorted_masked_terms_idxs = sorted(masked_terms_idxs, reverse=True)
    annotated_text = text
    for term_idx in sorted_masked_terms_idxs:
      start, end = terms_spans[term_idx]
      annotated_text = annotated_text[:start] + self.mask_text + annotated_text[end:]
    return annotated_text

  def get_annotations(self, disable_tqdm:bool=True):
    annotations = {}
    # For each document
    for document in tqdm(self.documents, desc="Getting annotations", disable=disable_tqdm):
      doc_annotations = []
      # For each split within the document
      for split in document["splits"]:
        split_span = split["text_span"]
        terms_spans = split["terms_spans"]
        masked_terms_idxs = split["masked_terms_idxs"]
        start_t = time.time()
        sorted_masked_terms_idx = sorted(masked_terms_idxs, reverse=False)
        
        # For each masked term in the split
        for term_idx in sorted_masked_terms_idx:
          local_span = terms_spans[term_idx] # Span within the sentence
          global_span = [split_span[0] + local_span[0], # Span within the text
                         split_span[0] + local_span[1]]
          doc_annotations.append(global_span)

      # Store annotations
      label = document["label"]
      name = self.label_to_name[label]
      annotations[name] = doc_annotations

    return annotations

  def tokenize_texts(self, texts):
    inputs = self.tokenizer(texts,
                            add_special_tokens=True,
                            padding="longest",  # Warning: If an input_text is longer than tokenizer.model_max_length, an error will raise on prediction
                            truncation=False,
                            return_offsets_mapping=True, # Requiered for annotations. Use it with ["offset_mapping"]
                            return_tensors="pt")
    return inputs

  def get_all_texts(self, use_annotated:bool):
        texts = []
        doc_to_text_idxs = {}
        for document in tqdm(self, total=len(self)):
            complete_text = document["text"]
            label = document["label"]
            splits = document["splits"]
            doc_inputs_idxs = []

            # Get texts for all splits
            for split in splits:
                split_span = split["text_span"]
                split_text = complete_text[split_span[0]:split_span[1]]
                if use_annotated:
                    split_text = self.annotate_text(split_text, split)
                
                # Store as input
                doc_inputs_idxs.append(len(texts))
                texts.append(split_text)
            
            # Store mapping for later retrieval
            doc_to_text_idxs[label] = doc_inputs_idxs
        
        return texts, doc_to_text_idxs

  def __len__(self):
    return len(self.documents)

  def __getitem__(self, index):
    document = self.documents[index]
    return document

#endregion

#region ###################################### Main CLI ######################################
if __name__ == "__main__":
    verbose=True

    # Load configuration
    if verbose: logging.info("######### START: CONFIGURATION #########")
    target_dir = argument_parsing()
    config = get_config_from_file(target_dir)
    petre = PETRE(**config)
    if verbose: logging.info("######### END: CONFIGURATION #########")
    
    # Run all sections
    petre.run(verbose=verbose)
#endregion
