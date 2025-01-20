'''
Implement metrics as a class
'''

import os
import warnings
import logging
import sys
import contextlib

from transformers.utils import logging

# Set the verbosity level to error to suppress info and warnings
logging.set_verbosity_error()


# Suppress warnings from other libraries
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.metrics import jaccard_distance
from nltk.util import ngrams
from bert_score import score
from collections import Counter
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.spatial.distance import euclidean
from tqdm import tqdm
from collections import defaultdict

import textstat

from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

from diversity import compression_ratio, homogenization_score, ngram_diversity_score

import contextlib
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class Metrics:
    def __init__(self, pairs, author_profile, author_index, compute_gt=True):
        '''
        pairs: list of tuples of reference and generated story turns
        '''
        self.pairs = pairs
        self.author_profile = author_profile
        self.author_index = author_index

        self.compute_gt = compute_gt

        # # load the NRC lexicon
        # self.val_dict, self.aro_dict, self.dom_dict = get_NRC_lexicon()


        # set the device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def compute_rouge(self):
        '''
        Compute the average ROUGE scores across all the pairs
        '''
        # Initialize the ROUGE scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Initialize accumulators for the scores
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []

        # Iterate over the pairs and compute ROUGE scores
        for ref, gen in self.pairs:
            scores = scorer.score(ref, gen)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)

        # Compute the average scores
        avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0
        avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0
        avg_rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0

        piontwise_scores = {
            'rouge1': rouge1_scores,
            'rouge2': rouge2_scores,
            'rougeL': rougeL_scores
        }

        final_scores = {
            'rouge1': avg_rouge1,
            'rouge2': avg_rouge2,
            'rougeL': avg_rougeL
        }

        # return piontwise_scores, final_scores
    
        return final_scores
    
    def compute_bleu(self):
        '''
        Compute the average BLEU scores across all the pairs
        '''
        # Initialize accumulators for the scores
        bleu_scores = []
        backwards_bleu_scores = []

        # Smoothing function to avoid zero scores for short sentences
        smoothing_function = SmoothingFunction().method1

        # Iterate over the pairs and compute BLEU scores
        for ref, gen in self.pairs:
            # Tokenize the reference and generated sentences
            ref_tokens = [ref.split()]
            gen_tokens = gen.split()
            # Compute the BLEU score
            bleu_score = sentence_bleu(ref_tokens, gen_tokens, smoothing_function=smoothing_function)
            bleu_scores.append(bleu_score)

            # Compute the BLEU score in the reverse direction
            # Tokenize the reference and generated sentences
            ref_tokens = ref.split()
            gen_tokens = [gen.split()]

            back_bleu_score = sentence_bleu(gen_tokens, ref_tokens, smoothing_function=smoothing_function)
            backwards_bleu_scores.append(back_bleu_score)

        # Compute the average BLEU score
        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0

        # Compute the average BLEU score in the reverse direction
        avg_backwards_bleu = sum(backwards_bleu_scores) / len(backwards_bleu_scores) if backwards_bleu_scores else 0

        pointwise_scores = {
            'bleu': bleu_scores,
            'backwards_bleu': backwards_bleu_scores
        }

        final_scores = {
            'bleu': avg_bleu,
            'backwards_bleu': avg_backwards_bleu
        }

        # return pointwise_scores, final_scores
        return final_scores
    
    def compute_bert_score(self):
        '''
        Compute the BERT score
        '''
        # Check if a GPU is available and set the device accordingly
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Initialize accumulators for the scores
        precision_scores = []
        recall_scores = []
        f1_scores = []

        # Batch size
        batch_size = 32

        # Iterate over the pairs in batches and compute BERT scores
        for i in range(0, len(self.pairs), batch_size):
            batch_pairs = self.pairs[i:i + batch_size]
            refs = [ref for ref, gen in batch_pairs]
            gens = [gen for ref, gen in batch_pairs]

            P, R, F1 = score(gens, refs, lang="en", verbose=False, device=device)
            precision_scores.extend(P.tolist())
            recall_scores.extend(R.tolist())
            f1_scores.extend(F1.tolist())

        # Compute the average scores
        avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
        avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

        # pointwise scores 
        pointwise_scores = {
            'precision': precision_scores,
            'recall': recall_scores,
            'f1': f1_scores
        }

        final_scores = {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1
        }

        # return pointwise_scores, final_scores
        return final_scores

    def compute_jaccard_distance(self):
        '''
        Compute the Jaccard distance
        '''
        # Initialize accumulators for the scores
        jaccard_scores = []

        # Iterate over the pairs and compute Jaccard distance
        for ref, gen in self.pairs:
            # Tokenize the reference and generated sentences
            ref_tokens = set(ref.split())
            gen_tokens = set(gen.split())

            # Compute the Jaccard distance
            score = jaccard_distance(ref_tokens, gen_tokens)
            jaccard_scores.append(score)

        # Compute the average Jaccard distance
        avg_jaccard = sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0


        # pointwise scores
        pointwise_scores = {
            'jaccard_distance': jaccard_scores
        }

        final_scores = {
            'jaccard_distance': avg_jaccard
        }

        # return pointwise_scores, final_scores  
        return final_scores  

    def diversity_scores(self, config, verbose=False):
        '''
        compute compression ratio, homogenization score, ngram diversity score, and lexical repition
        '''

        def get_homogenization_score(homo_type, verbose=False):
            # calculate overall homogenization score
            # with suppress_output():
            homogenization_score_gt = homogenization_score(gt, homo_type, verbose=False) if self.compute_gt else 'None'
            homogenization_score_gen = homogenization_score(gen, homo_type, verbose=False)

            # final scores
            overall_homo_score = {
                f'homogenization_score_gt_{homo_type}': homogenization_score_gt,
                f'homogenization_score_gen_{homo_type}': homogenization_score_gen
            }

            return overall_homo_score

    
        # Deconstruct the pairs
        gt, gen = zip(*self.pairs)

        # compression ratio
        if config['compression_ratio']:
            # calculate overall compression ratio
            compression_ratio_gt = compression_ratio(gt) if self.compute_gt else 'None'
            compression_ratio_gen = compression_ratio(gen)

            # final scores
            overall_compression_scores = {
                'compression_ratio_gt': compression_ratio_gt,
                'compression_ratio_gen': compression_ratio_gen
            }

            if verbose:
                print('Computed compression ratio')
        else:
            overall_compression_scores = 'None'
        

        # 1. Homogenization Rouge-L
        if config['homo_rougel']:
            overall_homo_rougel = get_homogenization_score('rougel')
            if verbose:
                print('Computed Homogenization RougeL Score')

        else:
            overall_homo_rougel = 'None'


        # 2. Homogenization BertScore
        if config['homo_bert']:
            overall_homo_bert = get_homogenization_score('bertscore')
            if verbose:
                print('Computed Homogenization BERTScore Score')

        else:
            overall_homo_bert = 'None'

        # 3. Homogenization BLEU
        if config['homo_bleu']:
            overall_homo_bleu = get_homogenization_score('bleu')
            if verbose:
                print('Computed Homogenization BLEU Score')
        else:
            overall_homo_bleu = 'None'


        if config['ngram']:
            # calculate overall ngram diversity score
            ngram_diversity_score_gt = ngram_diversity_score(gt, 4) if self.compute_gt else 'None'
            ngram_diversity_score_gen = ngram_diversity_score(gen, 4)

            # final scores
            overall_ngram_diversity_scores = {
                'ngram_diversity_score_gt': ngram_diversity_score_gt,
                'ngram_diversity_score_gen': ngram_diversity_score_gen
            }
            if verbose:
                print('Computed ngram diversity score')

        else:
            overall_ngram_diversity_scores = 'None'
        

        # final scores 
        final_scores = {
            'compression_scores': overall_compression_scores,
            'homogenization_scores_bleu': overall_homo_bleu,
            'homogenization_scores_rougel': overall_homo_rougel,
            'homogenization_scores_bertscore': overall_homo_bert,
            'ngram_diversity_scores': overall_ngram_diversity_scores,
        }

        return final_scores
    
    def length_scores(self):
        '''
        Calculate the average length of the stories
        '''

        # Deconstruct the pairs
        gt, gen = zip(*self.pairs)

        def compute_length_scores(stories):
            lengths = [len(story.split()) for story in stories]
            avg_length = sum(lengths) / len(lengths) if lengths else 0
            return lengths, avg_length

        # Compute the average length for ground truth and generated stories
        if self.compute_gt:
            lengths_gt, avg_length_gt = compute_length_scores(gt)
        else:
            lengths_gt, avg_length_gt = 'None', 'None'

        lengths_gen, avg_length_gen = compute_length_scores(gen)

        # pointwise scores
        pointwise_scores = {
            'lengths_gt': lengths_gt,
            'lengths_gen': lengths_gen
        }

        # final scores
        final_scores = {
            'avg_length_gt': avg_length_gt,
            'avg_length_gen': avg_length_gen
        }

        # return pointwise_scores, final_scores
        return final_scores

    def flesch_reading_scores(self):
        '''
        Calculate the Flesch reading ease scores
        '''

        # Deconstruct the pairs
        gt, gen = zip(*self.pairs)

        def compute_flesch_reading_scores(stories):
            scores = [textstat.flesch_reading_ease(story) for story in stories]
            avg_score = sum(scores) / len(scores) if scores else 0
            return scores, avg_score

        # Compute the Flesch reading ease scores for ground truth and generated stories
        if self.compute_gt:
            flesch_reading_scores_gt, avg_flesch_reading_score_gt = compute_flesch_reading_scores(gt)
        else:
            flesch_reading_scores_gt, avg_flesch_reading_score_gt = 'None', 'None'
        flesch_reading_scores_gen, avg_flesch_reading_score_gen = compute_flesch_reading_scores(gen)

        # pointwise scores
        pointwise_scores = {
            'flesch_reading_scores_gt': flesch_reading_scores_gt,
            'flesch_reading_scores_gen': flesch_reading_scores_gen
        }

        # final scores
        final_scores = {
            'avg_flesch_reading_score_gt': avg_flesch_reading_score_gt,
            'avg_flesch_reading_score_gen': avg_flesch_reading_score_gen
        }

        # return pointwise_scores, final_scores
        return final_scores

    # Helper function to get embeddings
    def get_luar_embeddings(self, text_series, luar_tokenizer, luar_embedder):
        # Batch size for embedding computation
        batch_size = 32
        num_batches = (len(text_series) - 1) // batch_size + 1
        embeddings = []
        for i in range(num_batches):
            texts = list(text_series[i * batch_size: (i + 1) * batch_size])
            actual_batch_size = len(texts)

            # Tokenize and reshape
            tokenized_texts = luar_tokenizer(texts, max_length=512, padding="max_length", truncation=True, return_tensors="pt").to(self.device)
            tokenized_texts['input_ids'] = tokenized_texts['input_ids'].reshape(actual_batch_size, 1, -1)
            tokenized_texts['attention_mask'] = tokenized_texts['attention_mask'].reshape(actual_batch_size, 1, -1)

            # Get embeddings
            with torch.no_grad():
                out = luar_embedder(**tokenized_texts)
            embeddings.append(out.detach().cpu())
        return torch.cat(embeddings, dim=0)

    def luar_score(self):
        '''
        Compute the LUAR-based cosine similarity score for text pairs.
        '''
        # Load the LUAR tokenizer and model
        luar_tokenizer = AutoTokenizer.from_pretrained('rrivera1849/LUAR-CRUD', trust_remote_code=True)
        luar_embedder = AutoModel.from_pretrained('rrivera1849/LUAR-CRUD', trust_remote_code=True).to(self.device)

        # Initialize accumulators for cosine similarities
        cosine_similarities = []

        # Separate references and generations
        refs, gens = zip(*self.pairs)

        # Compute embeddings for references and generated texts
        ref_embeddings = self.get_luar_embeddings(refs, luar_tokenizer, luar_embedder)
        gen_embeddings = self.get_luar_embeddings(gens, luar_tokenizer, luar_embedder)

        # Compute cosine similarities for each pair
        for ref_emb, gen_emb in zip(ref_embeddings, gen_embeddings):
            cosine_sim = cosine_similarity(ref_emb.unsqueeze(0), gen_emb.unsqueeze(0))[0][0]
            cosine_similarities.append(round(float(cosine_sim), 4))

        # Compute the average cosine similarity
        avg_cosine_similarity = sum(cosine_similarities) / len(cosine_similarities) if cosine_similarities else 0

        # pointwise scores
        pointwise_scores = {
            'luar_cosine_similarities': cosine_similarities
        }

        # final scores
        final_scores = {
            'avg_luar_cosine_similarity': avg_cosine_similarity
        }

        # return pointwise_scores, final_scores
        return final_scores
    
    def author_attribution(self, verbose=False):
        '''
        Compute the author attribution scores
        '''

        # Separate references and generations
        refs, author_stories = zip(*self.pairs)
        author_index = self.author_index

        # Load the LUAR tokenizer and model
        luar_tokenizer = AutoTokenizer.from_pretrained('rrivera1849/LUAR-CRUD', trust_remote_code=True)
        luar_embedder = AutoModel.from_pretrained('rrivera1849/LUAR-CRUD', trust_remote_code=True).to(self.device)

        # get embeddings for author profile
        author_embeddings = []
        for author, stories in self.author_profile.items():
            embeddings = self.get_luar_embeddings(stories, luar_tokenizer, luar_embedder)
            # agregate the embeddings for each author
            author_embeddings.append(embeddings.mean(dim=0))
        
        # convert the list of embeddings to a tensor
        author_embeddings = torch.stack(author_embeddings)
        # if verbose:
        #     print('Computed author embeddings', author_embeddings.shape)

        # get embeddings for author test data
        test_embeddings = self.get_luar_embeddings(author_stories, luar_tokenizer, luar_embedder)
        # if verbose:
        #     print('Computed test embeddings', test_embeddings.shape)

        # compute cosine similarity between author profile and author test data (don't use for loop)
        cosine_similarities = cosine_similarity(test_embeddings, author_embeddings)
        
        if self.compute_gt:
            # calculate the cosine similarity between the ground truth and the author profile
            gt_embeddings = self.get_luar_embeddings(refs, luar_tokenizer, luar_embedder)            
            gt_cosine_similarities = cosine_similarity(gt_embeddings, author_embeddings)

        # NOTE: Authorship Attribution Cosine Similarity
        # Group cosine similarities by author index
        authorwise_scores = defaultdict(list)
        for idx, author_idx in enumerate(author_index):
            authorwise_scores[author_idx].append(cosine_similarities[idx, author_idx])

        # compute average cosine similarity for each author
        authorwise_avg_scores = {author: sum(scores) / len(scores) for author, scores in authorwise_scores.items()}

        # compute average of average cosine similarity
        author_attr_cosine_similarity = sum(authorwise_avg_scores.values()) / len(authorwise_avg_scores)

        # compute gt
        if self.compute_gt:
            # Group cosine similarities by author index
            gt_authorwise_scores = defaultdict(list)
            for idx, author_idx in enumerate(author_index):
                gt_authorwise_scores[author_idx].append(gt_cosine_similarities[idx, author_idx])

            # compute average cosine similarity for each author
            gt_authorwise_avg_scores = {author: sum(scores) / len(scores) for author, scores in gt_authorwise_scores.items()}

            # compute average of average cosine similarity
            gt_author_attr_cosine_similarity = sum(gt_authorwise_avg_scores.values()) / len(gt_authorwise_avg_scores)
        else:
            gt_author_attr_cosine_similarity = 'None'

        # final scores
        final_scores = {
            'author_attr_cosine_similarity_gt': gt_author_attr_cosine_similarity,
            'author_attr_cosine_similarity_gen': author_attr_cosine_similarity
        }

        # NOTE: Authorship Attribution Accuracy
        # # get the index of the maximum cosine similarity for each test data
        # predicted_author_index = np.argmax(cosine_similarities, axis=1)
        # if verbose:
        #     print('Computed predicted author index', predicted_author_index.shape)

        # # check if the predicted author index is correct with author_index
        # correct_predictions = (predicted_author_index == author_index).sum()

        # # compute the accuracy
        # accuracy = correct_predictions / len(author_index)

        # # pointwise scores
        # pointwise_scores = {
        #     'author_attribution_accuracy': accuracy
        # }

        # # final scores
        # final_scores = {
        #     'author_attribution_accuracy': accuracy
        # }

        return final_scores