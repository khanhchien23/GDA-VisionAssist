import math
import numpy as np
from collections import Counter, defaultdict
import re

class VGEvaluationMetrics:
    """
    Implements standard captioning metrics: BLEU-1..4, ROUGE-L, CIDEr.
    Designed for use with Visual Genome Region Descriptions.
    """
    
    @staticmethod
    def normalize_string(s):
        """Standard normalization for evaluation."""
        if not s:
            return ""
        s = s.lower().strip()
        s = re.sub(r'[^\w\s]', '', s) # Remove punctuation
        s = ' '.join(s.split()) # Normalize whitespace
        return s

    @staticmethod
    def tokenize(s):
        """Simple whitespace tokenizer after normalization."""
        return VGEvaluationMetrics.normalize_string(s).split()

    @staticmethod
    def compute_bleu(reference, hypothesis, max_n=4):
        """
        Computes BLEU-1, BLEU-2, BLEU-3, BLEU-4.
        Returns a dictionary with all scores.
        """
        ref_tokens = VGEvaluationMetrics.tokenize(reference)
        hyp_tokens = VGEvaluationMetrics.tokenize(hypothesis)
        
        scores = {}
        
        # Brevity Penalty
        if len(hyp_tokens) == 0:
            bp = 0.0
        else:
            bp = math.exp(min(1 - len(ref_tokens) / len(hyp_tokens), 0)) if len(hyp_tokens) > 0 else 0.0

        for n in range(1, max_n + 1):
            if len(hyp_tokens) < n:
                scores[f'Bleu_{n}'] = 0.0
                continue
            
            hyp_ngrams = Counter([tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens)-n+1)])
            ref_ngrams = Counter([tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)])
            
            # Clip counts
            clipped_counts = {ng: min(count, ref_ngrams[ng]) for ng, count in hyp_ngrams.items()}
            numerator = sum(clipped_counts.values())
            denominator = max(1, sum(hyp_ngrams.values()))
            
            precision = numerator / denominator
            scores[f'Bleu_{n}'] = precision

        # Compute Geometric Mean for final scores (Standard BLEU definition)
        # Note: Usually BLEU-4 implies the cumulative score.
        final_scores = {}
        for n in range(1, max_n + 1):
            precisions = [scores[f'Bleu_{i}'] for i in range(1, n + 1)]
            if any(p == 0 for p in precisions):
                score = 0.0
            else:
                log_sum = sum(math.log(p) for p in precisions)
                score = bp * math.exp(log_sum / n)
            final_scores[f'BLEU-{n}'] = score
            
        return final_scores

    @staticmethod
    def compute_rouge_l(reference, hypothesis):
        """
        Computes ROUGE-L (Longest Common Subsequence based).
        """
        ref_tokens = VGEvaluationMetrics.tokenize(reference)
        hyp_tokens = VGEvaluationMetrics.tokenize(hypothesis)
        
        if not ref_tokens or not hyp_tokens:
            return 0.0

        m = len(ref_tokens)
        n = len(hyp_tokens)
        
        # DP table for LCS
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_tokens[i-1] == hyp_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_len = dp[m][n]
        
        p = lcs_len / n if n > 0 else 0
        r = lcs_len / m if m > 0 else 0
        
        beta = 1.2 # ROUGE-L often uses beta=1.2 which favors recall
        if p + r == 0:
            return 0.0
        
        # Standard ROUGE F-measure
        f_score = ((1 + beta**2) * p * r) / (r + (beta**2) * p)
        return f_score

    @staticmethod
    def compute_cider_simple(reference, hypothesis):
        """
        Simplified CIDEr computation for a single reference.
        NOTE: True CIDEr requires IDF from the entire dataset. 
        This is a local approximation based on cosine similarity of TF-IDF vectors
        assuming the 'corpus' is just this single pair (IDF=1), which essentially degrades to Cosine Similarity of TF.
        
        For rigorous CIDEr, we need the whole dataset frequencies. 
        Will implement a clearer version if dataset stats are available.
        For now, this serves as a placeholder or 'local' content match.
        """
        # To truly support CIDEr, we'd need a pre-computed IDF dictionary.
        # Given the scope, we might fallback to a simpler n-gram cosine similarity or
        # implement a class that pre-scans the dataset.
        
        # Let's implement a "Sentence Comparison" using Cosine Similarity of 1-4 grams
        # This acts similarly to CIDEr without the global weighting.
        
        def get_ngrams(tokens, n):
            return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
            
        ref_tokens = VGEvaluationMetrics.tokenize(reference)
        hyp_tokens = VGEvaluationMetrics.tokenize(hypothesis)
        
        score_sum = 0
        count = 0
        
        for n in range(1, 5):
            ref_ng = Counter(get_ngrams(ref_tokens, n))
            hyp_ng = Counter(get_ngrams(hyp_tokens, n))
            
            if not ref_ng or not hyp_ng:
                continue
                
            # Vector magnitude (norm)
            ref_norm = math.sqrt(sum(v**2 for v in ref_ng.values()))
            hyp_norm = math.sqrt(sum(v**2 for v in hyp_ng.values()))
            
            if ref_norm == 0 or hyp_norm == 0:
                continue
                
            # Dot product
            dot_prod = sum(min(ref_ng[ng], hyp_ng[ng]) for ng in hyp_ng if ng in ref_ng)
            # Or strictly: sum(ref_ng[ng] * hyp_ng[ng]) for standard vector dot product
            # Standard CIDEr uses dot product of TF-IDF vectors.
            # Here we use dot product of TF vectors.
            dot_prod = sum(ref_ng.get(ng,0) * hyp_ng.get(ng,0) for ng in set(list(ref_ng.keys()) + list(hyp_ng.keys())))

            cosine = dot_prod / (ref_norm * hyp_norm)
            score_sum += cosine
            count += 1
            
        return (score_sum / count * 10.0) if count > 0 else 0.0

    @staticmethod
    def compute_object_accuracy(reference, hypothesis):
        """
        Checks recall of nouns/objects from reference in hypothesis.
        Uses simple tag matching.
        """
        # Basic noun extraction (naïve approach without spaCy/NTLK for speed/dependency-free)
        # We assume important objects are non-stopwords.
        
        # A simple list of common stopwords to ignore
        stopwords = {
            'a', 'an', 'the', 'is', 'are', 'in', 'on', 'of', 'with', 'to', 'and', 'at', 'it', 'that', 'this',
            'there', 'shows', 'marked', 'region', 'picture', 'image', 'photo', 'display', 'displays',
            'features', 'contains', 'has', 'visible', 'seen', 'being', 'some', 'any'
        }
        
        ref_words = set(VGEvaluationMetrics.tokenize(reference)) - stopwords
        hyp_words = set(VGEvaluationMetrics.tokenize(hypothesis)) - stopwords
        
        if not ref_words:
            return 1.0 # Empty reference implies nothing to miss
            
        # Recall: How many ref words are in hypothesis?
        matches = len(ref_words.intersection(hyp_words))
        recall = matches / len(ref_words)
        
        return recall
