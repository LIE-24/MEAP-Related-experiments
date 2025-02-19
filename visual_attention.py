import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import pandas as pd

class AttentionAnalyzer:
    def __init__(self, model_path):
        self.model_path = model_path
        print(f"\n  {model_path} Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            output_attentions=True,
            output_hidden_states=True,
            trust_remote_code=True,
            attn_implementation="eager",
            return_dict_in_generate=True
        )
        self.model.eval()
        self.eos_token = self.tokenizer.eos_token if self.tokenizer.eos_token else "</s>"
        print(f" {model_path} Loading model Success")
        
    def analyze_attention(self, query, context_before, answer, context_after, layer_idx=-1):

        full_text = f"{context_before} {answer} {context_after} {query} {self.eos_token}"
        
        inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        answer_tokens = self.tokenizer.encode(answer, add_special_tokens=False)
        context_after_tokens = self.tokenizer.encode(context_after, add_special_tokens=False)
        query_tokens = self.tokenizer.encode(query, add_special_tokens=False)
        
        text_before_answer = context_before  
        tokens_before = len(self.tokenizer.encode(text_before_answer, add_special_tokens=False))
        
        answer_start = tokens_before
        answer_end = answer_start + len(answer_tokens)
        
        context_after_start = answer_end
        context_after_end = context_after_start + len(context_after_tokens)
        
        query_start = context_after_end
        query_end = query_start + len(query_tokens)
        
        eos_position = query_end  
        
        with torch.no_grad():

            position_ids = torch.arange(len(tokens), dtype=torch.long, device=self.model.device)
            position_ids = position_ids.unsqueeze(0)

            outputs = self.model(
                **inputs,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
                position_ids=position_ids
            )

            if outputs.attentions is None:
                raise ValueError("Model did not output attention weights. Please check model configuration.")

            if layer_idx == -1:
                attention_weights = outputs.attentions[-1][0]  # (num_heads, seq_len, seq_len)
            else:
                attention_weights = outputs.attentions[layer_idx][0]
            
            attention_weights = attention_weights.mean(dim=0)  # (seq_len, seq_len)
        
        def calculate_contribution(start_idx, end_idx):
            if end_idx <= start_idx:
                return 0.0
            slice_sum = float(attention_weights[:, start_idx:end_idx].sum())
            num_tokens = end_idx - start_idx
            return slice_sum / num_tokens

        attention_contributions = {
            'context_before': calculate_contribution(0, answer_start),
            'answer': calculate_contribution(answer_start, answer_end),
            'context_after': calculate_contribution(answer_end, query_start),
            'query': calculate_contribution(query_start, eos_position),
            'eos': float(attention_weights[:, eos_position].sum()) / 1
        }

        total_contribution = sum(attention_contributions.values())
        normalized_contributions = {k: v/total_contribution for k, v in attention_contributions.items()}
        
        detailed_info = {
            'attention_weights': attention_weights.cpu().numpy(),  # (seq_len, seq_len)
            'tokens': tokens,
            'positions': {
                'eos': 0,
                'context_before': (1, answer_start),
                'answer': (answer_start, answer_end),
                'context_after': (answer_end, query_start),
                'query': (query_start, query_end)
            }
        }
        
        return normalized_contributions, detailed_info

def visualize_attention(detailed_info, model_name, case_name, test_case_index):
    attention_weights = detailed_info['attention_weights']  # (seq_len, seq_len)
    tokens = detailed_info['tokens']
    
    df_attention = pd.DataFrame(attention_weights, index=tokens, columns=tokens)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_attention, cmap='viridis')
    plt.title(f'Attention Heatmap - {model_name} - {case_name} (Test Case {test_case_index})')
    plt.xlabel('Key Tokens')
    plt.ylabel('Query Tokens')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'attention_heatmap_{test_case_index}_{model_name}.png')
    plt.close()

def compare_models(test_cases):
    model_paths = [
        "/volume/demo/LLM/convert_model/mask/1b_0.15_v2/200b",
        "/volume/demo/LLM/convert_model/nomask/1b/200b"
        # add model aslo...
    ]
    
    analyzers = [AttentionAnalyzer(path) for path in model_paths]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n\n Testing {i}: {case['name']}")
        print("=" * 50)
        print(f"Query: {case['query']}")
        print(f"Answer: {case['answer']}")
        print("-" * 50)
        
        results = []
        detailed_results = []
        for j, analyzer in enumerate(analyzers):
            contributions, detailed_info = analyzer.analyze_attention(
                case['query'],
                case['context_before'],
                case['answer'],
                case['context_after']
            )
            results.append(contributions)
            detailed_results.append(detailed_info)
            
            model_name = "Mask" if j == 0 else "NoMask"
            print(f"\n{model_name}  attention contribute:")
            for part, score in contributions.items():
                print(f"{part}: {score:.3f}")
            
            visualize_attention(detailed_info, model_name, case['name'], i)

def run_experiments():
    test_cases = [
# {
# "name": "Marked Answer",
# "query": "Which ancient Greek physician is considered the 'Father of Medicine' and what ethical guidelines did he establish that doctors still follow today?",
# "context_before": "When examining the foundations of Western medicine:",
# "answer": "Hippocrates established medical ethics through the Hippocratic Oath, which emphasizes patient confidentiality and the principle of 'first, do no harm'",
# "context_after": ", revolutionizing medicine by separating it from religious and superstitious practices."
# }
# {
# "name":"text1",
# "query":"What is the most famous building on the Acropolis?",
# "context_before": "The Acropolis of Athens, perched above the city, is one of the most important archaeological sites in Greece. It contains several ancient buildings, ",
# "answer": "the most famous of which is the Parthenon.",
# "context_after": "The Acropolis was the center of religious and civic life in ancient Athens and has been a symbol of Greek culture for centuries. It remains a powerful reminder of ancient Greece’s artistic, political, and architectural achievements, attracting millions of visitors annually."
    
# },

# {
# "name":"text1",
# "query":"question:Who designed the Eiffel Tower?",
# "context_before": "In the heart of Paris, the Eiffel Tower stands tall, symbolizing both the city and the entire country.",
# "answer": "Designed by Gustave Eiffel",
# "context_after": ", it was completed in 1889 for the World’s Fair. Originally criticized for its unusual design, it has since become one of the most recognizable landmarks in the world. Tourists from all over the globe visit it every year, making it one of the most photographed monuments."
    
# }
# ,
# {
# "name":"text2",
# "query":"question:What was the purpose of the Great Wall of China?",
# "context_before": "The Great Wall of China, stretching over 13,000 miles, is one of the most impressive feats of ancient engineering.",
# "answer": "Built to protect Chinese states from invasions",
# "context_after": ", the wall took several dynasties over 2,000 years to complete. Its immense length and historical significance make it a popular tourist attraction today. The wall's construction involved countless workers, many of whom faced difficult conditions."
    
# }
# ,
# {
# "name":"text3",
# "query":"question:What famous equation did Albert Einstein create?",
# "context_before": "In the early 20th century, Albert Einstein introduced his theory of relativity, which changed the way we understand space, time, and gravity.",
# "answer": "His famous equation, E=mc²",
# "context_after": "shows the relationship between energy and mass. Einstein’s ideas revolutionized physics, and his work led to the development of technologies like GPS and nuclear energy. Despite facing initial skepticism, his theories were eventually proven through experiments and observations, earning him a Nobel Prize in Physics in 1921."
    
# }
{
"name": "text8",
"query": "question: What was the primary function of the Great Pyramids of Giza?",
"context_before": "The Great Pyramids of Giza, built during Egypt's Old Kingdom period, are massive stone structures that have endured for over 4,500 years.",
"answer": "The pyramids were built as elaborate tombs for the pharaohs, containing burial chambers, treasures, and provisions for the afterlife, while also serving as symbols of the pharaohs' power and religious significance",
"context_after": ", their construction remaining a remarkable feat of ancient engineering and architecture."
},
{
"name": "text9",
"query": "question: What was the purpose of Stonehenge?",
"context_before": "Stonehenge, located in Wiltshire, England, is a prehistoric monument of massive stones arranged in circular patterns.",
"answer": "Stonehenge served multiple purposes: it functioned as an ancient burial ground, a ceremonial site, a religious temple, and an astronomical calendar for tracking solar and lunar movements, while also serving as a place for ancestor worship and community gatherings",
"context_after": ", its exact construction methods still puzzling archaeologists today."
},
{
"name": "text10",
"query": "question: What was the main purpose of the Palace of Versailles?",
"context_before": "The Palace of Versailles, located near Paris, France, began as a hunting lodge before its massive expansion.",
"answer": "The Palace of Versailles served as the royal residence and seat of political power in France, housing the French royal court and government, demonstrating the absolute monarchy's power through its grandeur, hosting diplomatic events and ceremonies, while also serving as a symbol of French supremacy in art and architecture",
"context_after": ", until the French Revolution forced the royal family to leave in 1789."
}
## add test case also
    ]
    
    compare_models(test_cases)

if __name__ == "__main__":
    run_experiments()
