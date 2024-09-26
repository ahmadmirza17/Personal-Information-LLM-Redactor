# Question 1

## Part A: Challenges of Low-Resource Languages for NLP and LLMs

Based on the paper, low-resource languages like Singlish pose several challenges for natural language processing (NLP) and large language models (LLMs):

1. **Unique linguistic features**: As mentioned in Section 2, Singlish has acquired its own unique phonology, lexicon and syntax. This makes it significantly different from standard English, which most NLP models and LLMs are trained on.

2. **Lack of training data**: Low-resource languages, by definition, have limited amounts of high-quality labelled data available for training models. This makes it difficult to create robust NLP systems for these languages.

3. **Code-mixing**: The paper notes that in Singlish, "different languages are often combined within single utterances." This mixing of languages within sentences poses challenges for models trained on monolingual corpora.

4. **Context-specific terminology**: Low-resource languages often have content-specific terminology that may not be understood by models trained on more general language data. For example, the paper mentions terms like "ceca" and "sinkie" that have specific meanings in the Singaporean context, which are derogatory synecdoches.

5. **Evolving language**: The paper notes, "model may become less effective as the linguistic features of Singlish inevitably change over time", that new vocabulary will emerge in the online domain for Singlish. Low-resource languages, especially those used primarily in informal settings, can evolve rapidly, making it challenging for models to stay up-to-date.

### Tokenisation Challenges:

Tokenisation is a crucial step in NLP where text is broken down into smaller units (tokens) for processing. Low-resource languages pose challenges for tokenisation because:

1. **Unique character sets**: Low-resource languages may use character sets or scripts that are not well-represented in standard tokenisers.

2. **Word boundaries**: Languages with different writing systems or grammatical structures may have different conventions for word boundaries, which can affect tokenisation.

3. **Subword tokenisation**: Many modern NLP models use subword tokenisation to handle out-of-vocabulary words. However, these tokenisers are typically trained on high-resource languages and may not effectively capture the morphology of low-resource languages.

4. **Multilingual tokens**: In the case of code-mixed languages like Singlish, a single sentence might contain tokens from multiple languages, making it challenging to apply a single tokenisation strategy.

5. **Informal and evolving language**: Online and informal usage of low-resource languages (as seen in the Singlish examples in the paper) often includes non-standard spellings, abbreviations, and neologisms that may not be effectively captured by standard tokenisers.

These tokenisation challenges can lead to poor model performance, as the input representation may not accurately capture the linguistic nuances of the low-resource language. This, in turn, affects all downstream NLP tasks, including the content moderation task addressed by LionGuard.

## Part B: Implications for Safety Guardrails in Low-Resource Language Countries

This paper has several important implications for the development of safety guardrails for LLMs in countries with low-resource languages:

1. **Necessity of localisation**: The paper demonstrates that generalist moderation classifiers (like OpenAI's Moderation API, Jigsaw's Perspective API, and Meta's LlamaGuard) perform poorly on localised content. This implies that effective safety guardrails for LLMs need to be specifically adapted to local languages and contexts.

2. **Improved detection of local unsafe content**: LionGuard significantly outperformed existing moderation APIs in detecting unsafe content in Singlish. This suggests that localised safety guardrails could be much more effective at identifying and filtering out harmful content in low-resource languages.

3. **Scalable approach for creating datasets**: The paper presents a method for creating labelled datasets using automated LLM labelling, which could be adapted to other low-resource languages. This approach could help overcome the lack of training data typically associated with low-resource languages.

4. **Context-specific safety categories**: The paper developed a safety risk taxonomy aligned with local context, including consideration of local legislation. This implies that safety guardrails for LLMs may need to be tailored to specific cultural, legal, and social norms of different countries.

5. **Potential for transfer learning**: While LionGuard was specifically trained for Singlish, the approach could potentially be used to create similar models for other low-resource languages, or to fine-tune existing models for better performance in specific contexts.

6. **Importance of understanding local slang and references**: The error analysis showed that LionGuard was able to understand local slang and references that other models missed. This highlights the need for safety guardrails to be trained on or adapted to local linguistic nuances.

7. **Need for continual updating**: The paper notes that language evolves rapidly, especially in online contexts. This suggests that safety guardrails for low-resource languages may need frequent updating to remain effective.

8. **Potential for improving general models**: The paper suggests that localised models like LionGuard could be used to generate adversarial data to improve the performance of generalist moderation classifiers on low-resource languages.

9. **Ethical considerations**: The paper raises important ethical considerations about content moderation in different cultural contexts, implying that the development of safety guardrails needs to carefully consider local ethical standards and potential biases.

10. **Cost-effective solution**: The approach presented in the paper, using automated labelling and relatively simple classifiers, could provide a cost-effective way for countries with low-resource languages to develop effective safety guardrails for LLMs.

These implications suggest that developing effective safety guardrails for LLMs in countries with low-resource languages will require a combination of localised data collection, context-specific categorisation of unsafe content, and potentially the development of specialised models or fine-tuning approaches. The method presented in this paper offers a potential roadmap for achieving this in a scalable and relatively cost-effective manner.

## Part C: Evaluation Metrics for LionGuard

### Why PR-AUC was chosen as the evaluation metric

PR-AUC (Precision-Recall Area Under Curve) was chosen as the evaluation metric for LionGuard over other performance metrics for several reasons:

1. **Imbalanced Dataset**: The authors note that "Due to the heavily imbalanced
dataset, we chose the Precision-Recall AUC (PRAUC) as our evaluation metric...". PR-AUC is particularly useful for imbalanced datasets where the positive class (unsafe content) is relatively rare.

2. **Performance Across Thresholds**: PR-AUC "can better represent the classifier's ability to detect unsafe content across all score thresholds". This allows for a more comprehensive evaluation of the model's performance without being tied to a specific classification threshold.

3. **Consistency with Industry Standards**: The authors mention that PR-AUC "was also used by OpenAI (Markov et al., 2023) and LlamaGuard (Inan et al., 2023) in their evaluations". Using the same metric allows for easier comparison with other state-of-the-art models in content moderation.

4. **Focus on Positive Class**: In content moderation, correctly identifying unsafe content (the positive class) is typically more important than correctly identifying safe content. PR-AUC focuses on the model's performance on the positive class.

### Situations where F1-score would be preferable

F1-score might be preferable in the following situations:

1. **Balanced Datasets**: When the classes are more evenly distributed, F1-score can provide a good balance between precision and recall.

2. **Single Threshold Evaluation**: If you need to evaluate the model's performance at a specific decision threshold, F1-score provides a single value that balances precision and recall.

3. **Equal Importance of Classes**: In cases where false positives and false negatives are equally important, F1-score treats both types of errors with equal weight.

4. **Interpretability**: F1-score is often easier to interpret and explain to non-technical stakeholders, as it provides a single value between 0 and 1.

5. **Resource Constraints**: Calculating F1-score is computationally less expensive than PR-AUC, which might be relevant in resource-constrained environments.

6. **Binary Classification Focus**: F1-score is particularly suited for binary classification problems, whereas PR-AUC can be more complex to interpret for multi-class problems.

The choice between PR-AUC and F1-score (or any other metric) ultimately depends on the specific requirements of the task, the nature of the dataset, and the goals of the evaluation.

## Part D: Weaknesses in LionGuard's Methodology

Despite its innovative approach, LionGuard's methodology has several potential weaknesses:

1. **Limited Dataset Scope**: The dataset is primarily sourced from two online forums (HardwareZone and Reddit), which may not fully represent the diversity of Singlish usage across different contexts and demographics.

2. **Reliance on LLM Labelling**: The automated labelling process relies on existing LLMs, which may introduce biases or errors inherent in these models. The authors acknowledge that "we cannot completely guarantee the accuracy of our LLM labels".

3. **Consensus Approach Limitations**: Using a consensus approach for LLM labelling might lead to the exclusion of more nuanced or ambiguous cases, potentially reducing the diversity of the training data.

4. **Temporal Limitations**: As noted in the limitations section, the dataset is "a static, albeit up-to-date, snapshot of the online discourse in Singapore". This may not capture the rapidly evolving nature of online language.

5. **Potential Overfitting**: The model's strong performance on Singlish might come at the cost of reduced generalisability to other variants of English or other languages.

6. **Limited Human Validation**: While there was some human validation of the labels, the scale was relatively small (200 expert-labelled texts and about 12,000 crowd-sourced labelled texts) compared to the full dataset of 138,000 texts.

7. **Ethical Considerations**: The paper briefly mentions ethical considerations but does not deeply explore the potential biases or societal impacts of such a moderation system.

8. **Lack of Extensive Hyperparameter Tuning**: The authors admit that they "did not perform extensive experiments on varying model hyperparameters", which might mean the model's performance could potentially be improved further.

9. **Limited Exploration of Model Architectures**: The focus on relatively simple classifier models (ridge regression, XGBoost, and a simple neural network) might overlook potential benefits of more complex architectures.

10. **Difficulty in Reproducibility**: The reliance on proprietary LLMs for labelling (GPT-3.5, Claude 2, PaLM 2) may make it challenging for other researchers to exactly reproduce the results or apply the method to other languages.

11. **Potential for Concept Drift**: As online language evolves rapidly, the model may become less effective over time without regular updates, which could be resource-intensive.

12. **Limited Multi-label Performance**: While the model performs well on binary classification, its performance on multi-label classification (especially for categories with few positive examples) is less impressive.

These weaknesses suggest areas for potential improvement in future iterations of LionGuard or similar localised content moderation systems.

# Question 2: Evaluating LionGuard

## Data Analysis
Based on the Jupyter notebook, the relevant columns for evaluating LionGuard's performance are:

1. 'Hateful Score': LionGuard's output (1 = hateful, -1 = non-hateful)
2. 'annotation_selected': Ground truth label
3. Categorical columns: 't_function', 't_direction', 'p_target' (provide context about test cases)

## Key Findings

### 1. Overall Performance
- Hateful content: Mean score 0.95 (std: 0.33)
- Non-hateful content: Perfect mean score 1.00 (std: 0.00)

### 2. Performance by Functionality (t_function)
#### Excellent Performance (mean = 1.00 or close)
- derog_impl_h (implicit derogation)
- derog_neg_emote_h (negative emotions)
- phrase_opinion_h (hateful opinions)
- phrase_question_h (hateful questions)
- profanity_h (profanity)
- ref_subs_clause_h and ref_subs_sent_h (referential hate speech)
- threat_dir_h and threat_norm_h (threats)

#### Lower Performance
- derog_neg_attrib_h (negative attributes): mean = 0.83, std = 0.56
- spell_leet_h (leet speak): mean = 0.65, std = 0.77
- spell_space_add_h (added spaces): mean = 0.57, std = 0.83

### 3. Performance by Directionality (t_direction)
- General hate speech: mean = 0.97, std = 0.25
- Directed hate speech: mean = 0.90, std = 0.43

### 4. Performance by Target Group (p_target)
- Consistent performance across groups (mean scores 0.92 to 0.99)
- Lower performance: Chinese, Homosexual, Seniors targets (mean = 0.92)
- Highest performance: Christian and physically disabled targets (mean = 0.99)

## Key Strengths
1. Excellent performance on non-hateful content (low false positive rate)
2. Strong performance across various hate speech functions, especially explicit forms
3. Consistent performance across different target groups (low demographic bias)

## Areas for Improvement
1. Handling of spelling variations and leetspeak
2. Detection of subtle hate speech using negative attributes
3. Performance on directed hate speech compared to general hate speech

## Conclusion
LionGuard shows strong overall performance on the SGHateCheck dataset, excelling at identifying non-hateful content and most forms of explicit hate speech. Main areas for improvement include detecting subtle or obfuscated forms of hate speech, such as those using leetspeak or creative spelling variations. These challenges align with common issues faced by hate speech detection models in dealing with evolving online language and intentional obfuscation tactics.

# Question 3: Improving LionGuard

Based on the insights from the LionGuard and SGHateCheck papers, as well as broader trends in hate speech detection, here are several strategies to improve LionGuard:

## 1. Enhanced Multilingual and Code-Switching Capabilities

LionGuard could be improved by enhancing its ability to handle multiple languages and code-switching, which is common in Singapore's linguistic landscape.

### Proposed Improvement:
Implement a multi-task learning approach that jointly learns language identification and hate speech detection.

```python
class ImprovedLionGuard(nn.Module):
    def __init__(self, num_languages, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained("xlm-roberta-base")
        self.language_classifier = nn.Linear(768, num_languages)
        self.hate_classifier = nn.Linear(768, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        language_logits = self.language_classifier(pooled_output)
        hate_logits = self.hate_classifier(pooled_output)
        return language_logits, hate_logits
```

## 2. Contextual Embedding Fine-tuning

Fine-tune the underlying language model on a large corpus of Singaporean social media text to better capture local linguistic nuances.

### Proposed Improvement:

Use masked language modeling (MLM) on a large corpus of Singaporean text before fine-tuning for hate speech detection.

```python
# Pseudo-code for MLM fine-tuning
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")

# Load Singaporean text corpus
sg_corpus = load_singapore_corpus()

# Fine-tune using MLM
model = fine_tune_mlm(model, tokenizer, sg_corpus)

# Use this fine-tuned model as the base for hate speech detection
```

## 3. Adversarial Training

Implement adversarial training to make the model more robust against spelling variations and intentional obfuscation.

### Proposed Improvement:

Generate adversarial examples by applying random transformations to input text and train the model to be invariant to these changes.

```python
def generate_adversarial(text):
    # Apply random transformations (e.g., character swaps, deletions)
    return transformed_text

def adversarial_loss(model, inputs, labels):
    adv_inputs = generate_adversarial(inputs)
    original_pred = model(inputs)
    adv_pred = model(adv_inputs)
    return nn.functional.kl_div(original_pred, adv_pred)

# Include adversarial loss in training
```

## 4. Hierarchical Attention Mechanism

Implement a hierarchical attention mechanism to better capture the context and improve performance on directed hate speech.

### Proposed Improvement:

Add a hierarchical structure to the model that first attends to individual words, then to sentences, and finally to the entire document.

```python
class HierarchicalAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.word_attention = Attention(...)
        self.sentence_attention = Attention(...)
        self.document_attention = Attention(...)
    
    def forward(self, inputs):
        word_ctx = self.word_attention(inputs)
        sentence_ctx = self.sentence_attention(word_ctx)
        doc_ctx = self.document_attention(sentence_ctx)
        return doc_ctx
```

## 5. Explainable AI Integration

Incorporate explainable AI techniques to provide interpretability for the model's decisions, which can help in refining the model and building trust.

### Proposed Improvement:

Integrate LIME (Local Interpretable Model-agnostic Explanations) or SHAP (SHapley Additive exPlanations) into the model pipeline.

```python
from lime.lime_text import LimeTextExplainer

def explain_prediction(model, text):
    explainer = LimeTextExplainer(class_names=['non-hate', 'hate'])
    exp = explainer.explain_instance(text, model.predict_proba)
    return exp
```

## 6. Dynamic Updating Mechanism

Implement a mechanism for continuous learning to adapt to evolving language patterns and new forms of hate speech.

### Proposed Improvement:

Set up a pipeline for periodic retraining with newly collected and human-verified data.

```python
def update_model(model, new_data):
    # Fine-tune model on new data
    model = fine_tune(model, new_data)
    return model

# Pseudo-code for periodic updating
while True:
    new_data = collect_new_data()
    verified_data = human_verification(new_data)
    model = update_model(model, verified_data)
    time.sleep(UPDATE_INTERVAL)
```

These improvements address the main challenges identified in the SGHateCheck evaluation, particularly in handling linguistic variations, context-dependent hate speech, and evolving language patterns. They also align with broader trends in NLP for more robust, adaptable, and interpretable models.