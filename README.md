Overview

Political discourse is rich in ambiguity—a feature that often serves strategic purposes. In high-stakes settings such as televised presidential debates or interviews, politicians frequently employ evasive communication strategies that leave audiences with multiple interpretations of what was actually said and misguided impressions on whether the information requested was conveyed. This phenomenon, known as equivocation or evasion in academic literature, is well-studied in political science but has received limited attention in computational linguistics. Bull (2003) report that politicians gave clear responses to only 39-46% of questions during televised interviews, while non-politicians had a significantly higher 70-89% reply rate. This stark contrast highlights the strategic nature of political communication and the need for automated tools to analyze response clarity at scale.

Our proposed CLARITY shared task aims to address this gap by introducing a computational approach to detecting and classifying response ambiguity in political discourse. Building on well grounded theory on equivocation and leveraging recent advancements in language modeling, we propose a task that challenges participants to automatically classify the clarity of responses in question/answer (QA) pairs extracted from presidential interviews.

What makes this task particularly compelling is leveraging a novel, two-level taxonomy approach, derived from our paper (Thomas et al., 2024), presented in EMNLP 2024:

A high-level clarity/ambiguity classification.
A fine-grained classification of 9 evasion techniques stemming from political discourse.
This hierarchical approach not only provides a deeper understanding of political discourse but also, as our preliminary experiments show, can lead to improved classification performance when the two levels are used in conjunction.

The CLARITY task will attract researchers from diverse communities, including NLP researchers interested in discourse analysis, semantic understanding, and reasoning over long contexts, fact-checking seeking to detect if an answer is factual but irrelevant, and other downstream NLP tasks such as question answering and dialogue systems. Moreover, by providing a standardized dataset and evaluation framework, CLARITY will facilitate political speech discourse analysis at scale, allowing for comparisons across politicians, time periods, and contexts. Contributing to the development of more transparent and accountable political communication and provide insights to media analysts examining patterns in political interviews and press conferences.

Can your system unmask a seasoned politician’s dodge?


Tasks & Evaluation

Task 1 - Clarity-level Classification
Given a question and an answer, classify the answer as Clear Reply, Ambiguous or Clear Non-Reply.


Task 2 - Evasion-level Classification
Given a question and an answer, classify the answer into one of the 9 evasion techniques.


Evaluation

Both tasks will be evaluated using macro F1-score, ensuring balanced performance across all classes. Evaluation will be conducted on both the official test set and a held-out private evaluation set to ensure robust performance assessment.
