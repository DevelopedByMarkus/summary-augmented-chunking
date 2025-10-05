## Description of the Config files

To have a more efficient way of handling the setting of the retriever part, the setting of them are part of a json file.
This also helps reproducibility.

```
{
  "strategy_type": "baseline" (dense retrieval only) or "hybrid" (dense and sparse retrieval),
  "chunking_strategy": {
    "strategy_name": "naive" (fixed-size chunking), "rcts" (recursive chunking with RCTS library), "summary_naive" (SAC with fixed-size chunking), "summary_rcts" (SAC with recursive chunking with RCTS library),
    "chunk_size": 500 (ceiling of chunk size in characters),
    "chunk_overlap_ratio": 0.0 (overlap ratio between chunks, e.g. 0.2 means 20% overlap),
    "summary_model": { (model to generate summaries for SAC)
      "company": "openai",
      "model": "gpt-4o-mini"
    },
    "summary_prompt_template": "XYZ" (prompt template to generate summaries, see below for details),
    "prompt_target_char_length": 300 (target character length for the summary in the prompt),
    "summary_truncation_length": 350 (max length of the document to be summarized, longer documents will be first rerun and if necessary truncated)
  },
  "embedding_model": { (model to generate embeddings of chunks and query)
    "company": "huggingface",
    "model": "thenlper/gte-large"
  },
  "embedding_top_k": 128 (number of top-k chunks to retrieve from embedding search),
  "bm25_top_k": 128 (number of top-k chunks to retrieve from BM25 search, only for hybrid method),
  "fusion_top_k": 64 (number of top-k chunks to fuse from embedding and BM25 search, only for hybrid method)
  "fusion_weight": 1.0 (weight for embedding score in fusion, only for hybrid method. 1 means dense retrieval only and 0 means sparse retrieval only),
  "rerank_model": null (model that reranks the top-k retrieved chunks),
  "rerank_top_k": [1, 2, 4, 8, 16, 32, 64] (final_top_k_values. Must always be set, even if no reranker is used. Must be a list of integers),
  "token_limit": null (Unused at the moment)
}
```

### Summary Prompt Templates

1. Non-expert Prompt Template for summaries:
```
"System: You are an expert legal document summarizer. User: Summarize the following legal document text. Focus on extracting the most important entities, core purpose, and key legal topics. The summary must be concise, maximum {target_char_length} characters long, and optimized for providing context to smaller text chunks. Output only the summary text. Document:\n{document_content}"
```

2. Expert Prompt Template for summaries:
```
"System: You are a legal summarization expert.\nUser: Your task is to generate a highly distinct, structured summary of the provided legal document. The primary goal is to extract the unique identifiers that differentiate this document from others of the same type. This summary will be used as context to smaller text chunks for a retrieval system.\n\nFollow this two-step process:\n- First, internally identify the document type from the following\noptions: Non-Disclosure Agreement (NDA), Privacy Policy, or Other.\n- Second, generate the summary based on the specific template corresponding to the identified document type.\n\nDocument type Non-Disclosure Agreement (NDA): An NDA is a legally binding contract between specific parties that outlines confidential information to be kept secret.\nIf the document is an NDA, your summary should align with the following\ntemplate:\n- Definition of Confidential Information, specifying what types of information are considered confidential, e.g. such as: Technical data, Business plans, Customer lists, Trade secrets, Financial information\n- Parties to the Agreement identifying the disclosing party and the receiving party (or both, if mutual NDA), e.g. such as: Full legal names, Affiliates or representatives covered, Roles of each party\n- Obligations of the receiving party outlining what the receiving party is required to do, e.g. such as: Keeping the information secret, Limiting disclosure to authorized personnel, Using the information only for specified purposes\n- Exclusions from confidentiality describing information that is not protected under the NDA, such as: Information already known to the receiving party, Publicly available information, Information disclosed by third parties lawfully, Independently developed information\n- Specifying any exceptions where disclosure is allowed, such as: To employees or advisors under similar obligations, If required by law or court order (with notice to the disclosing party)\n- Term and Duration, defining how long the confidentiality obligation\nlasts: Often includes both the duration of the agreement and the period during which information remains protected (e.g., \"3 years after\ntermination\")\n- Purpose of Disclosure (Use Limitation), stating the specific reason the information is being shared (e.g., for evaluating a partnership, conducting due diligence, etc.) and prohibits other uses.\n- Remedies for Breach, detailing the consequences of violating the NDA, which may include: Injunctive relief (court orders to stop disclosure), Damages, Legal fees\n- Governing Law and Jurisdiction, identifying which country/state’s laws apply and where disputes will be settled.\n- Miscellaneous Clauses (Boilerplate), may include: No license granted, Entire agreement clause, Amendment process, Counterparts and signatures\n\nDocument type Privacy Policy: A privacy policy is issued by a private or public entity to inform users how their personal data is processed (e.g., collected, used, shared, stored).\nIf the document is a privacy policy, your summary should align with the following template:\n- Personal Data Collected and Processed, specifying what categories of personal data are collected and how. This may include: Name and surname, Contact information, Financial details, Device and browser data, Location information, Inferred preferences or behaviors\n- Identity and Contact Details of the Controller, identifying the organisation responsible for the processing. May include: Full legal name of the controller, Contact email or phone number, Details of any representative (if applicable)\n- Purposes of Processing, outlining why the personal data is collected and how it will be used. Examples include: Service provision and operation, Personalisation of content or features, Marketing and advertising, Analytics and performance monitoring, Payment processing\n- Legal Basis for Processing, specifying the lawful grounds relied upon.\nThese are: Consent of the data subject, Performance of a contract, Compliance with a legal obligation, Protection of vital interests, Task carried out in the public interest, Legitimate interests of the controller or third party\n- Recipients of the Data, listing who may receive the data, including:\nService providers and processors, Business partners, Public authorities (where legally required), Affiliates and subsidiaries\n- International Data Transfers, describing whether personal data is transferred outside the jurisdiction and, if so: Destination countries, Safeguards applied (e.g., Standard Contractual Clauses, adequacy decisions)\n- Data Retention, defining how long the personal data will be stored, or the criteria for determining the period. May include: Fixed retention periods, Purpose-based retention (e.g., \"as long as necessary to provide the service\"), Archiving or deletion policies\n- Data Subject Rights, explaining individuals’ rights under data protection law, including: Right to access personal data, Right to rectify inaccuracies, Right to erasure (\"right to be forgotten\"), Right to restrict or object to processing, Right to data portability\n- Right to Lodge a Complaint, providing information on: The data subject’s right to contact a supervisory authority, Name or link to the competent authority\n- Automated Decision-Making, disclosing whether such processing occurs and, if so: The logic involved, Potential significance of the decisions, Expected consequences for the data subject\n\nOther document type: If the document does not match the types above, summarize the following general legal document in a structured, concise way. Identify for your summary the important entities, core purpose, and other unique identifiers that differentiate this document from others of the same type.\n\nGeneral Rules:\n- The summary must be concise and under {target_char_length} characters.\n- IMPORTANT: The summary must contain the document type, e.g. Privacy Policy, Non-Disclosure Agreement (NDA), etc.\n- Ignore every field in the template where the information is not present in the document\n- Prioritize extracting the most critical identifiers, such as parties, dates, and the specific subject matter.\n- Output ONLY the final summary text!\n\nHere is the document you should summarize:\n{document_content}\n"
```
