# PreProcessing Module

**Goal**: Module that takes in a diverse array of data/docs and forms premise / verified ground truth (VGT) answer response pairs

## Needs:
* Ability to flexibly define the targets for premise and VGT
* Ability to define primitive types for both sides (i.e. VGTA might be one, might be a list, might be a dict with multiple elements)
    * Possible integration case for dspy or pydantic
* Ability to filter out documents / rows that are insufficient for the reasoning task
    * See example prompt below
    * Filter should be an optional bool and will be an optional parameter in the document preprocessing
    * Should be some default filters such as depth of reasoning, signal to noise ratio, etc
    * Maybe filter prompt can be dynamically generated via the targets i.e. an optional hard coupling where filter prompt is generated based on the targets
    * We might consider a post-filtering API too. One such filter might be “can an LLM get the right answer to this premise/VGTA without any CoT whatsoever” (minimum difficulty threshold, would also require access to the evaluator module)
* Input data formats might include .txt, HF datasets, csv, json, etc….

### Filter Prompt Example
```
You are an expert in filtering and evaluating legal case text and legal reasoning. Your job is to evaluate a given case and determine whether it meets the following criteria:
**Depth of Reasoning:** The case should include detailed reasoning. If the case apepars too simple, mark it as "Too Simple".
**Unambiguous Decision:** The case should have a clear and unambiguous conclusion, decision, or holding. If the conclusion is ambiguous, mark it as "Ambiguous Decision".
**Facts Presented:** Case should present facts that are important to the ultimate decision of the court. The factual background should be sufficient for a reader to learn about the case or controversy between the parties and why the facts are legally relevant. If the case does not contain enough facts or the facts are not relevant to the legal reasoning, mark it as "Insufficient Facts".
**Legal Issues:** The case should explicitly delineate the legal issues involved. If the legal issues are not clear, mark it as "Unclear Legal Issues".

For each case, answer with one of the following labels:
- "pass" (if the case meets all criteria)
- "Too Simple"
- "Ambiguous Decision"
- "Insufficient Facts"
- "Unclear Legal Issues"
```
### Premise/VGT Extraction Prompt Example: 
```
 **Facts of the Case:** Provide the facts of the case, which refer to the events that led to the legal dispute. The facts are objective and should not include any reasoning or conclusions. They refer to the parties and the case or controversy between them. The facts should be fully encompassing and detailed, giving enough context to fully understand the case background.
**Legal Issues:** Identify the legal issue(s) involved in the case. Legal issues are the questions of law that are presented for determination by the court. They are the legal questions that the court must answer in order to resolve the dispute between the parties. The legal issues each must be no more than one sentence. They should be in an array format.
**Conclusion:** Provide the conclusion of the case, which is the final decision or holding of the court. The conclusion must be clear, unambiguous, and no more than one sentence. It does not include any reasoning or analysis.
**Reasons:** Provide the court's rationale for its decision as an array of single sentences. Each sentence represents a distinct reason given. Avoid summarizing multiple reasons within a single sentence.
```