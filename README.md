###### Chain-of-Thought prompting for an SARCASM-aware support chatbot

In this repository there is the experiment with an open-source LLM running locally which acts as a chatbot for customer support especially for delayed deliveries. The so called customers in this setting express frustration either directly or through sarcasm. So, the experiment tests whether Chain-Of-Thought (COT) prompting helps the model to recognise a customer's emotion and true intent, and respond appropriately.



###### The Setup

|Model|Qwen3 8B|
|-|-|
|Runtime|[Ollama](https://ollama.com), running locally|
|Method|Chain-Of-Thought prompting (COT prompting)|
|Settings|Temperature of 0, fixed seed of 42, built-in thinking mode switched off|

The model's built-in thinking mode is switched off so that the only reasoning it does, comes from the COT prompting.



###### Conditions

The same (made-up) customer messages are run by three different conditions. All three share the same description of the company and of what the chatbot can and cannot do.

|Condition|Description|
|-|-|
|'baseline'|Plain support-agent prompt|
|'cot'|Adds reasoning steps before the reply. Reasoning is based on: emotion, sarcasm, need, strategy|
|'cot\_v2'|cot\_v2 extends the cot prompt with three extra reasoning steps: literal, situation, and meaning. Also adds four rules and one worked example.|

cot\_v2's four rules and worked example:

4 rules:

* The reply must follow explicitly from reasoning
* If situation is unclear, ask another question before assuming anything
* Never repeat an offer made to the customer, which the customer already rejected earlier
* Do not promise anything to the customer, which the chatbot cannot do by itself

Worked example with a sarcastic message with full reasoning and expected reply.

Note: The worked example in cot\_v2 is not one of the test messages.



###### Test cases

There are 11 test messages, defined in the file cot\_experiment.py:

* Scenario 1 - direct frustration: 2 messages
* Scenario 2 - sarcasm: 3 messages
* Control: 1 sincere positive message, to check for over-detection of sarcasm
* Stress tests: an ambiguous message, a demand outside the chatbot's authority, a request for information it does not have, a prompt injection, and a three-turn conversation with escalating anger

###### 

###### How to run

1. Install [Ollama](https://ollama.com) and download the model:
   ollama pull qwen3:8b
2. Install the Python dependency:
   pip install ollama
3. Run the experiment:
   python cot\_experiment.py



After some minutes the test ends and it writes two files to a results folder: a '.json' file with the raw data and a '.md' file that shows the reasoning and actual reply of each condition per test case.



###### Main observations

* The baseline takes sarcasm with positive wording literally and thanks the customer for the "kind words". Both COT prompts respond to the complaint instead.
* The baseline claims things that conflict with its instructions, such as "your package is in transit". The COT prompts do not go past their stated limits.
* cot\_v2 asks a clarifying question for the ambiguous message and no longer labels a direct complaint as sarcasm.
* cot\_v2 misses sarcasm that cot detected, and it gives in to the prompt injection ("I can approve your refund"), which the baseline and cot both refuse.
* The reasoning steps do not always constrain each other or the reply. The model can describe the customer as disappointed while labelling the message as not sarcastic. (The reasoning steps are not always consistent with each other. For one sarcastic message, the model correctly states that the customer is disappointed about a late parcel, but in the same output it labels the situation as "unclear" and the message as not sarcastic)



###### Limitations

* The test set is small (only 11 cases) and each condition was run once, so the results are merely observations.
* The test messages are hand-written and English only.
* The chatbot has no access to any order data (ofcourse) and cannot literally hand over to a human agent; the handover is only described in its replies (as that's what should happen for a real company).



###### Files

|File|Content|
|-|-|
|'cot\_experiment.py'|Prompts, test cases and the code that runs the experiment|
|'requirements.txt'|Python dependency ('ollama')|
|'results'|Output of the experiment run|



