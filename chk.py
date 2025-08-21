from transformers import Seq2SeqTrainingArguments
import inspect

print("Class location:", inspect.getfile(Seq2SeqTrainingArguments))
print("\nConstructor:\n")
print(inspect.signature(Seq2SeqTrainingArguments.__init__))
