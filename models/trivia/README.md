# Extracting samples from the Trivia dataset

These samples were generated using the Llama2-13b for the TriviaQA dataset.

To extract the samples from the .pkl object into a .json file use the `read_model_samples.py`.

To run the script use the following commands:
```
python read_model_samples.py <path_to_file>.pkl
```
You can also send the command with arguments for a number of samples and generation per sample:
- `-num` or `--numSamples` to indicate the number of samples to extract;
- `-numGen` or `--numGenerations` to indicate the number of generations per sample.

Example: to run with the file `./llama2-13b_triviaqa_new_10_0.4_20/0.pkl`, and extract 2000 examples of the dataset with 10 generations each:

```
python read_model_samples.py ./llama2-13b_triviaqa_new_10_0.4_20/0.pkl -num 2000 -numGen 10
```
