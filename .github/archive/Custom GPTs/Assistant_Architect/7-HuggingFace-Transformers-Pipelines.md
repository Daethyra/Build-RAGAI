# HuggingFace - Transformers Library - Main Classes - Pipelines

[Read More](https://huggingface.co/docs/transformers/main_classes/pipelines)


The pipelines are a great and easy way to use models for inference. These pipelines are objects that abstract most of the complex code from the library, offering a simple API dedicated to several tasks, including Named Entity Recognition, Masked Language Modeling, Sentiment Analysis, Feature Extraction and Question Answering. See the task summary for examples of use.

### There are two categories of pipeline abstractions to be aware about:

The pipeline() which is the most powerful object encapsulating all other pipelines.
Task-specific pipelines are available for audio, computer vision, natural language processing, and multimodal tasks.
The pipeline abstraction

The pipeline abstraction is a wrapper around all the other available pipelines. It is instantiated as any other pipeline but can provide additional quality of life.

Simple call on one item:

Copied
>>> pipe = pipeline("text-classification")
>>> pipe("This restaurant is awesome")
[{'label': 'POSITIVE', 'score': 0.9998743534088135}]

If you want to use a specific model from the hub you can ignore the task if the model on the hub already defines it:

Copied
>>> pipe = pipeline(model="roberta-large-mnli")
>>> pipe("This restaurant is awesome")
[{'label': 'NEUTRAL', 'score': 0.7313136458396912}]

To call a pipeline on many items, you can call it with a list.

Copied
>>> pipe = pipeline("text-classification")
>>> pipe(["This restaurant is awesome", "This restaurant is awful"])
[{'label': 'POSITIVE', 'score': 0.9998743534088135},

To iterate over full datasets it is recommended to use a dataset directly. This means you don’t need to allocate the whole dataset at once, nor do you need to do batching yourself. This should work just as fast as custom loops on GPU. If it doesn’t don’t hesitate to create an issue.

Copied
import datasets
from transformers import pipeline
from transformers.pipelines.pt\_utils import KeyDataset
from tqdm.auto import tqdm

pipe = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h", device=0)
dataset = datasets.load\_dataset("superb", name="asr", split="test")

# KeyDataset (only \*pt\*) will simply return the item in the dict returned by the dataset item
for out in tqdm(pipe(KeyDataset(dataset, "file"))):
 print(out)
 # {"text": "NUMBER TEN FRESH NELLY IS WAITING ON YOU GOOD NIGHT HUSBAND"}
 # {"text": ....}
 # ....

For ease of use, a generator is also possible:

Copied
from transformers import pipeline

pipe = pipeline("text-classification")

def data():
 while True:
 # This could come from a dataset, a database, a queue or HTTP request
 # in a server
 # Caveat: because this is iterative, you cannot use `num\_workers > 1` variable
 # to use multiple threads to preprocess data. You can still have 1 thread that
 # does the preprocessing while the main runs the big inference
 yield "This is a test"

for out in pipe(data()):
 # {"text": "NUMBER TEN FRESH NELLY IS WAITING ON YOU GOOD NIGHT HUSBAND"}
 # {"text": ....}
 # ....
transformers.pipeline
<
source
>
( task: str = Nonemodel: typing.Union[str, ForwardRef('PreTrainedModel'), ForwardRef('TFPreTrainedModel'), NoneType] = Noneconfig: typing.Union[str, transformers.configuration\_utils.PretrainedConfig, NoneType] = Nonetokenizer: typing.Union[str, transformers.tokenization\_utils.PreTrainedTokenizer, ForwardRef('PreTrainedTokenizerFast'), NoneType] = Nonefeature\_extractor: typing.Union[str, ForwardRef('SequenceFeatureExtractor'), NoneType] = Noneimage\_processor: typing.Union[str, transformers.image\_processing\_utils.BaseImageProcessor, NoneType] = Noneframework: typing.Optional[str] = Nonerevision: typing.Optional[str] = Noneuse\_fast: bool = Truetoken: typing.Union[str, bool, NoneType] = Nonedevice: typing.Union[int, str, ForwardRef('torch.device'), NoneType] = Nonedevice\_map = Nonetorch\_dtype = Nonetrust\_remote\_code: typing.Optional[bool] = Nonemodel\_kwargs: typing.Dict[str, typing.Any] = Nonepipeline\_class: typing.Optional[typing.Any] = None\*\*kwargs ) → Pipeline

Expand 15 parameters

Parameters

task (str) — The task defining which pipeline will be returned. Currently accepted tasks are:

"audio-classification": will return a AudioClassificationPipeline.
"automatic-speech-recognition": will return a AutomaticSpeechRecognitionPipeline.
"conversational": will return a ConversationalPipeline.
"depth-estimation": will return a DepthEstimationPipeline.
"document-question-answering": will return a DocumentQuestionAnsweringPipeline.
"feature-extraction": will return a FeatureExtractionPipeline.
"fill-mask": will return a FillMaskPipeline:.
"image-classification": will return a ImageClassificationPipeline.
"mask-generation": will return a MaskGenerationPipeline.
"object-detection": will return a ObjectDetectionPipeline.
"question-answering": will return a QuestionAnsweringPipeline.
"summarization": will return a SummarizationPipeline.
"table-question-answering": will return a TableQuestionAnsweringPipeline.
"text2text-generation": will return a Text2TextGenerationPipeline.
"text-classification" (alias "sentiment-analysis" available): will return a TextClassificationPipeline.
"text-generation": will return a TextGenerationPipeline:.
"translation": will return a TranslationPipeline.
"translation\_xx\_to\_yy": will return a TranslationPipeline.
"video-classification": will return a VideoClassificationPipeline.
"visual-question-answering": will return a VisualQuestionAnsweringPipeline.
"zero-shot-classification": will return a ZeroShotClassificationPipeline.
model (str or PreTrainedModel or TFPreTrainedModel, optional) — The model that will be used by the pipeline to make predictions. This can be a model identifier or an actual instance of a pretrained model inheriting from PreTrainedModel (for PyTorch) or TFPreTrainedModel (for TensorFlow).

If not provided, the default for the task will be loaded.

config (str or PretrainedConfig, optional) — The configuration that will be used by the pipeline to instantiate the model. This can be a model identifier or an actual pretrained model configuration inheriting from PretrainedConfig.

If not provided, the default configuration file for the requested model will be used. That means that if model is given, its default configuration will be used. However, if model is not supplied, this task’s default model’s config is used instead.

tokenizer (str or PreTrainedTokenizer, optional) — The tokenizer that will be used by the pipeline to encode data for the model. This can be a model identifier or an actual pretrained tokenizer inheriting from PreTrainedTokenizer.

If not provided, the default tokenizer for the given model will be loaded (if it is a string). If model is not specified or not a string, then the default tokenizer for config is loaded (if it is a string). However, if config is also not given or not a string, then the default tokenizer for the given task will be loaded.

feature\_extractor (str or PreTrainedFeatureExtractor, optional) — The feature extractor that will be used by the pipeline to encode data for the model. This can be a model identifier or an actual pretrained feature extractor inheriting from PreTrainedFeatureExtractor.

Feature extractors are used for non-NLP models, such as Speech or Vision models as well as multi-modal models. Multi-modal models will also require a tokenizer to be passed.

If not provided, the default feature extractor for the given model will be loaded (if it is a string). If model is not specified or not a string, then the default feature extractor for config is loaded (if it is a string). However, if config is also not given or not a string, then the default feature extractor for the given task will be loaded.

framework (str, optional) — The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be installed.

If no framework is specified, will default to the one currently installed. If no framework is specified and both frameworks are installed, will default to the framework of the model, or to PyTorch if no model is provided.

revision (str, optional, defaults to "main") — When passing a task name or a string model identifier: The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a git-based system for storing models and other artifacts on huggingface.co, so revision can be any identifier allowed by git.
use\_fast (bool, optional, defaults to True) — Whether or not to use a Fast tokenizer if possible (a PreTrainedTokenizerFast).
device (int or str or torch.device) — Defines the device (e.g., "cpu", "cuda:1", "mps", or a GPU ordinal rank like 1) on which this pipeline will be allocated.
device\_map (str or Dict[str, Union[int, str, torch.device], optional) — Sent directly as model\_kwargs (just a simpler shortcut). When accelerate library is present, set device\_map="auto" to compute the most optimized device\_map automatically (see here for more information).

Do not use device\_map AND device at the same time as they will conflict

torch\_dtype (str or torch.dtype, optional) — Sent directly as model\_kwargs (just a simpler shortcut) to use the available precision for this model (torch.float16, torch.bfloat16, … or "auto").
trust\_remote\_code (bool, optional, defaults to False) — Whether or not to allow for custom code defined on the Hub in their own modeling, configuration, tokenization or even pipeline files. This option should only be set to True for repositories you trust and in which you have read the code, as it will execute code present on the Hub on your local machine.
model\_kwargs (Dict[str, Any], optional) — Additional dictionary of keyword arguments passed along to the model’s from\_pretrained(..., \*\*model\_kwargs) function.
kwargs (Dict[str, Any], optional) — Additional keyword arguments passed along to the specific pipeline init (see the documentation for the corresponding pipeline class for possible values).

Returns

Pipeline

A suitable pipeline for the task.

Utility factory method to build a Pipeline.

Pipelines are made of:

A tokenizer in charge of mapping raw textual input to token.
A model to make predictions from the inputs.
Some (optional) post processing for enhancing model’s output.

Examples:

Copied
>>> from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

>>> # Sentiment analysis pipeline
>>> analyzer = pipeline("sentiment-analysis")

>>> # Question answering pipeline, specifying the checkpoint identifier
>>> oracle = pipeline(
... "question-answering", model="distilbert-base-cased-distilled-squad", tokenizer="bert-base-cased"
... )

>>> # Named entity recognition pipeline, passing in a specific model and tokenizer
>>> model = AutoModelForTokenClassification.from\_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
>>> tokenizer = AutoTokenizer.from\_pretrained("bert-base-cased")
>>> recognizer = pipeline("ner", model=model, tokenizer=tokenizer)
Pipeline batching

All pipelines can use batching. This will work whenever the pipeline uses its streaming ability (so when passing lists or Dataset or generator).

Copied
from transformers import pipeline
from transformers.pipelines.pt\_utils import KeyDataset
import datasets

dataset = datasets.load\_dataset("imdb", name="plain\_text", split="unsupervised")
pipe = pipeline("text-classification", device=0)
for out in pipe(KeyDataset(dataset, "text"), batch\_size=8, truncation="only\_first"):
 print(out)
 # [{'label': 'POSITIVE', 'score': 0.9998743534088135}]
 # Exactly the same output as before, but the content are passed
 # as batches to the model

However, this is not automatically a win for performance. It can be either a 10x speedup or 5x slowdown depending on hardware, data and the actual model being used.

Example where it’s mostly a speedup:

Copied
from transformers import pipeline
from torch.utils.data import Dataset
from tqdm.auto import tqdm

pipe = pipeline("text-classification", device=0)

class MyDataset(Dataset):
 def \_\_len\_\_(self):
 return 5000

 def \_\_getitem\_\_(self, i):
 return "This is a test"

dataset = MyDataset()

for batch\_size in [1, 8, 64, 256]:
 print("-" \* 30)
 print(f"Streaming batch\_size={batch\_size}")
 for out in tqdm(pipe(dataset, batch\_size=batch\_size), total=len(dataset)):
 pass
Copied
# On GTX 970
------------------------------
Streaming no batching
100%|██████████████████████████████████████████████████████████████████████| 5000/5000 [00:26<00:00, 187.52it/s]
------------------------------
Streaming batch\_size=8
100%|█████████████████████████████████████████████████████████████████████| 5000/5000 [00:04<00:00, 1205.95it/s]
------------------------------
Streaming batch\_size=64
100%|█████████████████████████████████████████████████████████████████████| 5000/5000 [00:02<00:00, 2478.24it/s]
------------------------------
Streaming batch\_size=256
100%|█████████████████████████████████████████████████████████████████████| 5000/5000 [00:01<00:00, 2554.43it/s]
(diminishing returns, saturated the GPU)

Example where it’s most a slowdown:

Copied
class MyDataset(Dataset):
 def \_\_len\_\_(self):
 return 5000

 def \_\_getitem\_\_(self, i):
 if i % 64 == 0:
 n = 100
 else:
 n = 1
 return "This is a test" \* n

This is a occasional very long sentence compared to the other. In that case, the whole batch will need to be 400 tokens long, so the whole batch will be [64, 400] instead of [64, 4], leading to the high slowdown. Even worse, on bigger batches, the program simply crashes.

Copied
------------------------------
Streaming no batching
100%|█████████████████████████████████████████████████████████████████████| 1000/1000 [00:05<00:00, 183.69it/s]
------------------------------
Streaming batch\_size=8
100%|█████████████████████████████████████████████████████████████████████| 1000/1000 [00:03<00:00, 265.74it/s]
------------------------------
Streaming batch\_size=64
100%|██████████████████████████████████████████████████████████████████████| 1000/1000 [00:26<00:00, 37.80it/s]
------------------------------
Streaming batch\_size=256
 0%| | 0/1000 [00:00, ?it/s]
Traceback (most recent call last):
 File "/home/nicolas/src/transformers/test.py", line 42, in <module
 for out in tqdm(pipe(dataset, batch\_size=256), total=len(dataset)):
....
 q = q / math.sqrt(dim\_per\_head) # (bs, n\_heads, q\_length, dim\_per\_head)
RuntimeError: CUDA out of memory. Tried to allocate 376.00 MiB (GPU 0; 3.95 GiB total capacity; 1.72 GiB already allocated; 354.88 MiB free; 2.46 GiB reserved in total by PyTorch)

There are no good (general) solutions for this problem, and your mileage may vary depending on your use cases. Rule of thumb:

For users, a rule of thumb is:

Measure performance on your load, with your hardware. Measure, measure, and keep measuring. Real numbers are the only way to go.

If you are latency constrained (live product doing inference), don’t batch

If you are using CPU, don’t batch.

If you are using throughput (you want to run your model on a bunch of static data), on GPU, then:

If you have no clue about the size of the sequence\_length (“natural” data), by default don’t batch, measure and try tentatively to add it, add OOM checks to recover when it will fail (and it will at some point if you don’t control the sequence\_length.)
The larger the GPU the more likely batching is going to be more interesting

As soon as you enable batching, make sure you can handle OOMs nicely.

Pipeline chunk batching

zero-shot-classification and question-answering are slightly specific in the sense, that a single input might yield multiple forward pass of a model. Under normal circumstances, this would yield issues with batch\_size argument.

In order to circumvent this issue, both of these pipelines are a bit specific, they are ChunkPipeline instead of regular Pipeline. In short:

Copied
preprocessed = pipe.preprocess(inputs)

Now becomes:

Copied
all\_model\_outputs = []
for preprocessed in pipe.preprocess(inputs):
 model\_outputs = pipe.forward(preprocessed)
 all\_model\_outputs.append(model\_outputs)

This should be very transparent to your code because the pipelines are used in the same way.

This is a simplified view, since the pipeline can handle automatically the batch to ! Meaning you don’t have to care about how many forward passes you inputs are actually going to trigger, you can optimize the batch\_size independently of the inputs. The caveats from the previous section still apply.

Pipeline custom code

If you want to override a specific pipeline.

Don’t hesitate to create an issue for your task at hand, the goal of the pipeline is to be easy to use and support most cases, so transformers could maybe support your use case.

If you want to try simply you can:

Subclass your pipeline of choice
Copied
class MyPipeline(TextClassificationPipeline):
 def postprocess():
 # Your code goes here
 scores = scores \* 100
 # And here

my\_pipeline = MyPipeline(model=model, tokenizer=tokenizer, ...)
# or if you use \*pipeline\* function, then:
my\_pipeline = pipeline(model="xxxx", pipeline\_class=MyPipeline)

That should enable you to do all the custom code you want.

Implementing a pipeline

Implementing a new pipeline

Audio

Pipelines available for audio tasks include the following.

AudioClassificationPipeline
<
source
>
( \*args\*\*kwargs )

Expand 10 parameters

Parameters

model (PreTrainedModel or TFPreTrainedModel) — The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from PreTrainedModel for PyTorch and TFPreTrainedModel for TensorFlow.
tokenizer (PreTrainedTokenizer) — The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from PreTrainedTokenizer.
modelcard (str or ModelCard, optional) — Model card attributed to the model for this pipeline.
framework (str, optional) — The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be installed.

If no framework is specified, will default to the one currently installed. If no framework is specified and both frameworks are installed, will default to the framework of the model, or to PyTorch if no model is provided.

task (str, defaults to "") — A task-identifier for the pipeline.
num\_workers (int, optional, defaults to 8) — When the pipeline will use DataLoader (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.
args\_parser (ArgumentHandler, optional) — Reference to the object in charge of parsing supplied pipeline parameters.
device (int, optional, defaults to -1) — Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native torch.device or a str too.
binary\_output (bool, optional, defaults to False) — Flag indicating if the output the pipeline should happen in a binary format (i.e., pickle) or as raw text.

Audio classification pipeline using any AutoModelForAudioClassification. This pipeline predicts the class of a raw waveform or an audio file. In case of an audio file, ffmpeg should be installed to support multiple audio formats.

Example:

Copied
>>> from transformers import pipeline

>>> classifier = pipeline(model="superb/wav2vec2-base-superb-ks")
>>> classifier("https://huggingface.co/datasets/Narsil/asr\_dummy/resolve/main/1.flac")
[{'score': 0.997, 'label': '\_unknown\_'}, {'score': 0.002, 'label': 'left'}, {'score': 0.0, 'label': 'yes'}, {'score': 0.0, 'label': 'down'}, {'score': 0.0, 'label': 'stop'}]

Learn more about the basics of using a pipeline in the pipeline tutorial

This pipeline can currently be loaded from pipeline() using the following task identifier: "audio-classification".

See the list of available models on huggingface.co/models.

\_\_call\_\_
<
source
>
( inputs: typing.Union[numpy.ndarray, bytes, str]\*\*kwargs ) → A list of dict with the following keys

Expand 2 parameters

Parameters

inputs (np.ndarray or bytes or str or dict) — The inputs is either :
str that is the filename of the audio file, the file will be read at the correct sampling rate to get the waveform using ffmpeg. This requires ffmpeg to be installed on the system.
bytes it is supposed to be the content of an audio file and is interpreted by ffmpeg in the same way.
(np.ndarray of shape (n, ) of type np.float32 or np.float64) Raw audio at the correct sampling rate (no further check will be done)
dict form can be used to pass raw audio sampled at arbitrary sampling\_rate and let this pipeline do the resampling. The dict must be either be in the format {"sampling\_rate": int, "raw": np.array}, or {"sampling\_rate": int, "array": np.array}, where the key "raw" or "array" is used to denote the raw audio waveform.
top\_k (int, optional, defaults to None) — The number of top labels that will be returned by the pipeline. If the provided number is None or higher than the number of labels available in the model configuration, it will default to the number of labels.

Returns

A list of dict with the following keys

label (str) — The label predicted.
score (float) — The corresponding probability.

Classify the sequence(s) given as inputs. See the AutomaticSpeechRecognitionPipeline documentation for more information.

AutomaticSpeechRecognitionPipeline
<
source
>
( model: PreTrainedModelfeature\_extractor: typing.Union[ForwardRef('SequenceFeatureExtractor'), str] = Nonetokenizer: typing.Optional[transformers.tokenization\_utils.PreTrainedTokenizer] = Nonedecoder: typing.Union[ForwardRef('BeamSearchDecoderCTC'), str, NoneType] = Nonemodelcard: typing.Optional[transformers.modelcard.ModelCard] = Noneframework: typing.Optional[str] = Nonetask: str = ''args\_parser: ArgumentHandler = Nonedevice: typing.Union[int, ForwardRef('torch.device')] = Nonetorch\_dtype: typing.Union[str, ForwardRef('torch.dtype'), NoneType] = Nonebinary\_output: bool = False\*\*kwargs )

Expand 8 parameters

Parameters

model (PreTrainedModel or TFPreTrainedModel) — The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from PreTrainedModel for PyTorch and TFPreTrainedModel for TensorFlow.
tokenizer (PreTrainedTokenizer) — The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from PreTrainedTokenizer.
feature\_extractor (SequenceFeatureExtractor) — The feature extractor that will be used by the pipeline to encode waveform for the model.
chunk\_length\_s (float, optional, defaults to 0) — The input length for in each chunk. If chunk\_length\_s = 0 then chunking is disabled (default).

For more information on how to effectively use chunk\_length\_s, please have a look at the ASR chunking blog post.

stride\_length\_s (float, optional, defaults to chunk\_length\_s / 6) — The length of stride on the left and right of each chunk. Used only with chunk\_length\_s > 0. This enables the model to see more context and infer letters better than without this context but the pipeline discards the stride bits at the end to make the final reconstitution as perfect as possible.

For more information on how to effectively use stride\_length\_s, please have a look at the ASR chunking blog post.

framework (str, optional) — The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be installed. If no framework is specified, will default to the one currently installed. If no framework is specified and both frameworks are installed, will default to the framework of the model, or to PyTorch if no model is provided.
device (Union[int, torch.device], optional) — Device ordinal for CPU/GPU supports. Setting this to None will leverage CPU, a positive will run the model on the associated CUDA device id.
decoder (pyctcdecode.BeamSearchDecoderCTC, optional) — PyCTCDecode’s BeamSearchDecoderCTC can be passed for language model boosted decoding. See Wav2Vec2ProcessorWithLM for more information.

Pipeline that aims at extracting spoken text contained within some audio.

The input can be either a raw waveform or a audio file. In case of the audio file, ffmpeg should be installed for to support multiple audio formats

Example:

Copied
>>> from transformers import pipeline

>>> transcriber = pipeline(model="openai/whisper-base")
>>> transcriber("https://huggingface.co/datasets/Narsil/asr\_dummy/resolve/main/1.flac")
{'text': ' He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick, peppered flour-fatten sauce.'}

Learn more about the basics of using a pipeline in the pipeline tutorial

\_\_call\_\_
<
source
>
( inputs: typing.Union[numpy.ndarray, bytes, str]\*\*kwargs ) → Dict

Expand 4 parameters

Parameters

inputs (np.ndarray or bytes or str or dict) — The inputs is either :

str that is either the filename of a local audio file, or a public URL address to download the audio file. The file will be read at the correct sampling rate to get the waveform using ffmpeg. This requires ffmpeg to be installed on the system.
bytes it is supposed to be the content of an audio file and is interpreted by ffmpeg in the same way.
(np.ndarray of shape (n, ) of type np.float32 or np.float64) Raw audio at the correct sampling rate (no further check will be done)
dict form can be used to pass raw audio sampled at arbitrary sampling\_rate and let this pipeline do the resampling. The dict must be in the format {"sampling\_rate": int, "raw": np.array} with optionally a "stride": (left: int, right: int) than can ask the pipeline to treat the first left samples and last right samples to be ignored in decoding (but used at inference to provide more context to the model). Only use stride with CTC models.
return\_timestamps (optional, str or bool) — Only available for pure CTC models (Wav2Vec2, HuBERT, etc) and the Whisper model. Not available for other sequence-to-sequence models.

For CTC models, timestamps can take one of two formats:

"char": the pipeline will return timestamps along the text for every character in the text. For instance, if you get [{"text": "h", "timestamp": (0.5, 0.6)}, {"text": "i", "timestamp": (0.7, 0.9)}], then it means the model predicts that the letter “h” was spoken after 0.5 and before 0.6 seconds.

For the Whisper model, timestamps can take one of two formats:

"word": same as above for word-level CTC timestamps. Word-level timestamps are predicted through the dynamic-time warping (DTW) algorithm, an approximation to word-level timestamps by inspecting the cross-attention weights.
True: the pipeline will return timestamps along the text for segments of words in the text. For instance, if you get [{"text": " Hi there!", "timestamp": (0.5, 1.5)}], then it means the model predicts that the segment “Hi there!” was spoken after 0.5 and before 1.5 seconds. Note that a segment of text refers to a sequence of one or more words, rather than individual words as with word-level timestamps.
generate\_kwargs (dict, optional) — The dictionary of ad-hoc parametrization of generate\_config to be used for the generation call. For a complete overview of generate, check the following guide.
max\_new\_tokens (int, optional) — The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.

Returns

Dict

A dictionary with the following keys:

text (str): The recognized text.
chunks (optional(, List[Dict]) When using return\_timestamps, the chunks will become a list containing all the various text chunks identified by the model, e.g.\* [{"text": "hi ", "timestamp": (0.5, 0.9)}, {"text": "there", "timestamp": (1.0, 1.5)}]. The original full text can roughly be recovered by doing "".join(chunk["text"] for chunk in output["chunks"]).

Transcribe the audio sequence(s) given as inputs to text. See the AutomaticSpeechRecognitionPipeline documentation for more information.

TextToAudioPipeline
<
source
>
( \*argsvocoder = Nonesampling\_rate = None\*\*kwargs )

Text-to-audio generation pipeline using any AutoModelForTextToWaveform or AutoModelForTextToSpectrogram. This pipeline generates an audio file from an input text and optional other conditional inputs.

Example:

Copied
>>> from transformers import pipeline

>>> pipe = pipeline(model="suno/bark-small")
>>> output = pipe("Hey it's HuggingFace on the phone!")

>>> audio = output["audio"]
>>> sampling\_rate = output["sampling\_rate"]

Learn more about the basics of using a pipeline in the pipeline tutorial

This pipeline can currently be loaded from pipeline() using the following task identifiers: "text-to-speech" or "text-to-audio".

See the list of available models on huggingface.co/models.

\_\_call\_\_
<
source
>
( text\_inputs: typing.Union[str, typing.List[str]]\*\*forward\_params ) → A dict or a list of dict

Parameters

text\_inputs (str or List[str]) — The text(s) to generate.
forward\_params (optional) — Parameters passed to the model generation/forward method.

Returns

A dict or a list of dict

The dictionaries have two keys:

audio (np.ndarray of shape (nb\_channels, audio\_length)) — The generated audio waveform.
sampling\_rate (int) — The sampling rate of the generated audio waveform.

Generates speech/audio from the inputs. See the TextToAudioPipeline documentation for more information.

ZeroShotAudioClassificationPipeline
<
source
>
( \*\*kwargs )

Expand 10 parameters

Parameters

model (PreTrainedModel or TFPreTrainedModel) — The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from PreTrainedModel for PyTorch and TFPreTrainedModel for TensorFlow.
tokenizer (PreTrainedTokenizer) — The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from PreTrainedTokenizer.
modelcard (str or ModelCard, optional) — Model card attributed to the model for this pipeline.
framework (str, optional) — The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be installed.

If no framework is specified, will default to the one currently installed. If no framework is specified and both frameworks are installed, will default to the framework of the model, or to PyTorch if no model is provided.

task (str, defaults to "") — A task-identifier for the pipeline.
num\_workers (int, optional, defaults to 8) — When the pipeline will use DataLoader (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.
args\_parser (ArgumentHandler, optional) — Reference to the object in charge of parsing supplied pipeline parameters.
device (int, optional, defaults to -1) — Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native torch.device or a str too.
binary\_output (bool, optional, defaults to False) — Flag indicating if the output the pipeline should happen in a binary format (i.e., pickle) or as raw text.

Zero shot audio classification pipeline using ClapModel. This pipeline predicts the class of an audio when you provide an audio and a set of candidate\_labels.

Example:

Copied
>>> from transformers import pipeline
>>> from datasets import load\_dataset

>>> dataset = load\_dataset("ashraq/esc50")
>>> audio = next(iter(dataset["train"]["audio"]))["array"]
>>> classifier = pipeline(task="zero-shot-audio-classification", model="laion/clap-htsat-unfused")
>>> classifier(audio, candidate\_labels=["Sound of a dog", "Sound of vaccum cleaner"])
[{'score': 0.9996, 'label': 'Sound of a dog'}, {'score': 0.0004, 'label': 'Sound of vaccum cleaner'}]

Learn more about the basics of using a pipeline in the pipeline tutorial This audio classification pipeline can currently be loaded from pipeline() using the following task identifier: "zero-shot-audio-classification". See the list of available models on huggingface.co/models.

\_\_call\_\_
<
source
>
( audios: typing.Union[numpy.ndarray, bytes, str]\*\*kwargs )

Parameters

audios (str, List[str], np.array or List[np.array]) — The pipeline handles three types of inputs:
A string containing a http link pointing to an audio
An audio loaded in numpy
candidate\_labels (List[str]) — The candidate labels for this audio
hypothesis\_template (str, optional, defaults to "This is a sound of {}") — The sentence used in cunjunction with candidate\_labels to attempt the audio classification by replacing the placeholder with the candidate\_labels. Then likelihood is estimated by using logits\_per\_audio

Assign labels to the audio(s) passed as inputs.

Computer vision

Pipelines available for computer vision tasks include the following.

DepthEstimationPipeline
<
source
>
( \*args\*\*kwargs )

Expand 10 parameters

Parameters

model (PreTrainedModel or TFPreTrainedModel) — The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from PreTrainedModel for PyTorch and TFPreTrainedModel for TensorFlow.
tokenizer (PreTrainedTokenizer) — The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from PreTrainedTokenizer.
modelcard (str or ModelCard, optional) — Model card attributed to the model for this pipeline.
framework (str, optional) — The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be installed.

If no framework is specified, will default to the one currently installed. If no framework is specified and both frameworks are installed, will default to the framework of the model, or to PyTorch if no model is provided.

task (str, defaults to "") — A task-identifier for the pipeline.
num\_workers (int, optional, defaults to 8) — When the pipeline will use DataLoader (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.
args\_parser (ArgumentHandler, optional) — Reference to the object in charge of parsing supplied pipeline parameters.
device (int, optional, defaults to -1) — Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native torch.device or a str too.
binary\_output (bool, optional, defaults to False) — Flag indicating if the output the pipeline should happen in a binary format (i.e., pickle) or as raw text.

Depth estimation pipeline using any AutoModelForDepthEstimation. This pipeline predicts the depth of an image.

Example:

Copied
>>> from transformers import pipeline

>>> depth\_estimator = pipeline(task="depth-estimation", model="Intel/dpt-large")
>>> output = depth\_estimator("http://images.cocodataset.org/val2017/000000039769.jpg")
>>> # This is a tensor with the values being the depth expressed in meters for each pixel
>>> output["predicted\_depth"].shape
torch.Size([1, 384, 384])

Learn more about the basics of using a pipeline in the pipeline tutorial

This depth estimation pipeline can currently be loaded from pipeline() using the following task identifier: "depth-estimation".

See the list of available models on huggingface.co/models.

\_\_call\_\_
<
source
>
( images: typing.Union[str, typing.List[str], ForwardRef('Image.Image'), typing.List[ForwardRef('Image.Image')]]\*\*kwargs )

Parameters

images (str, List[str], PIL.Image or List[PIL.Image]) — The pipeline handles three types of images:

A string containing a http link pointing to an image
An image loaded in PIL directly

The pipeline accepts either a single image or a batch of images, which must then be passed as a string. Images in a batch must all be in the same format: all as http links, all as local paths, or all as PIL images.

top\_k (int, optional, defaults to 5) — The number of top labels that will be returned by the pipeline. If the provided number is higher than the number of labels available in the model configuration, it will default to the number of labels.
timeout (float, optional, defaults to None) — The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and the call may block forever.

Assign labels to the image(s) passed as inputs.

ImageClassificationPipeline
<
source
>
( \*args\*\*kwargs )

Expand 10 parameters

Parameters

model (PreTrainedModel or TFPreTrainedModel) — The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from PreTrainedModel for PyTorch and TFPreTrainedModel for TensorFlow.
tokenizer (PreTrainedTokenizer) — The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from PreTrainedTokenizer.
modelcard (str or ModelCard, optional) — Model card attributed to the model for this pipeline.
framework (str, optional) — The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be installed.

If no framework is specified, will default to the one currently installed. If no framework is specified and both frameworks are installed, will default to the framework of the model, or to PyTorch if no model is provided.

task (str, defaults to "") — A task-identifier for the pipeline.
num\_workers (int, optional, defaults to 8) — When the pipeline will use DataLoader (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.
args\_parser (ArgumentHandler, optional) — Reference to the object in charge of parsing supplied pipeline parameters.
device (int, optional, defaults to -1) — Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native torch.device or a str too.
binary\_output (bool, optional, defaults to False) — Flag indicating if the output the pipeline should happen in a binary format (i.e., pickle) or as raw text.

Image classification pipeline using any AutoModelForImageClassification. This pipeline predicts the class of an image.

Example:

Copied
>>> from transformers import pipeline

>>> classifier = pipeline(model="microsoft/beit-base-patch16-224-pt22k-ft22k")
>>> classifier("https://huggingface.co/datasets/Narsil/image\_dummy/raw/main/parrots.png")
[{'score': 0.442, 'label': 'macaw'}, {'score': 0.088, 'label': 'popinjay'}, {'score': 0.075, 'label': 'parrot'}, {'score': 0.073, 'label': 'parodist, lampooner'}, {'score': 0.046, 'label': 'poll, poll\_parrot'}]

Learn more about the basics of using a pipeline in the pipeline tutorial

This image classification pipeline can currently be loaded from pipeline() using the following task identifier: "image-classification".

See the list of available models on huggingface.co/models.

\_\_call\_\_
<
source
>
( images: typing.Union[str, typing.List[str], ForwardRef('Image.Image'), typing.List[ForwardRef('Image.Image')]]\*\*kwargs )

Parameters

images (str, List[str], PIL.Image or List[PIL.Image]) — The pipeline handles three types of images:

A string containing a http link pointing to an image
An image loaded in PIL directly

The pipeline accepts either a single image or a batch of images, which must then be passed as a string. Images in a batch must all be in the same format: all as http links, all as local paths, or all as PIL images.

top\_k (int, optional, defaults to 5) — The number of top labels that will be returned by the pipeline. If the provided number is higher than the number of labels available in the model configuration, it will default to the number of labels.
timeout (float, optional, defaults to None) — The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and the call may block forever.

Assign labels to the image(s) passed as inputs.

ImageSegmentationPipeline
<
source
>
( \*args\*\*kwargs )

Expand 10 parameters

Parameters

model (PreTrainedModel or TFPreTrainedModel) — The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from PreTrainedModel for PyTorch and TFPreTrainedModel for TensorFlow.
tokenizer (PreTrainedTokenizer) — The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from PreTrainedTokenizer.
modelcard (str or ModelCard, optional) — Model card attributed to the model for this pipeline.
framework (str, optional) — The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be installed.

If no framework is specified, will default to the one currently installed. If no framework is specified and both frameworks are installed, will default to the framework of the model, or to PyTorch if no model is provided.

task (str, defaults to "") — A task-identifier for the pipeline.
num\_workers (int, optional, defaults to 8) — When the pipeline will use DataLoader (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.
args\_parser (ArgumentHandler, optional) — Reference to the object in charge of parsing supplied pipeline parameters.
device (int, optional, defaults to -1) — Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native torch.device or a str too.
binary\_output (bool, optional, defaults to False) — Flag indicating if the output the pipeline should happen in a binary format (i.e., pickle) or as raw text.

Image segmentation pipeline using any AutoModelForXXXSegmentation. This pipeline predicts masks of objects and their classes.

Example:

Copied
>>> from transformers import pipeline

>>> segmenter = pipeline(model="facebook/detr-resnet-50-panoptic")
>>> segments = segmenter("https://huggingface.co/datasets/Narsil/image\_dummy/raw/main/parrots.png")
>>> len(segments)
2
>>> segments[0]["label"]
'bird'

>>> segments[1]["label"]
'bird'

>>> type(segments[0]["mask"]) # This is a black and white mask showing where is the bird on the original image.

>>> segments[0]["mask"].size
(768, 512)

This image segmentation pipeline can currently be loaded from pipeline() using the following task identifier: "image-segmentation".

See the list of available models on huggingface.co/models.

\_\_call\_\_
<
source
>
( images\*\*kwargs )

Expand 6 parameters

Parameters

images (str, List[str], PIL.Image or List[PIL.Image]) — The pipeline handles three types of images:

A string containing an HTTP(S) link pointing to an image
An image loaded in PIL directly

The pipeline accepts either a single image or a batch of images. Images in a batch must all be in the same format: all as HTTP(S) links, all as local paths, or all as PIL images.

subtask (str, optional) — Segmentation task to be performed, choose [semantic, instance and panoptic] depending on model capabilities. If not set, the pipeline will attempt tp resolve in the following order: panoptic, instance, semantic.
threshold (float, optional, defaults to 0.9) — Probability threshold to filter out predicted masks.
timeout (float, optional, defaults to None) — The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and the call may block forever.

Perform segmentation (detect masks & classes) in the image(s) passed as inputs.

ImageToImagePipeline
<
source
>
( \*args\*\*kwargs )

Expand 10 parameters

Parameters

model (PreTrainedModel or TFPreTrainedModel) — The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from PreTrainedModel for PyTorch and TFPreTrainedModel for TensorFlow.
tokenizer (PreTrainedTokenizer) — The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from PreTrainedTokenizer.
modelcard (str or ModelCard, optional) — Model card attributed to the model for this pipeline.
framework (str, optional) — The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be installed.

If no framework is specified, will default to the one currently installed. If no framework is specified and both frameworks are installed, will default to the framework of the model, or to PyTorch if no model is provided.

task (str, defaults to "") — A task-identifier for the pipeline.
num\_workers (int, optional, defaults to 8) — When the pipeline will use DataLoader (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.
args\_parser (ArgumentHandler, optional) — Reference to the object in charge of parsing supplied pipeline parameters.
device (int, optional, defaults to -1) — Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native torch.device or a str too.
binary\_output (bool, optional, defaults to False) — Flag indicating if the output the pipeline should happen in a binary format (i.e., pickle) or as raw text.

Image to Image pipeline using any AutoModelForImageToImage. This pipeline generates an image based on a previous image input.

Example:

Copied
>>> from PIL import Image
>>> import requests

>>> from transformers import pipeline

>>> upscaler = pipeline("image-to-image", model="caidas/swin2SR-classical-sr-x2-64")
>>> img = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
>>> img = img.resize((64, 64))
>>> upscaled\_img = upscaler(img)
>>> img.size
(64, 64)

>>> upscaled\_img.size
(144, 144)

This image to image pipeline can currently be loaded from pipeline() using the following task identifier: "image-to-image".

See the list of available models on huggingface.co/models.

\_\_call\_\_
<
source
>
( images: typing.Union[str, typing.List[str], ForwardRef('Image.Image'), typing.List[ForwardRef('Image.Image')]]\*\*kwargs )

Parameters

images (str, List[str], PIL.Image or List[PIL.Image]) — The pipeline handles three types of images:

A string containing a http link pointing to an image
An image loaded in PIL directly

The pipeline accepts either a single image or a batch of images, which must then be passed as a string. Images in a batch must all be in the same format: all as http links, all as local paths, or all as PIL images.

timeout (float, optional, defaults to None) — The maximum time in seconds to wait for fetching images from the web. If None, no timeout is used and the call may block forever.

Transform the image(s) passed as inputs.

ObjectDetectionPipeline
<
source
>
( \*args\*\*kwargs )

Expand 10 parameters

Parameters

model (PreTrainedModel or TFPreTrainedModel) — The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from PreTrainedModel for PyTorch and TFPreTrainedModel for TensorFlow.
tokenizer (PreTrainedTokenizer) — The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from PreTrainedTokenizer.
modelcard (str or ModelCard, optional) — Model card attributed to the model for this pipeline.
framework (str, optional) — The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be installed.

If no framework is specified, will default to the one currently installed. If no framework is specified and both frameworks are installed, will default to the framework of the model, or to PyTorch if no model is provided.

task (str, defaults to "") — A task-identifier for the pipeline.
num\_workers (int, optional, defaults to 8) — When the pipeline will use DataLoader (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.
args\_parser (ArgumentHandler, optional) — Reference to the object in charge of parsing supplied pipeline parameters.
device (int, optional, defaults to -1) — Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native torch.device or a str too.
binary\_output (bool, optional, defaults to False) — Flag indicating if the output the pipeline should happen in a binary format (i.e., pickle) or as raw text.

Object detection pipeline using any AutoModelForObjectDetection. This pipeline predicts bounding boxes of objects and their classes.

Example:

Copied
>>> from transformers import pipeline

>>> detector = pipeline(model="facebook/detr-resnet-50")
>>> detector("https://huggingface.co/datasets/Narsil/image\_dummy/raw/main/parrots.png")
[{'score': 0.997, 'label': 'bird', 'box': {'xmin': 69, 'ymin': 171, 'xmax': 396, 'ymax': 507}}, {'score': 0.999, 'label': 'bird', 'box': {'xmin': 398, 'ymin': 105, 'xmax': 767, 'ymax': 507}}]

>>> # x, y are expressed relative to the top left hand corner.

Learn more about the basics of using a pipeline in the pipeline tutorial

This object detection pipeline can currently be loaded from pipeline() using the following task identifier: "object-detection".

See the list of available models on huggingface.co/models.

\_\_call\_\_
<
source
>
( \*args\*\*kwargs )

Parameters

images (str, List[str], PIL.Image or List[PIL.Image]) — The pipeline handles three types of images:

A string containing an HTTP(S) link pointing to an image
An image loaded in PIL directly

The pipeline accepts either a single image or a batch of images. Images in a batch must all be in the same format: all as HTTP(S) links, all as local paths, or all as PIL images.

threshold (float, optional, defaults to 0.9) — The probability necessary to make a prediction.
timeout (float, optional, defaults to None) — The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and the call may block forever.

Detect objects (bounding boxes & classes) in the image(s) passed as inputs.

VideoClassificationPipeline
<
source
>
( \*args\*\*kwargs )

Expand 10 parameters

Parameters

model (PreTrainedModel or TFPreTrainedModel) — The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from PreTrainedModel for PyTorch and TFPreTrainedModel for TensorFlow.
tokenizer (PreTrainedTokenizer) — The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from PreTrainedTokenizer.
modelcard (str or ModelCard, optional) — Model card attributed to the model for this pipeline.
framework (str, optional) — The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be installed.

If no framework is specified, will default to the one currently installed. If no framework is specified and both frameworks are installed, will default to the framework of the model, or to PyTorch if no model is provided.

task (str, defaults to "") — A task-identifier for the pipeline.
num\_workers (int, optional, defaults to 8) — When the pipeline will use DataLoader (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.
args\_parser (ArgumentHandler, optional) — Reference to the object in charge of parsing supplied pipeline parameters.
device (int, optional, defaults to -1) — Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native torch.device or a str too.
binary\_output (bool, optional, defaults to False) — Flag indicating if the output the pipeline should happen in a binary format (i.e., pickle) or as raw text.

Video classification pipeline using any AutoModelForVideoClassification. This pipeline predicts the class of a video.

This video classification pipeline can currently be loaded from pipeline() using the following task identifier: "video-classification".

See the list of available models on huggingface.co/models.

\_\_call\_\_
<
source
>
( videos: typing.Union[str, typing.List[str]]\*\*kwargs )

Parameters

videos (str, List[str]) — The pipeline handles three types of videos:

A string containing a http link pointing to a video

The pipeline accepts either a single video or a batch of videos, which must then be passed as a string. Videos in a batch must all be in the same format: all as http links or all as local paths.

top\_k (int, optional, defaults to 5) — The number of top labels that will be returned by the pipeline. If the provided number is higher than the number of labels available in the model configuration, it will default to the number of labels.

Assign labels to the video(s) passed as inputs.

ZeroShotImageClassificationPipeline
<
source
>
( \*\*kwargs )

Expand 10 parameters

Parameters

model (PreTrainedModel or TFPreTrainedModel) — The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from PreTrainedModel for PyTorch and TFPreTrainedModel for TensorFlow.
tokenizer (PreTrainedTokenizer) — The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from PreTrainedTokenizer.
modelcard (str or ModelCard, optional) — Model card attributed to the model for this pipeline.
framework (str, optional) — The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be installed.

If no framework is specified, will default to the one currently installed. If no framework is specified and both frameworks are installed, will default to the framework of the model, or to PyTorch if no model is provided.

task (str, defaults to "") — A task-identifier for the pipeline.
num\_workers (int, optional, defaults to 8) — When the pipeline will use DataLoader (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.
args\_parser (ArgumentHandler, optional) — Reference to the object in charge of parsing supplied pipeline parameters.
device (int, optional, defaults to -1) — Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native torch.device or a str too.
binary\_output (bool, optional, defaults to False) — Flag indicating if the output the pipeline should happen in a binary format (i.e., pickle) or as raw text.

Zero shot image classification pipeline using CLIPModel. This pipeline predicts the class of an image when you provide an image and a set of candidate\_labels.

Example:

Copied
>>> from transformers import pipeline

>>> classifier = pipeline(model="openai/clip-vit-large-patch14")
>>> classifier(
... "https://huggingface.co/datasets/Narsil/image\_dummy/raw/main/parrots.png",
... candidate\_labels=["animals", "humans", "landscape"],
... )
[{'score': 0.965, 'label': 'animals'}, {'score': 0.03, 'label': 'humans'}, {'score': 0.005, 'label': 'landscape'}]

>>> classifier(
... "https://huggingface.co/datasets/Narsil/image\_dummy/raw/main/parrots.png",
... candidate\_labels=["black and white", "photorealist", "painting"],
... )
[{'score': 0.996, 'label': 'black and white'}, {'score': 0.003, 'label': 'photorealist'}, {'score': 0.0, 'label': 'painting'}]

Learn more about the basics of using a pipeline in the pipeline tutorial

This image classification pipeline can currently be loaded from pipeline() using the following task identifier: "zero-shot-image-classification".

See the list of available models on huggingface.co/models.

\_\_call\_\_
<
source
>
( images: typing.Union[str, typing.List[str], ForwardRef('Image'), typing.List[ForwardRef('Image')]]\*\*kwargs )

Parameters

images (str, List[str], PIL.Image or List[PIL.Image]) — The pipeline handles three types of images:

A string containing a http link pointing to an image
An image loaded in PIL directly
candidate\_labels (List[str]) — The candidate labels for this image
hypothesis\_template (str, optional, defaults to "This is a photo of {}") — The sentence used in cunjunction with candidate\_labels to attempt the image classification by replacing the placeholder with the candidate\_labels. Then likelihood is estimated by using logits\_per\_image
timeout (float, optional, defaults to None) — The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and the call may block forever.

Assign labels to the image(s) passed as inputs.

ZeroShotObjectDetectionPipeline
<
source
>
( \*\*kwargs )

Expand 10 parameters

Parameters

model (PreTrainedModel or TFPreTrainedModel) — The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from PreTrainedModel for PyTorch and TFPreTrainedModel for TensorFlow.
tokenizer (PreTrainedTokenizer) — The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from PreTrainedTokenizer.
modelcard (str or ModelCard, optional) — Model card attributed to the model for this pipeline.
framework (str, optional) — The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be installed.

If no framework is specified, will default to the one currently installed. If no framework is specified and both frameworks are installed, will default to the framework of the model, or to PyTorch if no model is provided.

task (str, defaults to "") — A task-identifier for the pipeline.
num\_workers (int, optional, defaults to 8) — When the pipeline will use DataLoader (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.
args\_parser (ArgumentHandler, optional) — Reference to the object in charge of parsing supplied pipeline parameters.
device (int, optional, defaults to -1) — Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native torch.device or a str too.
binary\_output (bool, optional, defaults to False) — Flag indicating if the output the pipeline should happen in a binary format (i.e., pickle) or as raw text.

Zero shot object detection pipeline using OwlViTForObjectDetection. This pipeline predicts bounding boxes of objects when you provide an image and a set of candidate\_labels.

Example:

Copied
>>> from transformers import pipeline

>>> detector = pipeline(model="google/owlvit-base-patch32", task="zero-shot-object-detection")
>>> detector(
... "http://images.cocodataset.org/val2017/000000039769.jpg",
... candidate\_labels=["cat", "couch"],
... )
[{'score': 0.287, 'label': 'cat', 'box': {'xmin': 324, 'ymin': 20, 'xmax': 640, 'ymax': 373}}, {'score': 0.254, 'label': 'cat', 'box': {'xmin': 1, 'ymin': 55, 'xmax': 315, 'ymax': 472}}, {'score': 0.121, 'label': 'couch', 'box': {'xmin': 4, 'ymin': 0, 'xmax': 642, 'ymax': 476}}]

>>> detector(
... "https://huggingface.co/datasets/Narsil/image\_dummy/raw/main/parrots.png",
... candidate\_labels=["head", "bird"],
... )
[{'score': 0.119, 'label': 'bird', 'box': {'xmin': 71, 'ymin': 170, 'xmax': 410, 'ymax': 508}}]

Learn more about the basics of using a pipeline in the pipeline tutorial

This object detection pipeline can currently be loaded from pipeline() using the following task identifier: "zero-shot-object-detection".

See the list of available models on huggingface.co/models.

\_\_call\_\_
<
source
>
( image: typing.Union[str, ForwardRef('Image.Image'), typing.List[typing.Dict[str, typing.Any]]]candidate\_labels: typing.Union[str, typing.List[str]] = None\*\*kwargs )

Parameters

image (str, PIL.Image or List[Dict[str, Any]]) — The pipeline handles three types of images:

A string containing an http url pointing to an image
An image loaded in PIL directly

You can use this parameter to send directly a list of images, or a dataset or a generator like so:

Detect objects (bounding boxes & classes) in the image(s) passed as inputs.

Natural Language Processing

Pipelines available for natural language processing tasks include the following.

ConversationalPipeline
class transformers.Conversation
<
source
>
( messages: typing.Union[str, typing.List[typing.Dict[str, str]]] = Noneconversation\_id: UUID = None\*\*deprecated\_kwargs )

Parameters

messages (Union[str, List[Dict[str, str]]], optional) — The initial messages to start the conversation, either a string, or a list of dicts containing “role” and “content” keys. If a string is passed, it is interpreted as a single message with the “user” role.
conversation\_id (uuid.UUID, optional) — Unique identifier for the conversation. If not provided, a random UUID4 id will be assigned to the conversation.

Utility class containing a conversation and its history. This class is meant to be used as an input to the ConversationalPipeline. The conversation contains several utility functions to manage the addition of new user inputs and generated model responses.

Usage:

Copied
conversation = Conversation("Going to the movies tonight - any suggestions?")
conversation.add\_message({"role": "assistant", "content": "The Big lebowski."})
add\_user\_input
<
source
>
( text: stroverwrite: bool = False )

Add a user input to the conversation for the next round. This is a legacy method that assumes that inputs must alternate user/assistant/user/assistant, and so will not add multiple user messages in succession. We recommend just using add\_message with role “user” instead.

append\_response
<
source
>
( response: str )

This is a legacy method. We recommend just using add\_message with an appropriate role instead.

mark\_processed
<
source
>
( )

This is a legacy method, as the Conversation no longer distinguishes between processed and unprocessed user input. We set a counter here to keep behaviour mostly backward-compatible, but in general you should just read the messages directly when writing new code.

class transformers.ConversationalPipeline
<
source
>
( \*args\*\*kwargs )

Expand 12 parameters

Parameters

model (PreTrainedModel or TFPreTrainedModel) — The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from PreTrainedModel for PyTorch and TFPreTrainedModel for TensorFlow.
tokenizer (PreTrainedTokenizer) — The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from PreTrainedTokenizer.
modelcard (str or ModelCard, optional) — Model card attributed to the model for this pipeline.
framework (str, optional) — The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be installed.

If no framework is specified, will default to the one currently installed. If no framework is specified and both frameworks are installed, will default to the framework of the model, or to PyTorch if no model is provided.

task (str, defaults to "") — A task-identifier for the pipeline.
num\_workers (int, optional, defaults to 8) — When the pipeline will use DataLoader (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.
args\_parser (ArgumentHandler, optional) — Reference to the object in charge of parsing supplied pipeline parameters.
device (int, optional, defaults to -1) — Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native torch.device or a str too.
binary\_output (bool, optional, defaults to False) — Flag indicating if the output the pipeline should happen in a binary format (i.e., pickle) or as raw text.

Multi-turn conversational pipeline.

Example:

Copied
>>> from transformers import pipeline, Conversation

>>> chatbot = pipeline(model="microsoft/DialoGPT-medium")
>>> conversation = Conversation("Going to the movies tonight - any suggestions?")
>>> conversation = chatbot(conversation)
>>> conversation.generated\_responses[-1]
'The Big Lebowski'

>>> conversation.add\_user\_input("Is it an action movie?")
>>> conversation = chatbot(conversation)
>>> conversation.generated\_responses[-1]
"It's a comedy."

Learn more about the basics of using a pipeline in the pipeline tutorial

This conversational pipeline can currently be loaded from pipeline() using the following task identifier: "conversational".

The models that this pipeline can use are models that have been fine-tuned on a multi-turn conversational task, currently: ‘microsoft/DialoGPT-small’, ‘microsoft/DialoGPT-medium’, ‘microsoft/DialoGPT-large’. See the up-to-date list of available models on huggingface.co/models.

\_\_call\_\_
<
source
>
( conversations: typing.Union[typing.List[typing.Dict], transformers.pipelines.conversational.Conversation, typing.List[transformers.pipelines.conversational.Conversation]]num\_workers = 0\*\*kwargs ) → Conversation or a list of Conversation

Parameters

conversations (a Conversation or a list of Conversation) — Conversation to generate responses for. Inputs can also be passed as a list of dictionaries with role and content keys - in this case, they will be converted to Conversation objects automatically. Multiple conversations in either format may be passed as a list.
clean\_up\_tokenization\_spaces (bool, optional, defaults to False) — Whether or not to clean up the potential extra spaces in the text output. generate\_kwargs — Additional keyword arguments to pass along to the generate method of the model (see the generate method corresponding to your framework here).

Returns

Conversation or a list of Conversation

Conversation(s) with updated generated responses for those containing a new user input.

Generate responses for the conversation(s) given as inputs.

FillMaskPipeline
class transformers.FillMaskPipeline
<
source
>
( model: typing.Union[ForwardRef('PreTrainedModel'), ForwardRef('TFPreTrainedModel')]tokenizer: typing.Optional[transformers.tokenization\_utils.PreTrainedTokenizer] = Nonefeature\_extractor: typing.Optional[ForwardRef('SequenceFeatureExtractor')] = Noneimage\_processor: typing.Optional[transformers.image\_processing\_utils.BaseImageProcessor] = Nonemodelcard: typing.Optional[transformers.modelcard.ModelCard] = Noneframework: typing.Optional[str] = Nonetask: str = ''args\_parser: ArgumentHandler = Nonedevice: typing.Union[int, ForwardRef('torch.device')] = Nonetorch\_dtype: typing.Union[str, ForwardRef('torch.dtype'), NoneType] = Nonebinary\_output: bool = False\*\*kwargs )

Expand 12 parameters

Parameters

model (PreTrainedModel or TFPreTrainedModel) — The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from PreTrainedModel for PyTorch and TFPreTrainedModel for TensorFlow.
tokenizer (PreTrainedTokenizer) — The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from PreTrainedTokenizer.
modelcard (str or ModelCard, optional) — Model card attributed to the model for this pipeline.
framework (str, optional) — The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be installed.

If no framework is specified, will default to the one currently installed. If no framework is specified and both frameworks are installed, will default to the framework of the model, or to PyTorch if no model is provided.

task (str, defaults to "") — A task-identifier for the pipeline.
num\_workers (int, optional, defaults to 8) — When the pipeline will use DataLoader (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.
args\_parser (ArgumentHandler, optional) — Reference to the object in charge of parsing supplied pipeline parameters.
device (int, optional, defaults to -1) — Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native torch.device or a str too.
binary\_output (bool, optional, defaults to False) — Flag indicating if the output the pipeline should happen in a binary format (i.e., pickle) or as raw text.
top\_k (int, defaults to 5) — The number of predictions to return.
targets (str or List[str], optional) — When passed, the model will limit the scores to the passed targets instead of looking up in the whole vocab. If the provided targets are not in the model vocab, they will be tokenized and the first resulting token will be used (with a warning, and that might be slower).

Masked language modeling prediction pipeline using any ModelWithLMHead. See the masked language modeling examples for more information.

Example:

Copied
>>> from transformers import pipeline

>>> fill\_masker = pipeline(model="bert-base-uncased")
>>> fill\_masker("This is a simple [MASK].")
[{'score': 0.042, 'token': 3291, 'token\_str': 'problem', 'sequence': 'this is a simple problem.'}, {'score': 0.031, 'token': 3160, 'token\_str': 'question', 'sequence': 'this is a simple question.'}, {'score': 0.03, 'token': 8522, 'token\_str': 'equation', 'sequence': 'this is a simple equation.'}, {'score': 0.027, 'token': 2028, 'token\_str': 'one', 'sequence': 'this is a simple one.'}, {'score': 0.024, 'token': 3627, 'token\_str': 'rule', 'sequence': 'this is a simple rule.'}]

Learn more about the basics of using a pipeline in the pipeline tutorial

This mask filling pipeline can currently be loaded from pipeline() using the following task identifier: "fill-mask".

The models that this pipeline can use are models that have been trained with a masked language modeling objective, which includes the bi-directional models in the library. See the up-to-date list of available models on huggingface.co/models.

This pipeline only works for inputs with exactly one token masked. Experimental: We added support for multiple masks. The returned values are raw model output, and correspond to disjoint probabilities where one might expect joint probabilities (See discussion).

This pipeline now supports tokenizer\_kwargs. For example try:

Copied
>>> from transformers import pipeline

>>> fill\_masker = pipeline(model="bert-base-uncased")
>>> tokenizer\_kwargs = {"truncation": True}
>>> fill\_masker(
... "This is a simple [MASK]. " + "...with a large amount of repeated text appended. " \* 100,
... tokenizer\_kwargs=tokenizer\_kwargs,
... )
\_\_call\_\_
<
source
>
( inputs\*args\*\*kwargs ) → A list or a list of list of dict

Parameters

args (str or List[str]) — One or several texts (or one list of prompts) with masked tokens.
targets (str or List[str], optional) — When passed, the model will limit the scores to the passed targets instead of looking up in the whole vocab. If the provided targets are not in the model vocab, they will be tokenized and the first resulting token will be used (with a warning, and that might be slower).
top\_k (int, optional) — When passed, overrides the number of predictions to return.

Returns

A list or a list of list of dict

Each result comes as list of dictionaries with the following keys:

sequence (str) — The corresponding input with the mask token prediction.
score (float) — The corresponding probability.
token (int) — The predicted token id (to replace the masked one).

Fill the masked token in the text(s) given as inputs.

NerPipeline
class transformers.TokenClassificationPipeline
<
source
>
( args\_parser = \*args\*\*kwargs )

Expand 14 parameters

Parameters

model (PreTrainedModel or TFPreTrainedModel) — The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from PreTrainedModel for PyTorch and TFPreTrainedModel for TensorFlow.
tokenizer (PreTrainedTokenizer) — The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from PreTrainedTokenizer.
modelcard (str or ModelCard, optional) — Model card attributed to the model for this pipeline.
framework (str, optional) — The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be installed.

If no framework is specified, will default to the one currently installed. If no framework is specified and both frameworks are installed, will default to the framework of the model, or to PyTorch if no model is provided.

task (str, defaults to "") — A task-identifier for the pipeline.
num\_workers (int, optional, defaults to 8) — When the pipeline will use DataLoader (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.
args\_parser (ArgumentHandler, optional) — Reference to the object in charge of parsing supplied pipeline parameters.
device (int, optional, defaults to -1) — Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native torch.device or a str too.
binary\_output (bool, optional, defaults to False) — Flag indicating if the output the pipeline should happen in a binary format (i.e., pickle) or as raw text.
ignore\_labels (List[str], defaults to ["O"]) — A list of labels to ignore.
grouped\_entities (bool, optional, defaults to False) — DEPRECATED, use aggregation\_strategy instead. Whether or not to group the tokens corresponding to the same entity together in the predictions or not.
stride (int, optional) — If stride is provided, the pipeline is applied on all the text. The text is split into chunks of size model\_max\_length. Works only with fast tokenizers and aggregation\_strategy different from NONE. The value of this argument defines the number of overlapping tokens between chunks. In other words, the model will shift forward by tokenizer.model\_max\_length - stride tokens each step.
aggregation\_strategy (str, optional, defaults to "none") — The strategy to fuse (or not) tokens based on the model prediction.

“none” : Will simply not do any aggregation and simply return raw results from the model
“simple” : Will attempt to group entities following the default schema. (A, B-TAG), (B, I-TAG), (C, I-TAG), (D, B-TAG2) (E, B-TAG2) will end up being [{“word”: ABC, “entity”: “TAG”}, {“word”: “D”, “entity”: “TAG2”}, {“word”: “E”, “entity”: “TAG2”}] Notice that two consecutive B tags will end up as different entities. On word based languages, we might end up splitting words undesirably : Imagine Microsoft being tagged as [{“word”: “Micro”, “entity”: “ENTERPRISE”}, {“word”: “soft”, “entity”: “NAME”}]. Look for FIRST, MAX, AVERAGE for ways to mitigate that and disambiguate words (on languages that support that meaning, which is basically tokens separated by a space). These mitigations will only work on real words, “New york” might still be tagged with two different entities.
“first” : (works only on word based models) Will use the SIMPLE strategy except that words, cannot end up with different tags. Words will simply use the tag of the first token of the word when there is ambiguity.

Named Entity Recognition pipeline using any ModelForTokenClassification. See the named entity recognition examples for more information.

Example:

Copied
>>> from transformers import pipeline

>>> token\_classifier = pipeline(model="Jean-Baptiste/camembert-ner", aggregation\_strategy="simple")
>>> sentence = "Je m'appelle jean-baptiste et je vis à montréal"
>>> tokens = token\_classifier(sentence)
>>> tokens
[{'entity\_group': 'PER', 'score': 0.9931, 'word': 'jean-baptiste', 'start': 12, 'end': 26}, {'entity\_group': 'LOC', 'score': 0.998, 'word': 'montréal', 'start': 38, 'end': 47}]

>>> token = tokens[0]
>>> # Start and end provide an easy way to highlight words in the original text.
>>> sentence[token["start"] : token["end"]]
' jean-baptiste'

>>> # Some models use the same idea to do part of speech.
>>> syntaxer = pipeline(model="vblagoje/bert-english-uncased-finetuned-pos", aggregation\_strategy="simple")
>>> syntaxer("My name is Sarah and I live in London")
[{'entity\_group': 'PRON', 'score': 0.999, 'word': 'my', 'start': 0, 'end': 2}, {'entity\_group': 'NOUN', 'score': 0.997, 'word': 'name', 'start': 3, 'end': 7}, {'entity\_group': 'AUX', 'score': 0.994, 'word': 'is', 'start': 8, 'end': 10}, {'entity\_group': 'PROPN', 'score': 0.999, 'word': 'sarah', 'start': 11, 'end': 16}, {'entity\_group': 'CCONJ', 'score': 0.999, 'word': 'and', 'start': 17, 'end': 20}, {'entity\_group': 'PRON', 'score': 0.999, 'word': 'i', 'start': 21, 'end': 22}, {'entity\_group': 'VERB', 'score': 0.998, 'word': 'live', 'start': 23, 'end': 27}, {'entity\_group': 'ADP', 'score': 0.999, 'word': 'in', 'start': 28, 'end': 30}, {'entity\_group': 'PROPN', 'score': 0.999, 'word': 'london', 'start': 31, 'end': 37}]

Learn more about the basics of using a pipeline in the pipeline tutorial

This token recognition pipeline can currently be loaded from pipeline() using the following task identifier: "ner" (for predicting the classes of tokens in a sequence: person, organisation, location or miscellaneous).

The models that this pipeline can use are models that have been fine-tuned on a token classification task. See the up-to-date list of available models on huggingface.co/models.

aggregate\_words
<
source
>
( entities: typing.List[dict]aggregation\_strategy: AggregationStrategy )

Override tokens from a given word that disagree to force agreement on word boundaries.

Example: micro|soft| com|pany| B-ENT I-NAME I-ENT I-ENT will be rewritten with first strategy as microsoft| company| B-ENT I-ENT

gather\_pre\_entities
<
source
>
( sentence: strinput\_ids: ndarrayscores: ndarrayoffset\_mapping: typing.Union[typing.List[typing.Tuple[int, int]], NoneType]special\_tokens\_mask: ndarrayaggregation\_strategy: AggregationStrategy )

Fuse various numpy arrays into dicts with all the information needed for aggregation

group\_entities
<
source
>
( entities: typing.List[dict] )

Parameters

entities (dict) — The entities predicted by the pipeline.

Find and group together the adjacent tokens with the same entity predicted.

group\_sub\_entities
<
source
>
( entities: typing.List[dict] )

Parameters

entities (dict) — The entities predicted by the pipeline.

Group together the adjacent tokens with the same entity predicted.

See TokenClassificationPipeline for all details.

QuestionAnsweringPipeline
class transformers.QuestionAnsweringPipeline
<
source
>
( model: typing.Union[ForwardRef('PreTrainedModel'), ForwardRef('TFPreTrainedModel')]tokenizer: PreTrainedTokenizermodelcard: typing.Optional[transformers.modelcard.ModelCard] = Noneframework: typing.Optional[str] = Nonetask: str = ''\*\*kwargs )

Expand 10 parameters

Parameters

model (PreTrainedModel or TFPreTrainedModel) — The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from PreTrainedModel for PyTorch and TFPreTrainedModel for TensorFlow.
tokenizer (PreTrainedTokenizer) — The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from PreTrainedTokenizer.
modelcard (str or ModelCard, optional) — Model card attributed to the model for this pipeline.
framework (str, optional) — The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be installed.

If no framework is specified, will default to the one currently installed. If no framework is specified and both frameworks are installed, will default to the framework of the model, or to PyTorch if no model is provided.

task (str, defaults to "") — A task-identifier for the pipeline.
num\_workers (int, optional, defaults to 8) — When the pipeline will use DataLoader (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.
args\_parser (ArgumentHandler, optional) — Reference to the object in charge of parsing supplied pipeline parameters.
device (int, optional, defaults to -1) — Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native torch.device or a str too.
binary\_output (bool, optional, defaults to False) — Flag indicating if the output the pipeline should happen in a binary format (i.e., pickle) or as raw text.

Question Answering pipeline using any ModelForQuestionAnswering. See the question answering examples for more information.

Example:

Copied
>>> from transformers import pipeline

>>> oracle = pipeline(model="deepset/roberta-base-squad2")
>>> oracle(question="Where do I live?", context="My name is Wolfgang and I live in Berlin")
{'score': 0.9191, 'start': 34, 'end': 40, 'answer': 'Berlin'}

Learn more about the basics of using a pipeline in the pipeline tutorial

This question answering pipeline can currently be loaded from pipeline() using the following task identifier: "question-answering".

The models that this pipeline can use are models that have been fine-tuned on a question answering task. See the up-to-date list of available models on huggingface.co/models.

\_\_call\_\_
<
source
>
( \*args\*\*kwargs ) → A dict or a list of dict

Expand 12 parameters

Parameters

args (SquadExample or a list of SquadExample) — One or several SquadExample containing the question and context.
doc\_stride (int, optional, defaults to 128) — If the context is too long to fit with the question for the model, it will be split in several chunks with some overlap. This argument controls the size of that overlap.
max\_answer\_len (int, optional, defaults to 15) — The maximum length of predicted answers (e.g., only answers with a shorter length are considered).
align\_to\_words (bool, optional, defaults to True) — Attempts to align the answer to real words. Improves quality on space separated langages. Might hurt on non-space-separated languages (like Japanese or Chinese)

Returns

A dict or a list of dict

Each result comes as a dictionary with the following keys:

score (float) — The probability associated to the answer.
start (int) — The character start index of the answer (in the tokenized version of the input).
answer (str) — The answer to the question.

Answer the question(s) given as inputs by using the context(s).

create\_sample
<
source
>
( question: typing.Union[str, typing.List[str]]context: typing.Union[str, typing.List[str]] ) → One or a list of SquadExample

Parameters

question (str or List[str]) — The question(s) asked.

Returns

One or a list of SquadExample

The corresponding SquadExample grouping question and context.

QuestionAnsweringPipeline leverages the SquadExample internally. This helper method encapsulate all the logic for converting question(s) and context(s) to SquadExample.

We currently support extractive question answering.

span\_to\_answer
<
source
>
( text: strstart: intend: int ) → Dictionary like `{‘answer’

Parameters

text (str) — The actual context to extract the answer from.
start (int) — The answer starting token index.

Returns

Dictionary like `{‘answer’

str, ‘start’: int, ‘end’: int}`

When decoding from token probabilities, this method maps token indexes to actual word in the initial context.

SummarizationPipeline
<
source
>
( \*args\*\*kwargs )

Expand 10 parameters

Parameters

model (PreTrainedModel or TFPreTrainedModel) — The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from PreTrainedModel for PyTorch and TFPreTrainedModel for TensorFlow.
tokenizer (PreTrainedTokenizer) — The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from PreTrainedTokenizer.
modelcard (str or ModelCard, optional) — Model card attributed to the model for this pipeline.
framework (str, optional) — The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be installed.

If no framework is specified, will default to the one currently installed. If no framework is specified and both frameworks are installed, will default to the framework of the model, or to PyTorch if no model is provided.

task (str, defaults to "") — A task-identifier for the pipeline.
num\_workers (int, optional, defaults to 8) — When the pipeline will use DataLoader (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.
args\_parser (ArgumentHandler, optional) — Reference to the object in charge of parsing supplied pipeline parameters.
device (int, optional, defaults to -1) — Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native torch.device or a str too.
binary\_output (bool, optional, defaults to False) — Flag indicating if the output the pipeline should happen in a binary format (i.e., pickle) or as raw text.

Summarize news articles and other documents.

This summarizing pipeline can currently be loaded from pipeline() using the following task identifier: "summarization".

The models that this pipeline can use are models that have been fine-tuned on a summarization task, which is currently, ’bart-large-cnn’, ’t5-small’, ’t5-base’, ’t5-large’, ’t5-3b’, ’t5-11b’. See the up-to-date list of available models on huggingface.co/models. For a list of available parameters, see the following documentation

Usage:

Copied
# use bart in pytorch
summarizer = pipeline("summarization")
summarizer("An apple a day, keeps the doctor away", min\_length=5, max\_length=20)

# use t5 in tf
summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf")
summarizer("An apple a day, keeps the doctor away", min\_length=5, max\_length=20)
\_\_call\_\_
<
source
>
( \*args\*\*kwargs ) → A list or a list of list of dict

Parameters

documents (str or List[str]) — One or several articles (or one list of articles) to summarize.
return\_text (bool, optional, defaults to True) — Whether or not to include the decoded texts in the outputs
clean\_up\_tokenization\_spaces (bool, optional, defaults to False) — Whether or not to clean up the potential extra spaces in the text output. generate\_kwargs — Additional keyword arguments to pass along to the generate method of the model (see the generate method corresponding to your framework here).

Returns

A list or a list of list of dict

Each result comes as a dictionary with the following keys:

summary\_text (str, present when return\_text=True) — The summary of the corresponding input.

Summarize the text(s) given as inputs.

TableQuestionAnsweringPipeline
<
source
>
( args\_parser = \*args\*\*kwargs )

Expand 10 parameters

Parameters

model (PreTrainedModel or TFPreTrainedModel) — The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from PreTrainedModel for PyTorch and TFPreTrainedModel for TensorFlow.
tokenizer (PreTrainedTokenizer) — The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from PreTrainedTokenizer.
modelcard (str or ModelCard, optional) — Model card attributed to the model for this pipeline.
framework (str, optional) — The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be installed.

If no framework is specified, will default to the one currently installed. If no framework is specified and both frameworks are installed, will default to the framework of the model, or to PyTorch if no model is provided.

task (str, defaults to "") — A task-identifier for the pipeline.
num\_workers (int, optional, defaults to 8) — When the pipeline will use DataLoader (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.
args\_parser (ArgumentHandler, optional) — Reference to the object in charge of parsing supplied pipeline parameters.
device (int, optional, defaults to -1) — Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native torch.device or a str too.
binary\_output (bool, optional, defaults to False) — Flag indicating if the output the pipeline should happen in a binary format (i.e., pickle) or as raw text.

Table Question Answering pipeline using a ModelForTableQuestionAnswering. This pipeline is only available in PyTorch.

Example:

Copied
>>> from transformers import pipeline

>>> oracle = pipeline(model="google/tapas-base-finetuned-wtq")
>>> table = {
... "Repository": ["Transformers", "Datasets", "Tokenizers"],
... "Stars": ["36542", "4512", "3934"],
... "Programming language": ["Python", "Python", "Rust, Python and NodeJS"],
... }
>>> oracle(query="How many stars does the transformers repository have?", table=table)
{'answer': 'AVERAGE > 36542', 'coordinates': [(0, 1)], 'cells': ['36542'], 'aggregator': 'AVERAGE'}

Learn more about the basics of using a pipeline in the pipeline tutorial

This tabular question answering pipeline can currently be loaded from pipeline() using the following task identifier: "table-question-answering".

The models that this pipeline can use are models that have been fine-tuned on a tabular question answering task. See the up-to-date list of available models on huggingface.co/models.

\_\_call\_\_
<
source
>
( \*args\*\*kwargs ) → A dictionary or a list of dictionaries containing results

Expand 5 parameters

Parameters

table (pd.DataFrame or Dict) — Pandas DataFrame or dictionary that will be converted to a DataFrame containing all the table values. See above for an example of dictionary.
query (str or List[str]) — Query or list of queries that will be sent to the model alongside the table.
sequential (bool, optional, defaults to False) — Whether to do inference sequentially or as a batch. Batching is faster, but models like SQA require the inference to be done sequentially to extract relations within sequences, given their conversational nature.
padding (bool, str or PaddingStrategy, optional, defaults to False) — Activates and controls padding. Accepts the following values:

True or 'longest': Pad to the longest sequence in the batch (or no padding if only a single sequence if provided).
'max\_length': Pad to a maximum length specified with the argument max\_length or to the maximum acceptable input length for the model if that argument is not provided.
False or 'do\_not\_pad' (default): No padding (i.e., can output a batch with sequences of different lengths).
truncation (bool, str or TapasTruncationStrategy, optional, defaults to False) — Activates and controls truncation. Accepts the following values:

True or 'drop\_rows\_to\_fit': Truncate to a maximum length specified with the argument max\_length or to the maximum acceptable input length for the model if that argument is not provided. This will truncate row by row, removing rows from the table.
False or 'do\_not\_truncate' (default): No truncation (i.e., can output batch with sequence lengths greater than the model maximum admissible input size).

Returns

A dictionary or a list of dictionaries containing results

Each result is a dictionary with the following keys:

answer (str) — The answer of the query given the table. If there is an aggregator, the answer will be preceded by AGGREGATOR >.
coordinates (List[Tuple[int, int]]) — Coordinates of the cells of the answers.
cells (List[str]) — List of strings made up of the answer cell values.
aggregator (str) — If the model has an aggregator, this returns the aggregator.

Answers queries according to a table. The pipeline accepts several types of inputs which are detailed below:

pipeline(table, query)
pipeline(table=table, query=query)
pipeline({"table": table, "query": query})

The table argument should be a dict or a DataFrame built from that dict, containing the whole table:

Example:

Copied
data = {
 "actors": ["brad pitt", "leonardo di caprio", "george clooney"],
 "age": ["56", "45", "59"],
 "date of birth": ["7 february 1967", "10 june 1996", "28 november 1967"],
}
This dictionary can be passed in as such, or can be converted to a pandas DataFrame:

Example:

Copied
import pandas as pd

table = pd.DataFrame.from\_dict(data)
TextClassificationPipeline
<
source
>
( \*\*kwargs )

Expand 12 parameters

Parameters

model (PreTrainedModel or TFPreTrainedModel) — The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from PreTrainedModel for PyTorch and TFPreTrainedModel for TensorFlow.
tokenizer (PreTrainedTokenizer) — The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from PreTrainedTokenizer.
modelcard (str or ModelCard, optional) — Model card attributed to the model for this pipeline.
framework (str, optional) — The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be installed.

If no framework is specified, will default to the one currently installed. If no framework is specified and both frameworks are installed, will default to the framework of the model, or to PyTorch if no model is provided.

task (str, defaults to "") — A task-identifier for the pipeline.
num\_workers (int, optional, defaults to 8) — When the pipeline will use DataLoader (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.
args\_parser (ArgumentHandler, optional) — Reference to the object in charge of parsing supplied pipeline parameters.
device (int, optional, defaults to -1) — Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native torch.device or a str too.
binary\_output (bool, optional, defaults to False) — Flag indicating if the output the pipeline should happen in a binary format (i.e., pickle) or as raw text.

"default": if the model has a single label, will apply the sigmoid function on the output. If the model has several labels, will apply the softmax function on the output.
"sigmoid": Applies the sigmoid function on the output.
"softmax": Applies the softmax function on the output.

Text classification pipeline using any ModelForSequenceClassification. See the sequence classification examples for more information.

Example:

Copied
>>> from transformers import pipeline

>>> classifier = pipeline(model="distilbert-base-uncased-finetuned-sst-2-english")
>>> classifier("This movie is disgustingly good !")
[{'label': 'POSITIVE', 'score': 1.0}]

>>> classifier("Director tried too much.")
[{'label': 'NEGATIVE', 'score': 0.996}]

Learn more about the basics of using a pipeline in the pipeline tutorial

This text classification pipeline can currently be loaded from pipeline() using the following task identifier: "sentiment-analysis" (for classifying sequences according to positive or negative sentiments).

If multiple classification labels are available (model.config.num\_labels >= 2), the pipeline will run a softmax over the results. If there is a single label, the pipeline will run a sigmoid over the result.

The models that this pipeline can use are models that have been fine-tuned on a sequence classification task. See the up-to-date list of available models on huggingface.co/models.

\_\_call\_\_
<
source
>
( \*args\*\*kwargs ) → A list or a list of list of dict

Expand 3 parameters

Parameters

args (str or List[str] or Dict[str], or List[Dict[str]]) — One or several texts to classify. In order to use text pairs for your classification, you can send a dictionary containing {"text", "text\_pair"} keys, or a list of those.
top\_k (int, optional, defaults to 1) — How many results to return.
function\_to\_apply (str, optional, defaults to "default") — The function to apply to the model outputs in order to retrieve the scores. Accepts four different values:

If this argument is not specified, then it will apply the following functions according to the number of labels:

If the model has a single label, will apply the sigmoid function on the output.

Possible values are:

"sigmoid": Applies the sigmoid function on the output.
"softmax": Applies the softmax function on the output.

Returns

A list or a list of list of dict

Each result comes as list of dictionaries with the following keys:

label (str) — The label predicted.
score (float) — The corresponding probability.

If top\_k is used, one such dictionary is returned per label.

Classify the text(s) given as inputs.

TextGenerationPipeline
class transformers.TextGenerationPipeline
<
source
>
( \*args\*\*kwargs )

Expand 10 parameters

Parameters

model (PreTrainedModel or TFPreTrainedModel) — The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from PreTrainedModel for PyTorch and TFPreTrainedModel for TensorFlow.
tokenizer (PreTrainedTokenizer) — The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from PreTrainedTokenizer.
modelcard (str or ModelCard, optional) — Model card attributed to the model for this pipeline.
framework (str, optional) — The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be installed.

If no framework is specified, will default to the one currently installed. If no framework is specified and both frameworks are installed, will default to the framework of the model, or to PyTorch if no model is provided.

task (str, defaults to "") — A task-identifier for the pipeline.
num\_workers (int, optional, defaults to 8) — When the pipeline will use DataLoader (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.
args\_parser (ArgumentHandler, optional) — Reference to the object in charge of parsing supplied pipeline parameters.
device (int, optional, defaults to -1) — Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native torch.device or a str too.
binary\_output (bool, optional, defaults to False) — Flag indicating if the output the pipeline should happen in a binary format (i.e., pickle) or as raw text.

Language generation pipeline using any ModelWithLMHead. This pipeline predicts the words that will follow a specified text prompt.

Example:

Copied
>>> from transformers import pipeline

>>> generator = pipeline(model="gpt2")
>>> generator("I can't believe you did such a ", do\_sample=False)
[{'generated\_text': "I can't believe you did such a icky thing to me. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I"}]

>>> # These parameters will return suggestions, and only the newly created text making it easier for prompting suggestions.
>>> outputs = generator("My tart needs some", num\_return\_sequences=4, return\_full\_text=False)

Learn more about the basics of using a pipeline in the pipeline tutorial. You can pass text generation parameters to this pipeline to control stopping criteria, decoding strategy, and more. Learn more about text generation parameters in Text generation strategies and Text generation.

This language generation pipeline can currently be loaded from pipeline() using the following task identifier: "text-generation".

The models that this pipeline can use are models that have been trained with an autoregressive language modeling objective, which includes the uni-directional models in the library (e.g. gpt2). See the list of available models on huggingface.co/models.

\_\_call\_\_
<
source
>
( text\_inputs\*\*kwargs ) → A list or a list of list of dict

Expand 7 parameters

Parameters

args (str or List[str]) — One or several prompts (or one list of prompts) to complete.
return\_tensors (bool, optional, defaults to False) — Whether or not to return the tensors of predictions (as token indices) in the outputs. If set to True, the decoded text is not returned.
clean\_up\_tokenization\_spaces (bool, optional, defaults to False) — Whether or not to clean up the potential extra spaces in the text output.
prefix (str, optional) — Prefix added to prompt.
handle\_long\_generation (str, optional) — By default, this pipelines does not handle long generation (ones that exceed in one form or the other the model maximum length). There is no perfect way to adress this (more info :https://github.com/huggingface/transformers/issues/14033#issuecomment-948385227). This provides common strategies to work around that problem depending on your use case.

None : default strategy where nothing in particular happens
"hole": Truncates left of input, and leaves a gap wide enough to let generation happen (might truncate a lot of the prompt and not suitable when generation exceed the model capacity)

generate\_kwargs — Additional keyword arguments to pass along to the generate method of the model (see the generate method corresponding to your framework here).

Returns

A list or a list of list of dict

Returns one of the following dictionaries (cannot return a combination of both generated\_text and generated\_token\_ids):

generated\_text (str, present when return\_text=True) — The generated text.

Complete the prompt(s) given as inputs.

Text2TextGenerationPipeline
<
source
>
( \*args\*\*kwargs )

Expand 10 parameters

Parameters

model (PreTrainedModel or TFPreTrainedModel) — The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from PreTrainedModel for PyTorch and TFPreTrainedModel for TensorFlow.
tokenizer (PreTrainedTokenizer) — The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from PreTrainedTokenizer.
modelcard (str or ModelCard, optional) — Model card attributed to the model for this pipeline.
framework (str, optional) — The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be installed.

If no framework is specified, will default to the one currently installed. If no framework is specified and both frameworks are installed, will default to the framework of the model, or to PyTorch if no model is provided.

task (str, defaults to "") — A task-identifier for the pipeline.
num\_workers (int, optional, defaults to 8) — When the pipeline will use DataLoader (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.
args\_parser (ArgumentHandler, optional) — Reference to the object in charge of parsing supplied pipeline parameters.
device (int, optional, defaults to -1) — Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native torch.device or a str too.
binary\_output (bool, optional, defaults to False) — Flag indicating if the output the pipeline should happen in a binary format (i.e., pickle) or as raw text.

Pipeline for text to text generation using seq2seq models.

Example:

Copied
>>> from transformers import pipeline

>>> generator = pipeline(model="mrm8488/t5-base-finetuned-question-generation-ap")
>>> generator(
... "answer: Manuel context: Manuel has created RuPERTa-base with the support of HF-Transformers and Google"
... )
[{'generated\_text': 'question: Who created the RuPERTa-base?'}]

Learn more about the basics of using a pipeline in the pipeline tutorial. You can pass text generation parameters to this pipeline to control stopping criteria, decoding strategy, and more. Learn more about text generation parameters in Text generation strategies and Text generation.

This Text2TextGenerationPipeline pipeline can currently be loaded from pipeline() using the following task identifier: "text2text-generation".

The models that this pipeline can use are models that have been fine-tuned on a translation task. See the up-to-date list of available models on huggingface.co/models. For a list of available parameters, see the following documentation

Usage:

Copied
text2text\_generator = pipeline("text2text-generation")
text2text\_generator("question: What is 42 ? context: 42 is the answer to life, the universe and everything")
\_\_call\_\_
<
source
>
( \*args\*\*kwargs ) → A list or a list of list of dict

Expand 5 parameters

Parameters

args (str or List[str]) — Input text for the encoder.
return\_tensors (bool, optional, defaults to False) — Whether or not to include the tensors of predictions (as token indices) in the outputs.
truncation (TruncationStrategy, optional, defaults to TruncationStrategy.DO\_NOT\_TRUNCATE) — The truncation strategy for the tokenization within the pipeline. TruncationStrategy.DO\_NOT\_TRUNCATE (default) will never truncate, but it is sometimes desirable to truncate the input to fit the model’s max\_length instead of throwing an error down the line. generate\_kwargs — Additional keyword arguments to pass along to the generate method of the model (see the generate method corresponding to your framework here).

Returns

A list or a list of list of dict

Each result comes as a dictionary with the following keys:

generated\_text (str, present when return\_text=True) — The generated text.

Generate the output text(s) using text(s) given as inputs.

check\_inputs
<
source
>
( input\_length: intmin\_length: intmax\_length: int )

Checks whether there might be something wrong with given input with regard to the model.

TokenClassificationPipeline
<
source
>
( args\_parser = \*args\*\*kwargs )

Expand 14 parameters

Parameters

model (PreTrainedModel or TFPreTrainedModel) — The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from PreTrainedModel for PyTorch and TFPreTrainedModel for TensorFlow.
tokenizer (PreTrainedTokenizer) — The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from PreTrainedTokenizer.
modelcard (str or ModelCard, optional) — Model card attributed to the model for this pipeline.
framework (str, optional) — The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be installed.

If no framework is specified, will default to the one currently installed. If no framework is specified and both frameworks are installed, will default to the framework of the model, or to PyTorch if no model is provided.

task (str, defaults to "") — A task-identifier for the pipeline.
num\_workers (int, optional, defaults to 8) — When the pipeline will use DataLoader (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.
args\_parser (ArgumentHandler, optional) — Reference to the object in charge of parsing supplied pipeline parameters.
device (int, optional, defaults to -1) — Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native torch.device or a str too.
binary\_output (bool, optional, defaults to False) — Flag indicating if the output the pipeline should happen in a binary format (i.e., pickle) or as raw text.
ignore\_labels (List[str], defaults to ["O"]) — A list of labels to ignore.
grouped\_entities (bool, optional, defaults to False) — DEPRECATED, use aggregation\_strategy instead. Whether or not to group the tokens corresponding to the same entity together in the predictions or not.
stride (int, optional) — If stride is provided, the pipeline is applied on all the text. The text is split into chunks of size model\_max\_length. Works only with fast tokenizers and aggregation\_strategy different from NONE. The value of this argument defines the number of overlapping tokens between chunks. In other words, the model will shift forward by tokenizer.model\_max\_length - stride tokens each step.
aggregation\_strategy (str, optional, defaults to "none") — The strategy to fuse (or not) tokens based on the model prediction.

“none” : Will simply not do any aggregation and simply return raw results from the model
“simple” : Will attempt to group entities following the default schema. (A, B-TAG), (B, I-TAG), (C, I-TAG), (D, B-TAG2) (E, B-TAG2) will end up being [{“word”: ABC, “entity”: “TAG”}, {“word”: “D”, “entity”: “TAG2”}, {“word”: “E”, “entity”: “TAG2”}] Notice that two consecutive B tags will end up as different entities. On word based languages, we might end up splitting words undesirably : Imagine Microsoft being tagged as [{“word”: “Micro”, “entity”: “ENTERPRISE”}, {“word”: “soft”, “entity”: “NAME”}]. Look for FIRST, MAX, AVERAGE for ways to mitigate that and disambiguate words (on languages that support that meaning, which is basically tokens separated by a space). These mitigations will only work on real words, “New york” might still be tagged with two different entities.
“first” : (works only on word based models) Will use the SIMPLE strategy except that words, cannot end up with different tags. Words will simply use the tag of the first token of the word when there is ambiguity.

Named Entity Recognition pipeline using any ModelForTokenClassification. See the named entity recognition examples for more information.

Example:

Copied
>>> from transformers import pipeline

>>> token\_classifier = pipeline(model="Jean-Baptiste/camembert-ner", aggregation\_strategy="simple")
>>> sentence = "Je m'appelle jean-baptiste et je vis à montréal"
>>> tokens = token\_classifier(sentence)
>>> tokens
[{'entity\_group': 'PER', 'score': 0.9931, 'word': 'jean-baptiste', 'start': 12, 'end': 26}, {'entity\_group': 'LOC', 'score': 0.998, 'word': 'montréal', 'start': 38, 'end': 47}]

>>> token = tokens[0]
>>> # Start and end provide an easy way to highlight words in the original text.
>>> sentence[token["start"] : token["end"]]
' jean-baptiste'

>>> # Some models use the same idea to do part of speech.
>>> syntaxer = pipeline(model="vblagoje/bert-english-uncased-finetuned-pos", aggregation\_strategy="simple")
>>> syntaxer("My name is Sarah and I live in London")
[{'entity\_group': 'PRON', 'score': 0.999, 'word': 'my', 'start': 0, 'end': 2}, {'entity\_group': 'NOUN', 'score': 0.997, 'word': 'name', 'start': 3, 'end': 7}, {'entity\_group': 'AUX', 'score': 0.994, 'word': 'is', 'start': 8, 'end': 10}, {'entity\_group': 'PROPN', 'score': 0.999, 'word': 'sarah', 'start': 11, 'end': 16}, {'entity\_group': 'CCONJ', 'score': 0.999, 'word': 'and', 'start': 17, 'end': 20}, {'entity\_group': 'PRON', 'score': 0.999, 'word': 'i', 'start': 21, 'end': 22}, {'entity\_group': 'VERB', 'score': 0.998, 'word': 'live', 'start': 23, 'end': 27}, {'entity\_group': 'ADP', 'score': 0.999, 'word': 'in', 'start': 28, 'end': 30}, {'entity\_group': 'PROPN', 'score': 0.999, 'word': 'london', 'start': 31, 'end': 37}]

Learn more about the basics of using a pipeline in the pipeline tutorial

This token recognition pipeline can currently be loaded from pipeline() using the following task identifier: "ner" (for predicting the classes of tokens in a sequence: person, organisation, location or miscellaneous).

The models that this pipeline can use are models that have been fine-tuned on a token classification task. See the up-to-date list of available models on huggingface.co/models.

\_\_call\_\_
<
source
>
( inputs: typing.Union[str, typing.List[str]]\*\*kwargs ) → A list or a list of list of dict

Expand 1 parameters

Parameters

inputs (str or List[str]) — One or several texts (or one list of texts) for token classification.

Returns

A list or a list of list of dict

Each result comes as a list of dictionaries (one for each token in the corresponding input, or each entity if this pipeline was instantiated with an aggregation\_strategy) with the following keys:

word (str) — The token/word classified. This is obtained by decoding the selected tokens. If you want to have the exact string in the original sentence, use start and end.
score (float) — The corresponding probability for entity.
entity (str) — The entity predicted for that token/word (it is named entity\_group when aggregation\_strategy is not "none".
index (int, only present when aggregation\_strategy="none") — The index of the corresponding token in the sentence.

Classify each token of the text(s) given as inputs.

aggregate\_words
<
source
>
( entities: typing.List[dict]aggregation\_strategy: AggregationStrategy )

Override tokens from a given word that disagree to force agreement on word boundaries.

Example: micro|soft| com|pany| B-ENT I-NAME I-ENT I-ENT will be rewritten with first strategy as microsoft| company| B-ENT I-ENT

gather\_pre\_entities
<
source
>
( sentence: strinput\_ids: ndarrayscores: ndarrayoffset\_mapping: typing.Union[typing.List[typing.Tuple[int, int]], NoneType]special\_tokens\_mask: ndarrayaggregation\_strategy: AggregationStrategy )

Fuse various numpy arrays into dicts with all the information needed for aggregation

group\_entities
<
source
>
( entities: typing.List[dict] )

Parameters

entities (dict) — The entities predicted by the pipeline.

Find and group together the adjacent tokens with the same entity predicted.

group\_sub\_entities
<
source
>
( entities: typing.List[dict] )

Parameters

entities (dict) — The entities predicted by the pipeline.

Group together the adjacent tokens with the same entity predicted.

TranslationPipeline
class transformers.TranslationPipeline
<
source
>
( \*args\*\*kwargs )

Expand 10 parameters

Parameters

model (PreTrainedModel or TFPreTrainedModel) — The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from PreTrainedModel for PyTorch and TFPreTrainedModel for TensorFlow.
tokenizer (PreTrainedTokenizer) — The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from PreTrainedTokenizer.
modelcard (str or ModelCard, optional) — Model card attributed to the model for this pipeline.
framework (str, optional) — The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be installed.

If no framework is specified, will default to the one currently installed. If no framework is specified and both frameworks are installed, will default to the framework of the model, or to PyTorch if no model is provided.

task (str, defaults to "") — A task-identifier for the pipeline.
num\_workers (int, optional, defaults to 8) — When the pipeline will use DataLoader (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.
args\_parser (ArgumentHandler, optional) — Reference to the object in charge of parsing supplied pipeline parameters.
device (int, optional, defaults to -1) — Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native torch.device or a str too.
binary\_output (bool, optional, defaults to False) — Flag indicating if the output the pipeline should happen in a binary format (i.e., pickle) or as raw text.

Translates from one language to another.

This translation pipeline can currently be loaded from pipeline() using the following task identifier: "translation\_xx\_to\_yy".

The models that this pipeline can use are models that have been fine-tuned on a translation task. See the up-to-date list of available models on huggingface.co/models. For a list of available parameters, see the following documentation

Usage:

Copied
en\_fr\_translator = pipeline("translation\_en\_to\_fr")
en\_fr\_translator("How old are you?")
\_\_call\_\_
<
source
>
( \*args\*\*kwargs ) → A list or a list of list of dict

Expand 6 parameters

Parameters

args (str or List[str]) — Texts to be translated.
return\_tensors (bool, optional, defaults to False) — Whether or not to include the tensors of predictions (as token indices) in the outputs.
src\_lang (str, optional) — The language of the input. Might be required for multilingual models. Will not have any effect for single pair translation models
tgt\_lang (str, optional) — The language of the desired output. Might be required for multilingual models. Will not have any effect for single pair translation models generate\_kwargs — Additional keyword arguments to pass along to the generate method of the model (see the generate method corresponding to your framework here).

Returns

A list or a list of list of dict

Each result comes as a dictionary with the following keys:

translation\_text (str, present when return\_text=True) — The translation.

Translate the text(s) given as inputs.

ZeroShotClassificationPipeline
<
source
>
( args\_parser = \*args\*\*kwargs )

Expand 10 parameters

Parameters

model (PreTrainedModel or TFPreTrainedModel) — The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from PreTrainedModel for PyTorch and TFPreTrainedModel for TensorFlow.
tokenizer (PreTrainedTokenizer) — The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from PreTrainedTokenizer.
modelcard (str or ModelCard, optional) — Model card attributed to the model for this pipeline.
framework (str, optional) — The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be installed.

If no framework is specified, will default to the one currently installed. If no framework is specified and both frameworks are installed, will default to the framework of the model, or to PyTorch if no model is provided.

task (str, defaults to "") — A task-identifier for the pipeline.
num\_workers (int, optional, defaults to 8) — When the pipeline will use DataLoader (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.
args\_parser (ArgumentHandler, optional) — Reference to the object in charge of parsing supplied pipeline parameters.
device (int, optional, defaults to -1) — Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native torch.device or a str too.
binary\_output (bool, optional, defaults to False) — Flag indicating if the output the pipeline should happen in a binary format (i.e., pickle) or as raw text.

NLI-based zero-shot classification pipeline using a ModelForSequenceClassification trained on NLI (natural language inference) tasks. Equivalent of text-classification pipelines, but these models don’t require a hardcoded number of potential classes, they can be chosen at runtime. It usually means it’s slower but it is much more flexible.

Any combination of sequences and labels can be passed and each combination will be posed as a premise/hypothesis pair and passed to the pretrained model. Then, the logit for entailment is taken as the logit for the candidate label being valid. Any NLI model can be used, but the id of the entailment label must be included in the model config’s :attr:~transformers.PretrainedConfig.label2id.

Example:

Copied
>>> from transformers import pipeline

>>> oracle = pipeline(model="facebook/bart-large-mnli")
>>> oracle(
... "I have a problem with my iphone that needs to be resolved asap!!",
... candidate\_labels=["urgent", "not urgent", "phone", "tablet", "computer"],
... )
{'sequence': 'I have a problem with my iphone that needs to be resolved asap!!', 'labels': ['urgent', 'phone', 'computer', 'not urgent', 'tablet'], 'scores': [0.504, 0.479, 0.013, 0.003, 0.002]}

>>> oracle(
... "I have a problem with my iphone that needs to be resolved asap!!",
... candidate\_labels=["english", "german"],
... )
{'sequence': 'I have a problem with my iphone that needs to be resolved asap!!', 'labels': ['english', 'german'], 'scores': [0.814, 0.186]}

Learn more about the basics of using a pipeline in the pipeline tutorial

This NLI pipeline can currently be loaded from pipeline() using the following task identifier: "zero-shot-classification".

The models that this pipeline can use are models that have been fine-tuned on an NLI task. See the up-to-date list of available models on huggingface.co/models.

\_\_call\_\_
<
source
>
( sequences: typing.Union[str, typing.List[str]]\*args\*\*kwargs ) → A dict or a list of dict

Expand 4 parameters

Parameters

sequences (str or List[str]) — The sequence(s) to classify, will be truncated if the model input is too large.
candidate\_labels (str or List[str]) — The set of possible class labels to classify each sequence into. Can be a single label, a string of comma-separated labels, or a list of labels.
hypothesis\_template (str, optional, defaults to "This example is {}.") — The template used to turn each label into an NLI-style hypothesis. This template must include a {} or similar syntax for the candidate label to be inserted into the template. For example, the default template is "This example is {}." With the candidate label "sports", this would be fed into the model like " sequence to classify  This example is sports . ". The default template works well in many cases, but it may be worthwhile to experiment with different templates depending on the task setting.
multi\_label (bool, optional, defaults to False) — Whether or not multiple candidate labels can be true. If False, the scores are normalized such that the sum of the label likelihoods for each sequence is 1. If True, the labels are considered independent and probabilities are normalized for each candidate by doing a softmax of the entailment score vs. the contradiction score.

Returns

A dict or a list of dict

Each result comes as a dictionary with the following keys:

sequence (str) — The sequence for which this is the output.
labels (List[str]) — The labels sorted by order of likelihood.
scores (List[float]) — The probabilities for each of the labels.

Classify the sequence(s) given as inputs. See the ZeroShotClassificationPipeline documentation for more information.

Multimodal

Pipelines available for multimodal tasks include the following.

DocumentQuestionAnsweringPipeline
<
source
>
( \*args\*\*kwargs )

Expand 10 parameters

Parameters

model (PreTrainedModel or TFPreTrainedModel) — The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from PreTrainedModel for PyTorch and TFPreTrainedModel for TensorFlow.
tokenizer (PreTrainedTokenizer) — The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from PreTrainedTokenizer.
modelcard (str or ModelCard, optional) — Model card attributed to the model for this pipeline.
framework (str, optional) — The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be installed.

If no framework is specified, will default to the one currently installed. If no framework is specified and both frameworks are installed, will default to the framework of the model, or to PyTorch if no model is provided.

task (str, defaults to "") — A task-identifier for the pipeline.
num\_workers (int, optional, defaults to 8) — When the pipeline will use DataLoader (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.
args\_parser (ArgumentHandler, optional) — Reference to the object in charge of parsing supplied pipeline parameters.
device (int, optional, defaults to -1) — Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native torch.device or a str too.
binary\_output (bool, optional, defaults to False) — Flag indicating if the output the pipeline should happen in a binary format (i.e., pickle) or as raw text.

Document Question Answering pipeline using any AutoModelForDocumentQuestionAnswering. The inputs/outputs are similar to the (extractive) question answering pipeline; however, the pipeline takes an image (and optional OCR’d words/boxes) as input instead of text context.

Example:

Copied
>>> from transformers import pipeline

>>> document\_qa = pipeline(model="impira/layoutlm-document-qa")
>>> document\_qa(
... image="https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png",
... question="What is the invoice number?",
... )
[{'score': 0.425, 'answer': 'us-001', 'start': 16, 'end': 16}]

Learn more about the basics of using a pipeline in the pipeline tutorial

This document question answering pipeline can currently be loaded from pipeline() using the following task identifier: "document-question-answering".

The models that this pipeline can use are models that have been fine-tuned on a document question answering task. See the up-to-date list of available models on huggingface.co/models.

\_\_call\_\_
<
source
>
( image: typing.Union[ForwardRef('Image.Image'), str]question: typing.Optional[str] = Noneword\_boxes: typing.Tuple[str, typing.List[float]] = None\*\*kwargs ) → A dict or a list of dict

Expand 12 parameters

Parameters

image (str or PIL.Image) — The pipeline handles three types of images:

A string containing a http link pointing to an image
An image loaded in PIL directly

The pipeline accepts either a single image or a batch of images. If given a single image, it can be broadcasted to multiple questions.

question (str) — A question to ask of the document.
word\_boxes (List[str, Tuple[float, float, float, float]], optional) — A list of words and bounding boxes (normalized 0->1000). If you provide this optional input, then the pipeline will use these words and boxes instead of running OCR on the image to derive them for models that need them (e.g. LayoutLM). This allows you to reuse OCR’d results across many invocations of the pipeline without having to re-run it each time.
top\_k (int, optional, defaults to 1) — The number of answers to return (will be chosen by order of likelihood). Note that we return less than top\_k answers if there are not enough options available within the context.
doc\_stride (int, optional, defaults to 128) — If the words in the document are too long to fit with the question for the model, it will be split in several chunks with some overlap. This argument controls the size of that overlap.
max\_answer\_len (int, optional, defaults to 15) — The maximum length of predicted answers (e.g., only answers with a shorter length are considered).
lang (str, optional) — Language to use while running OCR. Defaults to english.
tesseract\_config (str, optional) — Additional flags to pass to tesseract while running OCR.
timeout (float, optional, defaults to None) — The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and the call may block forever.

Returns

A dict or a list of dict

Each result comes as a dictionary with the following keys:

score (float) — The probability associated to the answer.
start (int) — The start word index of the answer (in the OCR’d version of the input or provided word\_boxes).
answer (str) — The answer to the question.
words (list[int]) — The index of each word/box pair that is in the answer

Answer the question(s) given as inputs by using the document(s). A document is defined as an image and an optional list of (word, box) tuples which represent the text in the document. If the word\_boxes are not provided, it will use the Tesseract OCR engine (if available) to extract the words and boxes automatically for LayoutLM-like models which require them as input. For Donut, no OCR is run.

You can invoke the pipeline several ways:

pipeline(image=image, question=question)
pipeline([{"image": image, "question": question}])
FeatureExtractionPipeline
<
source
>
( model: typing.Union[ForwardRef('PreTrainedModel'), ForwardRef('TFPreTrainedModel')]tokenizer: typing.Optional[transformers.tokenization\_utils.PreTrainedTokenizer] = Nonefeature\_extractor: typing.Optional[ForwardRef('SequenceFeatureExtractor')] = Noneimage\_processor: typing.Optional[transformers.image\_processing\_utils.BaseImageProcessor] = Nonemodelcard: typing.Optional[transformers.modelcard.ModelCard] = Noneframework: typing.Optional[str] = Nonetask: str = ''args\_parser: ArgumentHandler = Nonedevice: typing.Union[int, ForwardRef('torch.device')] = Nonetorch\_dtype: typing.Union[str, ForwardRef('torch.dtype'), NoneType] = Nonebinary\_output: bool = False\*\*kwargs )

Expand 9 parameters

Parameters

model (PreTrainedModel or TFPreTrainedModel) — The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from PreTrainedModel for PyTorch and TFPreTrainedModel for TensorFlow.
tokenizer (PreTrainedTokenizer) — The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from PreTrainedTokenizer.
modelcard (str or ModelCard, optional) — Model card attributed to the model for this pipeline.
framework (str, optional) — The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be installed.

If no framework is specified, will default to the one currently installed. If no framework is specified and both frameworks are installed, will default to the framework of the model, or to PyTorch if no model is provided.

return\_tensors (bool, optional) — If True, returns a tensor according to the specified framework, otherwise returns a list.
task (str, defaults to "") — A task-identifier for the pipeline.
args\_parser (ArgumentHandler, optional) — Reference to the object in charge of parsing supplied pipeline parameters.
device (int, optional, defaults to -1) — Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id.
tokenize\_kwargs (dict, optional) — Additional dictionary of keyword arguments passed along to the tokenizer.

Feature extraction pipeline using no model head. This pipeline extracts the hidden states from the base transformer, which can be used as features in downstream tasks.

Example:

Copied
>>> from transformers import pipeline

>>> extractor = pipeline(model="bert-base-uncased", task="feature-extraction")
>>> result = extractor("This is a simple test.", return\_tensors=True)
>>> result.shape # This is a tensor of shape [1, sequence\_lenth, hidden\_dimension] representing the input string.
torch.Size([1, 8, 768])

Learn more about the basics of using a pipeline in the pipeline tutorial

This feature extraction pipeline can currently be loaded from pipeline() using the task identifier: "feature-extraction".

All models may be used for this pipeline. See a list of all models, including community-contributed models on huggingface.co/models.

\_\_call\_\_
<
source
>
( \*args\*\*kwargs ) → A nested list of float

Parameters

args (str or List[str]) — One or several texts (or one list of texts) to get the features of.

Returns

A nested list of float

The features computed by the model.

Extract the features of the input(s).

ImageToTextPipeline
<
source
>
( \*args\*\*kwargs )

Expand 10 parameters

Parameters

model (PreTrainedModel or TFPreTrainedModel) — The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from PreTrainedModel for PyTorch and TFPreTrainedModel for TensorFlow.
tokenizer (PreTrainedTokenizer) — The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from PreTrainedTokenizer.
modelcard (str or ModelCard, optional) — Model card attributed to the model for this pipeline.
framework (str, optional) — The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be installed.

If no framework is specified, will default to the one currently installed. If no framework is specified and both frameworks are installed, will default to the framework of the model, or to PyTorch if no model is provided.

task (str, defaults to "") — A task-identifier for the pipeline.
num\_workers (int, optional, defaults to 8) — When the pipeline will use DataLoader (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.
args\_parser (ArgumentHandler, optional) — Reference to the object in charge of parsing supplied pipeline parameters.
device (int, optional, defaults to -1) — Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native torch.device or a str too.
binary\_output (bool, optional, defaults to False) — Flag indicating if the output the pipeline should happen in a binary format (i.e., pickle) or as raw text.

Image To Text pipeline using a AutoModelForVision2Seq. This pipeline predicts a caption for a given image.

Example:

Copied
>>> from transformers import pipeline

>>> captioner = pipeline(model="ydshieh/vit-gpt2-coco-en")
>>> captioner("https://huggingface.co/datasets/Narsil/image\_dummy/raw/main/parrots.png")
[{'generated\_text': 'two birds are standing next to each other '}]

Learn more about the basics of using a pipeline in the pipeline tutorial

This image to text pipeline can currently be loaded from pipeline() using the following task identifier: “image-to-text”.

See the list of available models on huggingface.co/models.

\_\_call\_\_
<
source
>
( images: typing.Union[str, typing.List[str], ForwardRef('Image.Image'), typing.List[ForwardRef('Image.Image')]]\*\*kwargs ) → A list or a list of list of dict

Expand 4 parameters

Parameters

images (str, List[str], PIL.Image or List[PIL.Image]) — The pipeline handles three types of images:

A string containing a HTTP(s) link pointing to an image
An image loaded in PIL directly

The pipeline accepts either a single image or a batch of images.

max\_new\_tokens (int, optional) — The amount of maximum tokens to generate. By default it will use generate default.
generate\_kwargs (Dict, optional) — Pass it to send all of these arguments directly to generate allowing full control of this function.
timeout (float, optional, defaults to None) — The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and the call may block forever.

Returns

A list or a list of list of dict

Each result comes as a dictionary with the following key:

generated\_text (str) — The generated text.

Assign labels to the image(s) passed as inputs.

MaskGenerationPipeline
class transformers.MaskGenerationPipeline
<
source
>
( \*\*kwargs )

Expand 16 parameters

Parameters

model (PreTrainedModel or TFPreTrainedModel) — The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from PreTrainedModel for PyTorch and TFPreTrainedModel for TensorFlow.
tokenizer (PreTrainedTokenizer) — The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from PreTrainedTokenizer.
feature\_extractor (SequenceFeatureExtractor) — The feature extractor that will be used by the pipeline to encode the input.
points\_per\_batch (optional, int, default to 64) — Sets the number of points run simultaneously by the model. Higher numbers may be faster but use more GPU memory.
output\_bboxes\_mask (bool, optional, default to False) — Whether or not to output the bounding box predictions.
model (PreTrainedModel or TFPreTrainedModel) — The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from PreTrainedModel for PyTorch and TFPreTrainedModel for TensorFlow.
tokenizer (PreTrainedTokenizer) — The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from PreTrainedTokenizer.
modelcard (str or ModelCard, optional) — Model card attributed to the model for this pipeline.
framework (str, optional) — The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be installed.

If no framework is specified, will default to the one currently installed. If no framework is specified and both frameworks are installed, will default to the framework of the model, or to PyTorch if no model is provided.

task (str, defaults to "") — A task-identifier for the pipeline.
num\_workers (int, optional, defaults to 8) — When the pipeline will use DataLoader (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.
args\_parser (ArgumentHandler, optional) — Reference to the object in charge of parsing supplied pipeline parameters.
device (int, optional, defaults to -1) — Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native torch.device or a str too.
binary\_output (bool, optional, defaults to False) — Flag indicating if the output the pipeline should happen in a binary format (i.e., pickle) or as raw text.

Automatic mask generation for images using SamForMaskGeneration. This pipeline predicts binary masks for an image, given an image. It is a ChunkPipeline because you can seperate the points in a mini-batch in order to avoid OOM issues. Use the points\_per\_batch argument to control the number of points that will be processed at the same time. Default is 64.

The pipeline works in 3 steps:

preprocess: A grid of 1024 points evenly separated is generated along with bounding boxes and point labels. For more details on how the points and bounding boxes are created, check the \_generate\_crop\_boxes function. The image is also preprocessed using the image\_processor. This function yields a minibatch of points\_per\_batch.

forward: feeds the outputs of preprocess to the model. The image embedding is computed only once. Calls both self.model.get\_image\_embeddings and makes sure that the gradients are not computed, and the tensors and models are on the same device.

postprocess: The most important part of the automatic mask generation happens here. Three steps are induced:

image\_processor.postprocess\_masks (run on each minibatch loop): takes in the raw output masks, resizes them according to the image size, and transforms there to binary masks.
image\_processor.postprocess\_masks\_for\_amg applies the NSM on the mask to only keep relevant ones.

Example:

Copied
>>> from transformers import pipeline

>>> generator = pipeline(model="facebook/sam-vit-base", task="mask-generation")
>>> outputs = generator(
... "http://images.cocodataset.org/val2017/000000039769.jpg",
... )

>>> outputs = generator(
... "https://huggingface.co/datasets/Narsil/image\_dummy/raw/main/parrots.png", points\_per\_batch=128
... )

Learn more about the basics of using a pipeline in the pipeline tutorial

This segmentation pipeline can currently be loaded from pipeline() using the following task identifier: "mask-generation".

See the list of available models on huggingface.co/models.

\_\_call\_\_
<
source
>
( image\*argsnum\_workers = Nonebatch\_size = None\*\*kwargs ) → Dict

Expand 10 parameters

Parameters

inputs (np.ndarray or bytes or str or dict) — Image or list of images.
mask\_threshold (float, optional, defaults to 0.0) — Threshold to use when turning the predicted masks into binary values.
crops\_n\_layers (int, optional, defaults to 0) — If crops\_n\_layers>0, mask prediction will be run again on crops of the image. Sets the number of layers to run, where each layer has 2\*\*i\_layer number of image crops.
crop\_overlap\_ratio (float, optional, defaults to 512 / 1500) — Sets the degree to which crops overlap. In the first crop layer, crops will overlap by this fraction of the image length. Later layers with more crops scale down this overlap.
crop\_n\_points\_downscale\_factor (int, optional, defaults to 1) — The number of points-per-side sampled in layer n is scaled down by crop\_n\_points\_downscale\_factor\*\*n.
timeout (float, optional, defaults to None) — The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and the call may block forever.

Returns

Dict

A dictionary with the following keys:

mask (PIL.Image) — A binary mask of the detected object as a PIL Image of shape (width, height) of the original image. Returns a mask filled with zeros if no object is found.
score (optional float) — Optionally, when the model is capable of estimating a confidence of the “object” described by the label and the mask.

Generates binary segmentation masks

VisualQuestionAnsweringPipeline
<
source
>
( \*args\*\*kwargs )

Expand 10 parameters

Parameters

model (PreTrainedModel or TFPreTrainedModel) — The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from PreTrainedModel for PyTorch and TFPreTrainedModel for TensorFlow.
tokenizer (PreTrainedTokenizer) — The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from PreTrainedTokenizer.
modelcard (str or ModelCard, optional) — Model card attributed to the model for this pipeline.
framework (str, optional) — The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be installed.

If no framework is specified, will default to the one currently installed. If no framework is specified and both frameworks are installed, will default to the framework of the model, or to PyTorch if no model is provided.

task (str, defaults to "") — A task-identifier for the pipeline.
num\_workers (int, optional, defaults to 8) — When the pipeline will use DataLoader (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.
args\_parser (ArgumentHandler, optional) — Reference to the object in charge of parsing supplied pipeline parameters.
device (int, optional, defaults to -1) — Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native torch.device or a str too.
binary\_output (bool, optional, defaults to False) — Flag indicating if the output the pipeline should happen in a binary format (i.e., pickle) or as raw text.

Visual Question Answering pipeline using a AutoModelForVisualQuestionAnswering. This pipeline is currently only available in PyTorch.

Example:

Copied
>>> from transformers import pipeline

>>> oracle = pipeline(model="dandelin/vilt-b32-finetuned-vqa")
>>> image\_url = "https://huggingface.co/datasets/Narsil/image\_dummy/raw/main/lena.png"
>>> oracle(question="What is she wearing ?", image=image\_url)
[{'score': 0.948, 'answer': 'hat'}, {'score': 0.009, 'answer': 'fedora'}, {'score': 0.003, 'answer': 'clothes'}, {'score': 0.003, 'answer': 'sun hat'}, {'score': 0.002, 'answer': 'nothing'}]

>>> oracle(question="What is she wearing ?", image=image\_url, top\_k=1)
[{'score': 0.948, 'answer': 'hat'}]

>>> oracle(question="Is this a person ?", image=image\_url, top\_k=1)
[{'score': 0.993, 'answer': 'yes'}]

>>> oracle(question="Is this a man ?", image=image\_url, top\_k=1)
[{'score': 0.996, 'answer': 'no'}]

Learn more about the basics of using a pipeline in the pipeline tutorial

This visual question answering pipeline can currently be loaded from pipeline() using the following task identifiers: "visual-question-answering", "vqa".

The models that this pipeline can use are models that have been fine-tuned on a visual question answering task. See the up-to-date list of available models on huggingface.co/models.

\_\_call\_\_
<
source
>
( image: typing.Union[ForwardRef('Image.Image'), str]question: str = None\*\*kwargs ) → A dictionary or a list of dictionaries containing the result. The dictionaries contain the following keys

Expand 4 parameters

Parameters

image (str, List[str], PIL.Image or List[PIL.Image]) — The pipeline handles three types of images:

A string containing a http link pointing to an image
An image loaded in PIL directly

The pipeline accepts either a single image or a batch of images. If given a single image, it can be broadcasted to multiple questions.

question (str, List[str]) — The question(s) asked. If given a single question, it can be broadcasted to multiple images.
top\_k (int, optional, defaults to 5) — The number of top labels that will be returned by the pipeline. If the provided number is higher than the number of labels available in the model configuration, it will default to the number of labels.
timeout (float, optional, defaults to None) — The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and the call may block forever.

Returns

A dictionary or a list of dictionaries containing the result. The dictionaries contain the following keys

label (str) — The label identified by the model.

Answers open-ended questions about images. The pipeline accepts several types of inputs which are detailed below:

pipeline(image=image, question=question)
Parent class: Pipeline
class transformers.Pipeline
<
source
>
( model: typing.Union[ForwardRef('PreTrainedModel'), ForwardRef('TFPreTrainedModel')]tokenizer: typing.Optional[transformers.tokenization\_utils.PreTrainedTokenizer] = Nonefeature\_extractor: typing.Optional[ForwardRef('SequenceFeatureExtractor')] = Noneimage\_processor: typing.Optional[transformers.image\_processing\_utils.BaseImageProcessor] = Nonemodelcard: typing.Optional[transformers.modelcard.ModelCard] = Noneframework: typing.Optional[str] = Nonetask: str = ''args\_parser: ArgumentHandler = Nonedevice: typing.Union[int, ForwardRef('torch.device')] = Nonetorch\_dtype: typing.Union[str, ForwardRef('torch.dtype'), NoneType] = Nonebinary\_output: bool = False\*\*kwargs )

Expand 10 parameters

Parameters

model (PreTrainedModel or TFPreTrainedModel) — The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from PreTrainedModel for PyTorch and TFPreTrainedModel for TensorFlow.
tokenizer (PreTrainedTokenizer) — The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from PreTrainedTokenizer.
modelcard (str or ModelCard, optional) — Model card attributed to the model for this pipeline.
framework (str, optional) — The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be installed.

If no framework is specified, will default to the one currently installed. If no framework is specified and both frameworks are installed, will default to the framework of the model, or to PyTorch if no model is provided.

task (str, defaults to "") — A task-identifier for the pipeline.
num\_workers (int, optional, defaults to 8) — When the pipeline will use DataLoader (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.
args\_parser (ArgumentHandler, optional) — Reference to the object in charge of parsing supplied pipeline parameters.
device (int, optional, defaults to -1) — Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native torch.device or a str too.
binary\_output (bool, optional, defaults to False) — Flag indicating if the output the pipeline should happen in a binary format (i.e., pickle) or as raw text.

The Pipeline class is the class from which all pipelines inherit. Refer to this class for methods shared across different pipelines.

Base class implementing pipelined operations. Pipeline workflow is defined as a sequence of the following operations:

Input -> Tokenization -> Model Inference -> Post-Processing (task dependent) -> Output

Pipeline supports running on CPU or GPU through the device argument (see below).

Some pipeline, like for instance FeatureExtractionPipeline ('feature-extraction') output large tensor object as nested-lists. In order to avoid dumping such large structure as textual data we provide the binary\_output constructor argument. If set to True, the output will be stored in the pickle format.

check\_model\_type
<
source
>
( supported\_models: typing.Union[typing.List[str], dict] )

Parameters

supported\_models (List[str] or dict) — The list of models supported by the pipeline, or a dictionary with model class values.

Check if the model class is in supported by the pipeline.

device\_placement
<
source
>
( )

Context Manager allowing tensor allocation on the user-specified device in framework agnostic way.

Examples:

Copied
# Explicitly ask for tensor allocation on CUDA device :0
pipe = pipeline(..., device=0)
 # Every framework specific tensor allocation will be done on the request device
 output = pipe(...)
ensure\_tensor\_on\_device
<
source
>
( \*\*inputs ) → Dict[str, torch.Tensor]

Parameters

inputs (keyword arguments that should be torch.Tensor, the rest is ignored) — The tensors to place on self.device.
Recursive on lists only. —

Returns

Dict[str, torch.Tensor]

The same as inputs but on the proper device.

Ensure PyTorch tensors are on the specified device.

postprocess
<
source
>
( model\_outputs: ModelOutput\*\*postprocess\_parameters: typing.Dict )

Postprocess will receive the raw outputs of the \_forward method, generally tensors, and reformat them into something more friendly. Generally it will output a list or a dict or results (containing just strings and numbers).

predict
<
source
>
( X )

Scikit / Keras interface to transformers’ pipelines. This method will forward to call().

preprocess
<
source
>
( input\_: typing.Any\*\*preprocess\_parameters: typing.Dict )

Preprocess will take the input\_ of a specific pipeline and return a dictionary of everything necessary for \_forward to run properly. It should contain at least one tensor, but might have arbitrary other items.

save\_pretrained
<
source
>
( save\_directory: strsafe\_serialization: bool = True )

Parameters

save\_directory (str) — A path to the directory where to saved. It will be created if it doesn’t exist.
safe\_serialization (str) — Whether to save the model using safetensors or the traditional way for PyTorch or Tensorflow.

Save the pipeline’s model and tokenizer.