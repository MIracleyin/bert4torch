# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# When adding a new object to this init, remember to add it twice: once inside the `_import_structure` dictionary and
# once inside the `if TYPE_CHECKING` branch. The `TYPE_CHECKING` should have import statements as usual, but they are
# only there for type checking. The `_import_structure` is a dictionary submodule to list of object names, and is used
# to defer the actual importing for when the objects are requested. This way `import transformers` provides the names
# in the namespace without actually importing anything (and especially none of the backends).

__version__ = "4.4.0.dev0"

# Work around to update TensorFlow's absl.logging threshold which alters the
# default Python logging output behavior when present.
# see: https://github.com/abseil/abseil-py/issues/99
# and: https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-500369493
try:
    import absl.logging
except ImportError:
    pass
else:
    absl.logging.set_verbosity("info")
    absl.logging.set_stderrthreshold("info")
    absl.logging._warn_preinit_stderr = False

from typing import TYPE_CHECKING

# Check the dependencies satisfy the minimal versions required.
from .file_utils import (
    _BaseLazyModule,
    is_flax_available,
    is_sentencepiece_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)
from .utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Base objects, independent of any specific backend
_import_structure = {
    "configuration_utils": ["PretrainedConfig"],
    "data": [
        "DataProcessor",
        "InputExample",
        "InputFeatures",
        "SingleSentenceClassificationProcessor",
        "SquadExample",
        "SquadFeatures",
        "SquadV1Processor",
        "SquadV2Processor",
        "glue_compute_metrics",
        "glue_convert_examples_to_features",
        "glue_output_modes",
        "glue_processors",
        "glue_tasks_num_labels",
        "squad_convert_examples_to_features",
        "xnli_compute_metrics",
        "xnli_output_modes",
        "xnli_processors",
        "xnli_tasks_num_labels",
    ],
    "file_utils": [
        "CONFIG_NAME",
        "MODEL_CARD_NAME",
        "PYTORCH_PRETRAINED_BERT_CACHE",
        "PYTORCH_TRANSFORMERS_CACHE",
        "SPIECE_UNDERLINE",
        "TF2_WEIGHTS_NAME",
        "TF_WEIGHTS_NAME",
        "TRANSFORMERS_CACHE",
        "WEIGHTS_NAME",
        "add_end_docstrings",
        "add_start_docstrings",
        "cached_path",
        "is_apex_available",
        "is_datasets_available",
        "is_faiss_available",
        "is_flax_available",
        "is_psutil_available",
        "is_py3nvml_available",
        "is_sentencepiece_available",
        "is_sklearn_available",
        "is_tf_available",
        "is_tokenizers_available",
        "is_torch_available",
        "is_torch_tpu_available",
    ],
    "hf_argparser": ["HfArgumentParser"],
    "integrations": [
        "is_comet_available",
        "is_optuna_available",
        "is_ray_available",
        "is_ray_tune_available",
        "is_tensorboard_available",
        "is_wandb_available",
    ],
    "modelcard": ["ModelCard"],
    "modeling_tf_pytorch_utils": [
        "convert_tf_weight_name_to_pt_weight_name",
        "load_pytorch_checkpoint_in_tf2_model",
        "load_pytorch_model_in_tf2_model",
        "load_pytorch_weights_in_tf2_model",
        "load_tf2_checkpoint_in_pytorch_model",
        "load_tf2_model_in_pytorch_model",
        "load_tf2_weights_in_pytorch_model",
    ],
    "models": [],
    # Models
    "models.wav2vec2": ["WAV_2_VEC_2_PRETRAINED_CONFIG_ARCHIVE_MAP", "Wav2Vec2Config", "Wav2Vec2Tokenizer"],
    "models.convbert": ["CONVBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "ConvBertConfig", "ConvBertTokenizer"],
    "models.albert": ["ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "AlbertConfig"],
    "models.auto": [
        "ALL_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "CONFIG_MAPPING",
        "MODEL_NAMES_MAPPING",
        "TOKENIZER_MAPPING",
        "AutoConfig",
        "AutoTokenizer",
    ],
    "models.bart": ["BartConfig", "BartTokenizer"],
    "models.barthez": [],
    "models.bert": [
        "BERT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "BasicTokenizer",
        "BertConfig",
        "BertTokenizer",
        "WordpieceTokenizer",
    ],
    "models.bert_generation": ["BertGenerationConfig"],
    "models.bert_japanese": ["BertJapaneseTokenizer", "CharacterTokenizer", "MecabTokenizer"],
    "models.bertweet": ["BertweetTokenizer"],
    "models.blenderbot": ["BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP", "BlenderbotConfig", "BlenderbotTokenizer"],
    "models.blenderbot_small": [
        "BLENDERBOT_SMALL_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "BlenderbotSmallConfig",
        "BlenderbotSmallTokenizer",
    ],
    "models.camembert": ["CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "CamembertConfig"],
    "models.ctrl": ["CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP", "CTRLConfig", "CTRLTokenizer"],
    "models.deberta": ["DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP", "DebertaConfig", "DebertaTokenizer"],
    "models.deberta_v2": ["DEBERTA_V2_PRETRAINED_CONFIG_ARCHIVE_MAP", "DebertaV2Config", "DebertaV2Tokenizer"],
    "models.distilbert": ["DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "DistilBertConfig", "DistilBertTokenizer"],
    "models.dpr": [
        "DPR_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "DPRConfig",
        "DPRContextEncoderTokenizer",
        "DPRQuestionEncoderTokenizer",
        "DPRReaderOutput",
        "DPRReaderTokenizer",
    ],
    "models.electra": ["ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP", "ElectraConfig", "ElectraTokenizer"],
    "models.encoder_decoder": ["EncoderDecoderConfig"],
    "models.flaubert": ["FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "FlaubertConfig", "FlaubertTokenizer"],
    "models.fsmt": ["FSMT_PRETRAINED_CONFIG_ARCHIVE_MAP", "FSMTConfig", "FSMTTokenizer"],
    "models.funnel": ["FUNNEL_PRETRAINED_CONFIG_ARCHIVE_MAP", "FunnelConfig", "FunnelTokenizer"],
    "models.gpt2": ["GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP", "GPT2Config", "GPT2Tokenizer"],
    "models.herbert": ["HerbertTokenizer"],
    "models.layoutlm": ["LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP", "LayoutLMConfig", "LayoutLMTokenizer"],
    "models.led": ["LED_PRETRAINED_CONFIG_ARCHIVE_MAP", "LEDConfig", "LEDTokenizer"],
    "models.longformer": ["LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "LongformerConfig", "LongformerTokenizer"],
    "models.lxmert": ["LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "LxmertConfig", "LxmertTokenizer"],
    "models.marian": ["MarianConfig"],
    "models.mbart": ["MBartConfig"],
    "models.mmbt": ["MMBTConfig"],
    "models.mobilebert": ["MOBILEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "MobileBertConfig", "MobileBertTokenizer"],
    "models.mpnet": ["MPNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "MPNetConfig", "MPNetTokenizer"],
    "models.mt5": ["MT5Config"],
    "models.openai": ["OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP", "OpenAIGPTConfig", "OpenAIGPTTokenizer"],
    "models.pegasus": ["PegasusConfig"],
    "models.phobert": ["PhobertTokenizer"],
    "models.prophetnet": ["PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "ProphetNetConfig", "ProphetNetTokenizer"],
    "models.rag": ["RagConfig", "RagRetriever", "RagTokenizer"],
    "models.reformer": ["REFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "ReformerConfig"],
    "models.retribert": ["RETRIBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "RetriBertConfig", "RetriBertTokenizer"],
    "models.roberta": ["ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP", "RobertaConfig", "RobertaTokenizer"],
    "models.squeezebert": ["SQUEEZEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "SqueezeBertConfig", "SqueezeBertTokenizer"],
    "models.t5": ["T5_PRETRAINED_CONFIG_ARCHIVE_MAP", "T5Config"],
    "models.tapas": ["TAPAS_PRETRAINED_CONFIG_ARCHIVE_MAP", "TapasConfig", "TapasTokenizer"],
    "models.transfo_xl": [
        "TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "TransfoXLConfig",
        "TransfoXLCorpus",
        "TransfoXLTokenizer",
    ],
    "models.xlm": ["XLM_PRETRAINED_CONFIG_ARCHIVE_MAP", "XLMConfig", "XLMTokenizer"],
    "models.xlm_prophetnet": ["XLM_PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "XLMProphetNetConfig"],
    "models.xlm_roberta": ["XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP", "XLMRobertaConfig"],
    "models.xlnet": ["XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "XLNetConfig"],
    "pipelines": [
        "Conversation",
        "ConversationalPipeline",
        "CsvPipelineDataFormat",
        "FeatureExtractionPipeline",
        "FillMaskPipeline",
        "JsonPipelineDataFormat",
        "NerPipeline",
        "PipedPipelineDataFormat",
        "Pipeline",
        "PipelineDataFormat",
        "QuestionAnsweringPipeline",
        "SummarizationPipeline",
        "TableQuestionAnsweringPipeline",
        "Text2TextGenerationPipeline",
        "TextClassificationPipeline",
        "TextGenerationPipeline",
        "TokenClassificationPipeline",
        "TranslationPipeline",
        "ZeroShotClassificationPipeline",
        "pipeline",
    ],
    "tokenization_utils": ["PreTrainedTokenizer"],
    "tokenization_utils_base": [
        "AddedToken",
        "BatchEncoding",
        "CharSpan",
        "PreTrainedTokenizerBase",
        "SpecialTokensMixin",
        "TensorType",
        "TokenSpan",
    ],
    "trainer_callback": [
        "DefaultFlowCallback",
        "EarlyStoppingCallback",
        "PrinterCallback",
        "ProgressCallback",
        "TrainerCallback",
        "TrainerControl",
        "TrainerState",
    ],
    "trainer_utils": ["EvalPrediction", "EvaluationStrategy", "SchedulerType", "set_seed"],
    "training_args": ["TrainingArguments"],
    "training_args_seq2seq": ["Seq2SeqTrainingArguments"],
    "training_args_tf": ["TFTrainingArguments"],
    "utils": ["logging"],
}

# sentencepiece-backed objects
if is_sentencepiece_available():
    _import_structure["models.albert"].append("AlbertTokenizer")
    _import_structure["models.barthez"].append("BarthezTokenizer")
    _import_structure["models.bert_generation"].append("BertGenerationTokenizer")
    _import_structure["models.camembert"].append("CamembertTokenizer")
    _import_structure["models.marian"].append("MarianTokenizer")
    _import_structure["models.mbart"].append("MBartTokenizer")
    _import_structure["models.mbart"].append("MBart50Tokenizer")
    _import_structure["models.mt5"].append("MT5Tokenizer")
    _import_structure["models.pegasus"].append("PegasusTokenizer")
    _import_structure["models.reformer"].append("ReformerTokenizer")
    _import_structure["models.t5"].append("T5Tokenizer")
    _import_structure["models.xlm_prophetnet"].append("XLMProphetNetTokenizer")
    _import_structure["models.xlm_roberta"].append("XLMRobertaTokenizer")
    _import_structure["models.xlnet"].append("XLNetTokenizer")
else:
    from .utils import dummy_sentencepiece_objects

    _import_structure["utils.dummy_sentencepiece_objects"] = [
        name for name in dir(dummy_sentencepiece_objects) if not name.startswith("_")
    ]

# tokenziers-backed objects
if is_tokenizers_available():
    # Fast tokenizers
    _import_structure["models.convbert"].append("ConvBertTokenizerFast")
    _import_structure["models.albert"].append("AlbertTokenizerFast")
    _import_structure["models.bart"].append("BartTokenizerFast")
    _import_structure["models.barthez"].append("BarthezTokenizerFast")
    _import_structure["models.bert"].append("BertTokenizerFast")
    _import_structure["models.camembert"].append("CamembertTokenizerFast")
    _import_structure["models.distilbert"].append("DistilBertTokenizerFast")
    _import_structure["models.dpr"].extend(
        ["DPRContextEncoderTokenizerFast", "DPRQuestionEncoderTokenizerFast", "DPRReaderTokenizerFast"]
    )
    _import_structure["models.electra"].append("ElectraTokenizerFast")
    _import_structure["models.funnel"].append("FunnelTokenizerFast")
    _import_structure["models.gpt2"].append("GPT2TokenizerFast")
    _import_structure["models.herbert"].append("HerbertTokenizerFast")
    _import_structure["models.layoutlm"].append("LayoutLMTokenizerFast")
    _import_structure["models.led"].append("LEDTokenizerFast")
    _import_structure["models.longformer"].append("LongformerTokenizerFast")
    _import_structure["models.lxmert"].append("LxmertTokenizerFast")
    _import_structure["models.mbart"].append("MBartTokenizerFast")
    _import_structure["models.mbart"].append("MBart50TokenizerFast")
    _import_structure["models.mobilebert"].append("MobileBertTokenizerFast")
    _import_structure["models.mpnet"].append("MPNetTokenizerFast")
    _import_structure["models.mt5"].append("MT5TokenizerFast")
    _import_structure["models.openai"].append("OpenAIGPTTokenizerFast")
    _import_structure["models.pegasus"].append("PegasusTokenizerFast")
    _import_structure["models.reformer"].append("ReformerTokenizerFast")
    _import_structure["models.retribert"].append("RetriBertTokenizerFast")
    _import_structure["models.roberta"].append("RobertaTokenizerFast")
    _import_structure["models.squeezebert"].append("SqueezeBertTokenizerFast")
    _import_structure["models.t5"].append("T5TokenizerFast")
    _import_structure["models.xlm_roberta"].append("XLMRobertaTokenizerFast")
    _import_structure["models.xlnet"].append("XLNetTokenizerFast")
    _import_structure["tokenization_utils_fast"] = ["PreTrainedTokenizerFast"]

    if is_sentencepiece_available():
        _import_structure["convert_slow_tokenizer"] = ["SLOW_TO_FAST_CONVERTERS", "convert_slow_tokenizer"]
else:
    from .utils import dummy_tokenizers_objects

    _import_structure["utils.dummy_tokenizers_objects"] = [
        name for name in dir(dummy_tokenizers_objects) if not name.startswith("_")
    ]

# PyTorch-backed objects
if is_torch_available():
    _import_structure["benchmark.benchmark"] = ["PyTorchBenchmark"]
    _import_structure["benchmark.benchmark_args"] = ["PyTorchBenchmarkArguments"]
    _import_structure["data.data_collator"] = [
        "DataCollator",
        "DataCollatorForLanguageModeling",
        "DataCollatorForPermutationLanguageModeling",
        "DataCollatorForSeq2Seq",
        "DataCollatorForSOP",
        "DataCollatorForTokenClassification",
        "DataCollatorForWholeWordMask",
        "DataCollatorWithPadding",
        "default_data_collator",
    ]
    _import_structure["data.datasets"] = [
        "GlueDataset",
        "GlueDataTrainingArguments",
        "LineByLineTextDataset",
        "LineByLineWithRefDataset",
        "LineByLineWithSOPTextDataset",
        "SquadDataset",
        "SquadDataTrainingArguments",
        "TextDataset",
        "TextDatasetForNextSentencePrediction",
    ]
    _import_structure["generation_beam_search"] = ["BeamScorer", "BeamSearchScorer"]
    _import_structure["generation_logits_process"] = [
        "HammingDiversityLogitsProcessor",
        "LogitsProcessor",
        "LogitsProcessorList",
        "LogitsWarper",
        "MinLengthLogitsProcessor",
        "NoBadWordsLogitsProcessor",
        "NoRepeatNGramLogitsProcessor",
        "PrefixConstrainedLogitsProcessor",
        "RepetitionPenaltyLogitsProcessor",
        "TemperatureLogitsWarper",
        "TopKLogitsWarper",
        "TopPLogitsWarper",
    ]
    _import_structure["generation_utils"] = ["top_k_top_p_filtering"]
    _import_structure["modeling_utils"] = ["Conv1D", "PreTrainedModel", "apply_chunking_to_forward", "prune_layer"]
    # PyTorch models structure

    _import_structure["models.wav2vec2"].extend(
        [
            "WAV_2_VEC_2_PRETRAINED_MODEL_ARCHIVE_LIST",
            "Wav2Vec2ForCTC",
            "Wav2Vec2ForMaskedLM",
            "Wav2Vec2Model",
            "Wav2Vec2PreTrainedModel",
        ]
    )
    _import_structure["models.convbert"].extend(
        [
            "CONVBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "ConvBertForMaskedLM",
            "ConvBertForMultipleChoice",
            "ConvBertForQuestionAnswering",
            "ConvBertForSequenceClassification",
            "ConvBertForTokenClassification",
            "ConvBertLayer",
            "ConvBertModel",
            "ConvBertPreTrainedModel",
            "load_tf_weights_in_convbert",
        ]
    )
    _import_structure["models.albert"].extend(
        [
            "ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "AlbertForMaskedLM",
            "AlbertForMultipleChoice",
            "AlbertForPreTraining",
            "AlbertForQuestionAnswering",
            "AlbertForSequenceClassification",
            "AlbertForTokenClassification",
            "AlbertModel",
            "AlbertPreTrainedModel",
            "load_tf_weights_in_albert",
        ]
    )
    _import_structure["models.auto"].extend(
        [
            "MODEL_FOR_CAUSAL_LM_MAPPING",
            "MODEL_FOR_MASKED_LM_MAPPING",
            "MODEL_FOR_MULTIPLE_CHOICE_MAPPING",
            "MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING",
            "MODEL_FOR_PRETRAINING_MAPPING",
            "MODEL_FOR_QUESTION_ANSWERING_MAPPING",
            "MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING",
            "MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING",
            "MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING",
            "MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING",
            "MODEL_MAPPING",
            "MODEL_WITH_LM_HEAD_MAPPING",
            "AutoModel",
            "AutoModelForCausalLM",
            "AutoModelForMaskedLM",
            "AutoModelForMultipleChoice",
            "AutoModelForNextSentencePrediction",
            "AutoModelForPreTraining",
            "AutoModelForQuestionAnswering",
            "AutoModelForSeq2SeqLM",
            "AutoModelForSequenceClassification",
            "AutoModelForTableQuestionAnswering",
            "AutoModelForTokenClassification",
            "AutoModelWithLMHead",
        ]
    )
    _import_structure["models.bart"].extend(
        [
            "BART_PRETRAINED_MODEL_ARCHIVE_LIST",
            "BartForCausalLM",
            "BartForConditionalGeneration",
            "BartForQuestionAnswering",
            "BartForSequenceClassification",
            "BartModel",
            "BartPretrainedModel",
            "PretrainedBartModel",
        ]
    )
    _import_structure["models.bert"].extend(
        [
            "BERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "BertForMaskedLM",
            "BertForMultipleChoice",
            "BertForNextSentencePrediction",
            "BertForPreTraining",
            "BertForQuestionAnswering",
            "BertForSequenceClassification",
            "BertForTokenClassification",
            "BertLayer",
            "BertLMHeadModel",
            "BertModel",
            "BertPreTrainedModel",
            "load_tf_weights_in_bert",
        ]
    )
    _import_structure["models.bert_generation"].extend(
        [
            "BertGenerationDecoder",
            "BertGenerationEncoder",
            "load_tf_weights_in_bert_generation",
        ]
    )
    _import_structure["models.blenderbot"].extend(
        [
            "BLENDERBOT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "BlenderbotForConditionalGeneration",
            "BlenderbotModel",
            "BlenderbotForCausalLM",
        ]
    )
    _import_structure["models.blenderbot_small"].extend(
        [
            "BLENDERBOT_SMALL_PRETRAINED_MODEL_ARCHIVE_LIST",
            "BlenderbotSmallForConditionalGeneration",
            "BlenderbotSmallModel",
            "BlenderbotSmallForCausalLM",
        ]
    )
    _import_structure["models.camembert"].extend(
        [
            "CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "CamembertForCausalLM",
            "CamembertForMaskedLM",
            "CamembertForMultipleChoice",
            "CamembertForQuestionAnswering",
            "CamembertForSequenceClassification",
            "CamembertForTokenClassification",
            "CamembertModel",
        ]
    )
    _import_structure["models.ctrl"].extend(
        [
            "CTRL_PRETRAINED_MODEL_ARCHIVE_LIST",
            "CTRLForSequenceClassification",
            "CTRLLMHeadModel",
            "CTRLModel",
            "CTRLPreTrainedModel",
        ]
    )
    _import_structure["models.deberta"].extend(
        [
            "DEBERTA_PRETRAINED_MODEL_ARCHIVE_LIST",
            "DebertaForSequenceClassification",
            "DebertaModel",
            "DebertaForMaskedLM",
            "DebertaPreTrainedModel",
            "DebertaForTokenClassification",
            "DebertaForQuestionAnswering",
        ]
    )
    _import_structure["models.deberta_v2"].extend(
        [
            "DEBERTA_V2_PRETRAINED_MODEL_ARCHIVE_LIST",
            "DebertaV2ForSequenceClassification",
            "DebertaV2Model",
            "DebertaV2ForMaskedLM",
            "DebertaV2PreTrainedModel",
            "DebertaV2ForTokenClassification",
            "DebertaV2ForQuestionAnswering",
        ]
    )
    _import_structure["models.distilbert"].extend(
        [
            "DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "DistilBertForMaskedLM",
            "DistilBertForMultipleChoice",
            "DistilBertForQuestionAnswering",
            "DistilBertForSequenceClassification",
            "DistilBertForTokenClassification",
            "DistilBertModel",
            "DistilBertPreTrainedModel",
        ]
    )
    _import_structure["models.dpr"].extend(
        [
            "DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "DPRContextEncoder",
            "DPRPretrainedContextEncoder",
            "DPRPretrainedQuestionEncoder",
            "DPRPretrainedReader",
            "DPRQuestionEncoder",
            "DPRReader",
        ]
    )
    _import_structure["models.electra"].extend(
        [
            "ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST",
            "ElectraForMaskedLM",
            "ElectraForMultipleChoice",
            "ElectraForPreTraining",
            "ElectraForQuestionAnswering",
            "ElectraForSequenceClassification",
            "ElectraForTokenClassification",
            "ElectraModel",
            "ElectraPreTrainedModel",
            "load_tf_weights_in_electra",
        ]
    )
    _import_structure["models.encoder_decoder"].append("EncoderDecoderModel")
    _import_structure["models.flaubert"].extend(
        [
            "FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "FlaubertForMultipleChoice",
            "FlaubertForQuestionAnswering",
            "FlaubertForQuestionAnsweringSimple",
            "FlaubertForSequenceClassification",
            "FlaubertForTokenClassification",
            "FlaubertModel",
            "FlaubertWithLMHeadModel",
        ]
    )
    _import_structure["models.fsmt"].extend(["FSMTForConditionalGeneration", "FSMTModel", "PretrainedFSMTModel"])
    _import_structure["models.funnel"].extend(
        [
            "FUNNEL_PRETRAINED_MODEL_ARCHIVE_LIST",
            "FunnelBaseModel",
            "FunnelForMaskedLM",
            "FunnelForMultipleChoice",
            "FunnelForPreTraining",
            "FunnelForQuestionAnswering",
            "FunnelForSequenceClassification",
            "FunnelForTokenClassification",
            "FunnelModel",
            "load_tf_weights_in_funnel",
        ]
    )
    _import_structure["models.gpt2"].extend(
        [
            "GPT2_PRETRAINED_MODEL_ARCHIVE_LIST",
            "GPT2DoubleHeadsModel",
            "GPT2ForSequenceClassification",
            "GPT2LMHeadModel",
            "GPT2Model",
            "GPT2PreTrainedModel",
            "load_tf_weights_in_gpt2",
        ]
    )
    _import_structure["models.layoutlm"].extend(
        [
            "LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST",
            "LayoutLMForMaskedLM",
            "LayoutLMForSequenceClassification",
            "LayoutLMForTokenClassification",
            "LayoutLMModel",
        ]
    )
    _import_structure["models.led"].extend(
        [
            "LED_PRETRAINED_MODEL_ARCHIVE_LIST",
            "LEDForConditionalGeneration",
            "LEDForQuestionAnswering",
            "LEDForSequenceClassification",
            "LEDModel",
        ]
    )
    _import_structure["models.longformer"].extend(
        [
            "LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "LongformerForMaskedLM",
            "LongformerForMultipleChoice",
            "LongformerForQuestionAnswering",
            "LongformerForSequenceClassification",
            "LongformerForTokenClassification",
            "LongformerModel",
            "LongformerSelfAttention",
        ]
    )
    _import_structure["models.lxmert"].extend(
        [
            "LxmertEncoder",
            "LxmertForPreTraining",
            "LxmertForQuestionAnswering",
            "LxmertModel",
            "LxmertPreTrainedModel",
            "LxmertVisualFeatureEncoder",
            "LxmertXLayer",
        ]
    )
    _import_structure["models.marian"].extend(["MarianModel", "MarianMTModel", "MarianForCausalLM"])
    _import_structure["models.mbart"].extend(
        [
            "MBartForCausalLM",
            "MBartForConditionalGeneration",
            "MBartForQuestionAnswering",
            "MBartForSequenceClassification",
            "MBartModel",
        ]
    )
    _import_structure["models.mmbt"].extend(["MMBTForClassification", "MMBTModel", "ModalEmbeddings"])
    _import_structure["models.mobilebert"].extend(
        [
            "MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "MobileBertForMaskedLM",
            "MobileBertForMultipleChoice",
            "MobileBertForNextSentencePrediction",
            "MobileBertForPreTraining",
            "MobileBertForQuestionAnswering",
            "MobileBertForSequenceClassification",
            "MobileBertForTokenClassification",
            "MobileBertLayer",
            "MobileBertModel",
            "MobileBertPreTrainedModel",
            "load_tf_weights_in_mobilebert",
        ]
    )
    _import_structure["models.mpnet"].extend(
        [
            "MPNET_PRETRAINED_MODEL_ARCHIVE_LIST",
            "MPNetForMaskedLM",
            "MPNetForMultipleChoice",
            "MPNetForQuestionAnswering",
            "MPNetForSequenceClassification",
            "MPNetForTokenClassification",
            "MPNetLayer",
            "MPNetModel",
            "MPNetPreTrainedModel",
        ]
    )
    _import_structure["models.mt5"].extend(["MT5EncoderModel", "MT5ForConditionalGeneration", "MT5Model"])
    _import_structure["models.openai"].extend(
        [
            "OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "OpenAIGPTDoubleHeadsModel",
            "OpenAIGPTForSequenceClassification",
            "OpenAIGPTLMHeadModel",
            "OpenAIGPTModel",
            "OpenAIGPTPreTrainedModel",
            "load_tf_weights_in_openai_gpt",
        ]
    )
    _import_structure["models.pegasus"].extend(
        ["PegasusForConditionalGeneration", "PegasusModel", "PegasusForCausalLM"]
    )
    _import_structure["models.prophetnet"].extend(
        [
            "PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST",
            "ProphetNetDecoder",
            "ProphetNetEncoder",
            "ProphetNetForCausalLM",
            "ProphetNetForConditionalGeneration",
            "ProphetNetModel",
            "ProphetNetPreTrainedModel",
        ]
    )
    _import_structure["models.rag"].extend(["RagModel", "RagSequenceForGeneration", "RagTokenForGeneration"])
    _import_structure["models.reformer"].extend(
        [
            "REFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "ReformerAttention",
            "ReformerForMaskedLM",
            "ReformerForQuestionAnswering",
            "ReformerForSequenceClassification",
            "ReformerLayer",
            "ReformerModel",
            "ReformerModelWithLMHead",
        ]
    )
    _import_structure["models.retribert"].extend(
        ["RETRIBERT_PRETRAINED_MODEL_ARCHIVE_LIST", "RetriBertModel", "RetriBertPreTrainedModel"]
    )
    _import_structure["models.roberta"].extend(
        [
            "ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST",
            "RobertaForCausalLM",
            "RobertaForMaskedLM",
            "RobertaForMultipleChoice",
            "RobertaForQuestionAnswering",
            "RobertaForSequenceClassification",
            "RobertaForTokenClassification",
            "RobertaModel",
        ]
    )
    _import_structure["models.squeezebert"].extend(
        [
            "SQUEEZEBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "SqueezeBertForMaskedLM",
            "SqueezeBertForMultipleChoice",
            "SqueezeBertForQuestionAnswering",
            "SqueezeBertForSequenceClassification",
            "SqueezeBertForTokenClassification",
            "SqueezeBertModel",
            "SqueezeBertModule",
            "SqueezeBertPreTrainedModel",
        ]
    )
    _import_structure["models.t5"].extend(
        [
            "T5_PRETRAINED_MODEL_ARCHIVE_LIST",
            "T5EncoderModel",
            "T5ForConditionalGeneration",
            "T5Model",
            "T5PreTrainedModel",
            "load_tf_weights_in_t5",
        ]
    )
    _import_structure["models.tapas"].extend(
        [
            "TAPAS_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TapasForMaskedLM",
            "TapasForQuestionAnswering",
            "TapasForSequenceClassification",
            "TapasModel",
        ]
    )
    _import_structure["models.transfo_xl"].extend(
        [
            "TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST",
            "AdaptiveEmbedding",
            "TransfoXLForSequenceClassification",
            "TransfoXLLMHeadModel",
            "TransfoXLModel",
            "TransfoXLPreTrainedModel",
            "load_tf_weights_in_transfo_xl",
        ]
    )
    _import_structure["models.xlm"].extend(
        [
            "XLM_PRETRAINED_MODEL_ARCHIVE_LIST",
            "XLMForMultipleChoice",
            "XLMForQuestionAnswering",
            "XLMForQuestionAnsweringSimple",
            "XLMForSequenceClassification",
            "XLMForTokenClassification",
            "XLMModel",
            "XLMPreTrainedModel",
            "XLMWithLMHeadModel",
        ]
    )
    _import_structure["models.xlm_prophetnet"].extend(
        [
            "XLM_PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST",
            "XLMProphetNetDecoder",
            "XLMProphetNetEncoder",
            "XLMProphetNetForCausalLM",
            "XLMProphetNetForConditionalGeneration",
            "XLMProphetNetModel",
        ]
    )
    _import_structure["models.xlm_roberta"].extend(
        [
            "XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST",
            "XLMRobertaForCausalLM",
            "XLMRobertaForMaskedLM",
            "XLMRobertaForMultipleChoice",
            "XLMRobertaForQuestionAnswering",
            "XLMRobertaForSequenceClassification",
            "XLMRobertaForTokenClassification",
            "XLMRobertaModel",
        ]
    )
    _import_structure["models.xlnet"].extend(
        [
            "XLNET_PRETRAINED_MODEL_ARCHIVE_LIST",
            "XLNetForMultipleChoice",
            "XLNetForQuestionAnswering",
            "XLNetForQuestionAnsweringSimple",
            "XLNetForSequenceClassification",
            "XLNetForTokenClassification",
            "XLNetLMHeadModel",
            "XLNetModel",
            "XLNetPreTrainedModel",
            "load_tf_weights_in_xlnet",
        ]
    )
    _import_structure["optimization"] = [
        "Adafactor",
        "AdamW",
        "get_constant_schedule",
        "get_constant_schedule_with_warmup",
        "get_cosine_schedule_with_warmup",
        "get_cosine_with_hard_restarts_schedule_with_warmup",
        "get_linear_schedule_with_warmup",
        "get_polynomial_decay_schedule_with_warmup",
        "get_scheduler",
    ]
    _import_structure["trainer"] = ["Trainer"]
    _import_structure["trainer_pt_utils"] = ["torch_distributed_zero_first"]
    _import_structure["trainer_seq2seq"] = ["Seq2SeqTrainer"]
else:
    from .utils import dummy_pt_objects

    _import_structure["utils.dummy_pt_objects"] = [name for name in dir(dummy_pt_objects) if not name.startswith("_")]

# TensorFlow-backed objects
if is_tf_available():
    _import_structure["benchmark.benchmark_args_tf"] = ["TensorFlowBenchmarkArguments"]
    _import_structure["benchmark.benchmark_tf"] = ["TensorFlowBenchmark"]
    _import_structure["generation_tf_utils"] = ["tf_top_k_top_p_filtering"]
    _import_structure["modeling_tf_utils"] = [
        "TFPreTrainedModel",
        "TFSequenceSummary",
        "TFSharedEmbeddings",
        "shape_list",
    ]
    # TensorFlow models structure

    _import_structure["models.convbert"].extend(
        [
            "TF_CONVBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFConvBertForMaskedLM",
            "TFConvBertForMultipleChoice",
            "TFConvBertForQuestionAnswering",
            "TFConvBertForSequenceClassification",
            "TFConvBertForTokenClassification",
            "TFConvBertLayer",
            "TFConvBertModel",
            "TFConvBertPreTrainedModel",
        ]
    )
    _import_structure["models.albert"].extend(
        [
            "TF_ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFAlbertForMaskedLM",
            "TFAlbertForMultipleChoice",
            "TFAlbertForPreTraining",
            "TFAlbertForQuestionAnswering",
            "TFAlbertForSequenceClassification",
            "TFAlbertForTokenClassification",
            "TFAlbertMainLayer",
            "TFAlbertModel",
            "TFAlbertPreTrainedModel",
        ]
    )
    _import_structure["models.auto"].extend(
        [
            "TF_MODEL_FOR_CAUSAL_LM_MAPPING",
            "TF_MODEL_FOR_MASKED_LM_MAPPING",
            "TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING",
            "TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING",
            "TF_MODEL_FOR_PRETRAINING_MAPPING",
            "TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING",
            "TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING",
            "TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING",
            "TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING",
            "TF_MODEL_MAPPING",
            "TF_MODEL_WITH_LM_HEAD_MAPPING",
            "TFAutoModel",
            "TFAutoModelForCausalLM",
            "TFAutoModelForMaskedLM",
            "TFAutoModelForMultipleChoice",
            "TFAutoModelForPreTraining",
            "TFAutoModelForQuestionAnswering",
            "TFAutoModelForSeq2SeqLM",
            "TFAutoModelForSequenceClassification",
            "TFAutoModelForTokenClassification",
            "TFAutoModelWithLMHead",
        ]
    )
    _import_structure["models.bart"].extend(["TFBartForConditionalGeneration", "TFBartModel", "TFBartPretrainedModel"])
    _import_structure["models.bert"].extend(
        [
            "TF_BERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFBertEmbeddings",
            "TFBertForMaskedLM",
            "TFBertForMultipleChoice",
            "TFBertForNextSentencePrediction",
            "TFBertForPreTraining",
            "TFBertForQuestionAnswering",
            "TFBertForSequenceClassification",
            "TFBertForTokenClassification",
            "TFBertLMHeadModel",
            "TFBertMainLayer",
            "TFBertModel",
            "TFBertPreTrainedModel",
        ]
    )
    _import_structure["models.blenderbot"].extend(["TFBlenderbotForConditionalGeneration", "TFBlenderbotModel"])
    _import_structure["models.blenderbot_small"].extend(
        ["TFBlenderbotSmallForConditionalGeneration", "TFBlenderbotSmallModel"]
    )
    _import_structure["models.camembert"].extend(
        [
            "TF_CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFCamembertForMaskedLM",
            "TFCamembertForMultipleChoice",
            "TFCamembertForQuestionAnswering",
            "TFCamembertForSequenceClassification",
            "TFCamembertForTokenClassification",
            "TFCamembertModel",
        ]
    )
    _import_structure["models.ctrl"].extend(
        [
            "TF_CTRL_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFCTRLForSequenceClassification",
            "TFCTRLLMHeadModel",
            "TFCTRLModel",
            "TFCTRLPreTrainedModel",
        ]
    )
    _import_structure["models.distilbert"].extend(
        [
            "TF_DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFDistilBertForMaskedLM",
            "TFDistilBertForMultipleChoice",
            "TFDistilBertForQuestionAnswering",
            "TFDistilBertForSequenceClassification",
            "TFDistilBertForTokenClassification",
            "TFDistilBertMainLayer",
            "TFDistilBertModel",
            "TFDistilBertPreTrainedModel",
        ]
    )
    _import_structure["models.dpr"].extend(
        [
            "TF_DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TF_DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TF_DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFDPRContextEncoder",
            "TFDPRPretrainedContextEncoder",
            "TFDPRPretrainedQuestionEncoder",
            "TFDPRPretrainedReader",
            "TFDPRQuestionEncoder",
            "TFDPRReader",
        ]
    )
    _import_structure["models.electra"].extend(
        [
            "TF_ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFElectraForMaskedLM",
            "TFElectraForMultipleChoice",
            "TFElectraForPreTraining",
            "TFElectraForQuestionAnswering",
            "TFElectraForSequenceClassification",
            "TFElectraForTokenClassification",
            "TFElectraModel",
            "TFElectraPreTrainedModel",
        ]
    )
    _import_structure["models.flaubert"].extend(
        [
            "TF_FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFFlaubertForMultipleChoice",
            "TFFlaubertForQuestionAnsweringSimple",
            "TFFlaubertForSequenceClassification",
            "TFFlaubertForTokenClassification",
            "TFFlaubertModel",
            "TFFlaubertWithLMHeadModel",
        ]
    )
    _import_structure["models.funnel"].extend(
        [
            "TF_FUNNEL_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFFunnelBaseModel",
            "TFFunnelForMaskedLM",
            "TFFunnelForMultipleChoice",
            "TFFunnelForPreTraining",
            "TFFunnelForQuestionAnswering",
            "TFFunnelForSequenceClassification",
            "TFFunnelForTokenClassification",
            "TFFunnelModel",
        ]
    )
    _import_structure["models.gpt2"].extend(
        [
            "TF_GPT2_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFGPT2DoubleHeadsModel",
            "TFGPT2ForSequenceClassification",
            "TFGPT2LMHeadModel",
            "TFGPT2MainLayer",
            "TFGPT2Model",
            "TFGPT2PreTrainedModel",
        ]
    )
    _import_structure["models.led"].extend(["TFLEDForConditionalGeneration", "TFLEDModel", "TFLEDPreTrainedModel"])
    _import_structure["models.longformer"].extend(
        [
            "TF_LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFLongformerForMaskedLM",
            "TFLongformerForMultipleChoice",
            "TFLongformerForQuestionAnswering",
            "TFLongformerForSequenceClassification",
            "TFLongformerForTokenClassification",
            "TFLongformerModel",
            "TFLongformerSelfAttention",
        ]
    )
    _import_structure["models.lxmert"].extend(
        [
            "TF_LXMERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFLxmertForPreTraining",
            "TFLxmertMainLayer",
            "TFLxmertModel",
            "TFLxmertPreTrainedModel",
            "TFLxmertVisualFeatureEncoder",
        ]
    )
    _import_structure["models.marian"].extend(["TFMarianMTModel", "TFMarianModel"])
    _import_structure["models.mbart"].extend(["TFMBartForConditionalGeneration", "TFMBartModel"])
    _import_structure["models.mobilebert"].extend(
        [
            "TF_MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFMobileBertForMaskedLM",
            "TFMobileBertForMultipleChoice",
            "TFMobileBertForNextSentencePrediction",
            "TFMobileBertForPreTraining",
            "TFMobileBertForQuestionAnswering",
            "TFMobileBertForSequenceClassification",
            "TFMobileBertForTokenClassification",
            "TFMobileBertMainLayer",
            "TFMobileBertModel",
            "TFMobileBertPreTrainedModel",
        ]
    )
    _import_structure["models.mpnet"].extend(
        [
            "TF_MPNET_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFMPNetForMaskedLM",
            "TFMPNetForMultipleChoice",
            "TFMPNetForQuestionAnswering",
            "TFMPNetForSequenceClassification",
            "TFMPNetForTokenClassification",
            "TFMPNetMainLayer",
            "TFMPNetModel",
            "TFMPNetPreTrainedModel",
        ]
    )
    _import_structure["models.mt5"].extend(["TFMT5EncoderModel", "TFMT5ForConditionalGeneration", "TFMT5Model"])
    _import_structure["models.openai"].extend(
        [
            "TF_OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFOpenAIGPTDoubleHeadsModel",
            "TFOpenAIGPTForSequenceClassification",
            "TFOpenAIGPTLMHeadModel",
            "TFOpenAIGPTMainLayer",
            "TFOpenAIGPTModel",
            "TFOpenAIGPTPreTrainedModel",
        ]
    )
    _import_structure["models.pegasus"].extend(["TFPegasusForConditionalGeneration", "TFPegasusModel"])
    _import_structure["models.roberta"].extend(
        [
            "TF_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFRobertaForMaskedLM",
            "TFRobertaForMultipleChoice",
            "TFRobertaForQuestionAnswering",
            "TFRobertaForSequenceClassification",
            "TFRobertaForTokenClassification",
            "TFRobertaMainLayer",
            "TFRobertaModel",
            "TFRobertaPreTrainedModel",
        ]
    )
    _import_structure["models.t5"].extend(
        [
            "TF_T5_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFT5EncoderModel",
            "TFT5ForConditionalGeneration",
            "TFT5Model",
            "TFT5PreTrainedModel",
        ]
    )
    _import_structure["models.transfo_xl"].extend(
        [
            "TF_TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFAdaptiveEmbedding",
            "TFTransfoXLForSequenceClassification",
            "TFTransfoXLLMHeadModel",
            "TFTransfoXLMainLayer",
            "TFTransfoXLModel",
            "TFTransfoXLPreTrainedModel",
        ]
    )
    _import_structure["models.xlm"].extend(
        [
            "TF_XLM_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFXLMForMultipleChoice",
            "TFXLMForQuestionAnsweringSimple",
            "TFXLMForSequenceClassification",
            "TFXLMForTokenClassification",
            "TFXLMMainLayer",
            "TFXLMModel",
            "TFXLMPreTrainedModel",
            "TFXLMWithLMHeadModel",
        ]
    )
    _import_structure["models.xlm_roberta"].extend(
        [
            "TF_XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFXLMRobertaForMaskedLM",
            "TFXLMRobertaForMultipleChoice",
            "TFXLMRobertaForQuestionAnswering",
            "TFXLMRobertaForSequenceClassification",
            "TFXLMRobertaForTokenClassification",
            "TFXLMRobertaModel",
        ]
    )
    _import_structure["models.xlnet"].extend(
        [
            "TF_XLNET_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFXLNetForMultipleChoice",
            "TFXLNetForQuestionAnsweringSimple",
            "TFXLNetForSequenceClassification",
            "TFXLNetForTokenClassification",
            "TFXLNetLMHeadModel",
            "TFXLNetMainLayer",
            "TFXLNetModel",
            "TFXLNetPreTrainedModel",
        ]
    )
    _import_structure["optimization_tf"] = ["AdamWeightDecay", "GradientAccumulator", "WarmUp", "create_optimizer"]
    _import_structure["trainer_tf"] = ["TFTrainer"]

else:
    from .utils import dummy_tf_objects

    _import_structure["utils.dummy_tf_objects"] = [name for name in dir(dummy_tf_objects) if not name.startswith("_")]

# FLAX-backed objects
if is_flax_available():
    _import_structure["modeling_flax_utils"] = ["FlaxPreTrainedModel"]
    _import_structure["models.auto"].extend(["FLAX_MODEL_MAPPING", "FlaxAutoModel"])
    _import_structure["models.bert"].extend(["FlaxBertForMaskedLM", "FlaxBertModel"])
    _import_structure["models.roberta"].append("FlaxRobertaModel")
else:
    from .utils import dummy_flax_objects

    _import_structure["utils.dummy_flax_objects"] = [
        name for name in dir(dummy_flax_objects) if not name.startswith("_")
    ]


# Direct imports for type-checking
if TYPE_CHECKING:
    # Configuration

    # Data

    # Files and general utilities
    from .file_utils import (
        is_flax_available,
        is_sentencepiece_available,
        is_tf_available,
        is_tokenizers_available,
        is_torch_available,
    )

    # Integrations

    # Model Cards

    # TF 2.0 <=> PyTorch conversion utilities

    # Pipelines

    # Tokenization

    # Trainer

    if is_sentencepiece_available():
        pass
    else:
        pass

    if is_tokenizers_available():

        if is_sentencepiece_available():
            pass
    else:
        pass

    # Modeling
    if is_torch_available():

        # Benchmarks

        # Optimization

        # Trainer
        pass
    else:
        pass

    # TensorFlow
    if is_tf_available():

        pass

        # Benchmarks

        # Optimization

        # Trainer


    else:
        # Import the same objects as dummies to get them in the namespace.
        # They will raise an import error if the user tries to instantiate / use them.
        pass

    if is_flax_available():
        pass
    else:
        # Import the same objects as dummies to get them in the namespace.
        # They will raise an import error if the user tries to instantiate / use them.
        pass
else:
    import importlib
    import os
    import sys

    class _LazyModule(_BaseLazyModule):
        """
        Module class that surfaces all objects but only performs associated imports when the objects are requested.
        """

        __file__ = globals()["__file__"]
        __path__ = [os.path.dirname(__file__)]

        def _get_module(self, module_name: str):
            return importlib.import_module("." + module_name, self.__name__)

        def __getattr__(self, name: str):
            # Special handling for the version, which is a constant from this module and not imported in a submodule.
            if name == "__version__":
                return __version__
            return super().__getattr__(name)

    sys.modules[__name__] = _LazyModule(__name__, _import_structure)


if not is_tf_available() and not is_torch_available() and not is_flax_available():
    logger.warning(
        "None of PyTorch, TensorFlow >= 2.0, or Flax have been found. "
        "Models won't be available and only tokenizers, configuration "
        "and file/data utilities can be used."
    )
