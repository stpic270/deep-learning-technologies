import json
import pickle
import datasets
from datasets.tasks import QuestionAnsweringExtractive


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@inproceedings{rurebus,
  Address = {Moscow, Russia},
  Author = {Ivanin, Vitaly and Artemova, Ekaterina and Batura, Tatiana and Ivanov, Vladimir and Sarkisyan, Veronika and Tutubalina, Elena and Smurov, Ivan},
  Title = {RuREBus-2020 Shared Task: Russian Relation Extraction for Business},
  Booktitle = {Computational  Linguistics  and  Intellectual  Technologies:  Proceedings of the International Conference “Dialog” [Komp’iuternaia Lingvistika  i  Intellektual’nye  Tehnologii:  Trudy  Mezhdunarodnoj  Konferentsii  “Dialog”]},
  Year = {2020}
}
"""

_DESCRIPTION = """\
RuREBus dataset is tokenized for NER task with sbert_large_nlu_ru  model\
"""

_URL = "https://github.com/stpic270/deep-learning-technologies/new/main"
_URLS = {
    "train": _URL + "train_instances.json",
    "test": _URL + "test_instances.json",
}


class RuREBus(datasets.BuilderConfig):
    """BuilderConfig for SQUAD."""

    def __init__(self, **kwargs):
        """BuilderConfig for RuREBus.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SquadConfig, self).__init__(**kwargs)


class Squad(datasets.GeneratorBasedBuilder):
    """RuREBus: RuREBus labeled for bert_large_nlu_ru model"""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "answers": list(
                        {   "id": datasets.Value("string")
                            "ner_tags": list(),
                            "tokens":list(),
                        }
                    ),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="https://github.com/dialogue-evaluation/RuREBus",
            citation=_CITATION,
            
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train.txt"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test.txt"]}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        key = 0
        with open(filepath, "rb") as f:
            dct = pickle.load(f)
            id_ = 0         
            for ner_tags, tokens in zip(dct['target'], dct['sentence']):
                
                yield id_, {
                             "answers": list(
                        {   "id": str(id_)
                            "ner_tags": ner_tags,
                            "tokens": tokens,
                            }
                             )
                        }
                id_ += 1
