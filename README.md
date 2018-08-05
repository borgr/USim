# USim
monolingual sentence similarity measure

Please cite our NAACL2018 paper if you use our measure.

`@inproceedings{choshen2018reference,
  title={Reference-less Measure of Faithfulness for Grammatical Error Correction},
  author={Choshen, Leshem and Abend, Omri},
  booktitle={Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers)},
  volume={2},
  pages={124--129},
  year={2018}
}
`

As USim uses currently the TUPA parser it should be installed

`pip install tupa`

Usage example:
python USim.py parse out.out -ss "I love rusty spoons", "nothing matters" -rs "he shares pretty cars", "nothing indeed"
