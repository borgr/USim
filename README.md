# USim
monolingual sentence similarity measure

Please cite our NAACL2018 paper if you use our measure.
`@InProceedings{choshen:2018,
  author    = {Choshen, Leshem  and  Abend, Omri},
  title     = {Reference-less Measure of Faithfulness for Grammatical Error Correction},
  booktitle = {Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2018},
  publisher = {Association for Computational Linguistics}
}
`

As USim uses currently the TUPA parser it should be installed
`pip install tupa`

Usage example:
python USim.py parse out.out -ss "I love rusty spoons", "nothing matters" -rs "he shares pretty cars", "nothing indeed"
