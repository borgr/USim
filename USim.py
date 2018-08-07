import os
import sys
import scipy
import argparse
from multiprocessing import Pool
import multiprocessing
from subprocess import call, Popen
import pickle
from functools import reduce
import operator
import platform
import re
from ucca.ioutil import file2passage
import codecs

import scripts.distances.align as align
from ucca.ioutil import passage2file
from ucca.convert import from_text

POOL_SIZE = multiprocessing.cpu_count()
full_rerank = True

from tupa.parse import Parser
from tupa.config import Config
Config("")

PARSER = None
PARSER_PATH = None
SENTENCE_ID_FILENAME = "sentenceIds.pkl"
PARSED_FILE = "parsed"


def main(args):
    if args.source_sentences is not None:
        ucca_parse_sentences(
            args.source_sentences + args.reference_sentences, args.parse_dir, args.parser_path)
        source_sentences, reference_sentences = args.source_sentences, args.reference_sentences
        res = [str(USim(s, r, args.parse_dir)) + "\n" for s,
               r in zip(source_sentences, reference_sentences)]
    else:
        ucca_parse_files(args.source_files + args.reference_files,
                         args.parse_dir, args.parser_path)

        source_files = []
        source_sentences = []
        for source_file in args.source_files:
            with open(source_file) as fl:
                for line in fl:
                    # source_sentences.append(line.strip())
                    source_files.append(source_file)
        reference_sentences = []
        reference_files = []
        for reference_file in args.reference_files:
            with open(reference_file) as fl:
                for line in fl:
                    # reference_sentences.append(line.strip())
                    reference_files.append(reference_file)
        res = [str(USim(s, r, args.parse_dir, i, i + len(source_sentences))) + "\n" for i, (s,
               r) in enumerate(zip(source_files, reference_files))]

    with open(args.output_file, "w") as fl:
        fl.writelines(res)
# a lot of code duplication because pooling doesn't react well to passing
# different lambdas as an argument


def normalize_sentence(s):
    s = re.sub(r"\W+", r" ", s)
    s = re.sub(r"(\s[a-zA-Z])\s([a-zA-Z]\s)", r"\1\2", s)
    s = s.lower()
    s = s.strip()
    return s


def rerank_by_uccasim(gamma=0.27):
    data_dir = ASSESS_DIR + "data" + os.sep
    # only used to extract source sentences
    first_nucle = data_dir + "references/" + "NUCLEA.m2"
    k_best_dir = data_dir + "K-best/"
    system_file = k_best_dir + "conll14st.output.1.best100"
    calculations_dir = "calculations_data/uccasim_rerank/"
    ucca_parse_dir = calculations_dir + "/ucca_parse/"
    full = "full" if full_rerank else ""
    output_file = full + str(gamma) + "_" + "uccasim_rank_results"
    out_text_file = calculations_dir + output_file
    out_res_file = calculations_dir + "score_" + output_file

    if not os.path.isfile(out_text_file):
        gold_file = first_nucle  # only used to extract source sentences
        print("acquiring source")
        source_sentences, _ = m2scorer.load_annotation(gold_file)

        source_sentences = source_sentences
        # load system hypotheses
        fin = m2scorer.smart_open(system_file, 'r')
        system_sentences = [line.strip() for line in fin.readlines()]
        fin.close()

        packed_system_sentences = get_roro_packed(system_sentences)

        print("parsing")
        ucca_parse(reduce(operator.add, packed_system_sentences) +
                   source_sentences, ucca_parse_dir)

        print("reranking")
        # find top ranking
        pool = Pool(POOL_SIZE)
        assert(len(packed_system_sentences) == len(source_sentences))
        if full_rerank:
            results = pool.starmap(referece_less_full_rerank, zip(source_sentences, packed_system_sentences, [
                                   ucca_parse_dir] * len(packed_system_sentences), [gamma] * len(packed_system_sentences)))
        else:
            results = pool.starmap(referece_less_oracle, zip(source_sentences, packed_system_sentences, [
                                   ucca_parse_dir] * len(packed_system_sentences), [gamma] * len(packed_system_sentences)))
        pool.close()
        pool.join()
        results = list(results)
        if full_rerank:
            results = [x for y in results for x in y]
        sentences = "\n".join(list(zip(*results))[0])
        results = list(zip(*results))[1]
        results = "\n".join([str(x) for x in results])

        print("writing to " + out_text_file)
        with codecs.open(out_text_file, "w+", "utf-8") as fl:
            fl.write(sentences)
        with open(out_res_file, "w+") as fl:
            fl.write(results)


def parse_location(output_dir, filename, sentence_num=None):
    filename = os.path.splitext(os.path.basename(filename))[0]
    cur_dir = os.path.join(output_dir, filename)
    if sentence_num is None:
        return cur_dir
    return os.path.join(cur_dir, str(sentence_num) + ".xml")


def get_parser(model_path):
    global PARSER
    global PARSER_PATH
    if PARSER_PATH is not model_path or PARSER is None:
        PARSER_PATH = model_path
        PARSER = Parser(model_path)
    return PARSER


def ucca_parse_sentences(sentences, output_dir, model_path, clean=False, normalize_sentence=normalize_sentence):
    sentences = list(set([normalize_sentence(sentence)
                          for sentence in sentences]))
    output_dir = os.path.realpath(output_dir)
    to_parse = get_parsed_subdirs(sentences, output_dir)
    to_parse = [sent for sent, loc in zip(sentences, to_parse) if loc is None]
    if to_parse:
        i = 0
        out_path = os.path.join(output_dir, "parse_batch" + str(i))
        while os.path.isfile(os.path.join(out_path, SENTENCE_ID_FILENAME)):
            i += 1
            out_path = os.path.join(output_dir, "parse_batch" + str(i))
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        print("Output folder:", out_path)

        for i, sentence in enumerate(to_parse):
            # adds sentences to sentence ids memory
            tmp = get_sentence_id(sentence, out_path, True, normalize_sentence)
            assert tmp == i, (tmp, i)
        print(to_parse)
        print("Parsing", len(to_parse), "sentences.", len(
            sentences) - len(to_parse), "sentences already parsed.")
        _ucca_parse_text(to_parse, out_path, "", clean,
                         normalize_sentence, model_path)
    else:
        print("All", len(sentences), "sentences already parsed")


def ucca_parse_files(filenames, output_dir, model_path, clean=False, normalize_sentence=lambda x: x):
    output_dir = os.path.realpath(output_dir)
    if filenames:
        for filename in filenames:
            cur_output_dir = parse_location(output_dir, filename)
            if os.path.isdir(cur_output_dir):
                print("File already parsed in", cur_output_dir)
            else:
                os.makedirs(cur_output_dir)
                with open(filename, "r") as fl:
                    text = fl.readlines()
                _ucca_parse_text(text, output_dir, filename,
                                 clean, normalize_sentence, model_path)


def _ucca_parse_text(text, output_dir, filename, clean, normalize_sentence, model_path):
    text = [normalize_sentence(x) for x in text]
    text = from_text(text, split=True, one_per_line=True)
    text = list(text)
    parser = get_parser(model_path)
    out_location = os.path.dirname(parse_location(output_dir, filename, 0))
    if not os.path.isdir(out_location):
        os.makedirs(out_location)
    for i, (passage, *_) in enumerate(parser.parse(text)):
        passage2file(passage, parse_location(
            output_dir, filename, i))
    # create an empty file anounces parsing finished succsessfuly
    parsed_file = os.path.join(out_location, PARSED_FILE)
    with open(parsed_file, "w") as _:
        pass
    if clean:
        filenames = os.listdir(output_dir)
        for filename in filenames:
            if filename.endswith(".txt"):
                os.remove(os.path.join(output_dir, item))


_id_dics = {}


def get_parsed_subdirs(sentences, parse_dir):
    res = [None] * len(sentences)
    parse_dir = os.path.realpath(parse_dir)
    for parse_subdir, dirs, files in os.walk(parse_dir):
        if PARSED_FILE in files:
            for i, sentence in enumerate(sentences):
                if res[i] is None:  # avoid multiple lookups in case the sentence was already found once
                    try:
                        get_sentence_id(sentence, parse_subdir, False)
                        res[i] = parse_subdir
                    except KeyError:
                        pass
    return res


def get_parsed_subdir(sentence, parse_dir):
    parse_dir = os.path.realpath(parse_dir)
    for parse_subdir, dirs, files in os.walk(parse_dir):
        if PARSED_FILE in files and any((fl.endswith(SENTENCE_ID_FILENAME) for fl in files)):
            try:
                get_sentence_id(sentence, parse_subdir, False)
                return parse_subdir
            except KeyError:
                pass


def get_sentence_id(sentence, parse_dir, graceful=True, normalize_sentence=normalize_sentence):
    """ returns the sentence id in the parse_dir, 
        if graceful is true adds a new sentence id 
        if the sentence does not exist in the ids list,
        otherwise throws exception"""
    parse_dir = os.path.realpath(parse_dir)
    filename = SENTENCE_ID_FILENAME
    max_id = "max"
    sentence = normalize_sentence(sentence)
    if parse_dir in _id_dics:
        id_dic = _id_dics[parse_dir]
    elif not os.path.isfile(parse_dir + os.sep + filename):
        print("creating a new id list for file", parse_dir + os.sep + filename)
        id_dic = {max_id: -1}
        _id_dics[parse_dir] = id_dic
    else:
        with open(parse_dir + os.sep + filename, "rb") as fl:
            id_dic = pickle.load(fl)
            _id_dics[parse_dir] = id_dic
    if graceful and not sentence in id_dic:
        id_dic[max_id] += 1
        id_dic[sentence] = id_dic[max_id]
        with open(parse_dir + os.sep + filename, "wb+") as fl:
            pickle.dump(id_dic, fl)
    return id_dic[sentence]


def parsed_sentence2xml(sentence, parse_dir, sent_id=None, normalize_sentence=normalize_sentence):
    if sent_id is None:
        location = get_parsed_subdir(sentence, parse_dir)
        filename = parse_location(location, "", get_sentence_id(
            sentence, location, False, normalize_sentence))
        # print("reading parse from ", filename)
        # with open(filename) as fl:
        #     print("sentence:", sentence)
        #     print("xml first lines:", fl.readlines()[:30])
        return file2passage(filename)
    else:
        return file2passage(parse_location(parse_dir, sentence, sent_id))


def USim(source, sentence, parse_dir, source_id=None, sentence_id=None, normalize_sentence=normalize_sentence):
    """ accepts filename instead of sentence\source and a sentence id\source_sentence id for locating the file"""
    if align.regularize_word(source) == "":
        if align.regularize_word(sentence) == "":
            return 1
        else:
            return 0
    elif align.regularize_word(sentence) == "":
        return 0
    source_xml = parsed_sentence2xml(
        source, parse_dir, source_id, normalize_sentence)
    sentence_xml = parsed_sentence2xml(
        sentence, parse_dir, sentence_id, normalize_sentence)
    return align.fully_aligned_distance(source_xml, sentence_xml)


def announce_finish():
    if sys.platform == "linux":
        if set(("debian", "Ubuntu")) & set(platform.linux_distribution()):
            call(['speech-dispatcher'])  # start speech dispatcher
            call(['spd-say', '"your process has finished"'])
        else:
            # perhaps works only in ubuntu?
            a = Popen(
                ('play --no-show-progress --null --channels 1 synth %s sine %f' % (300, 2)).split())
    elif sys.platform == "darwin":
        call('say "your process has finished"'.split())
    else:
        import winsound
        winsound.Beep(300, 2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract USim scores')
    parser.add_argument(
        'parse_dir', help="Name of the directory to save and look for parsed sentences")
    parser.add_argument('output_file')
    parser.add_argument('-sf', '--source_files', nargs='+')
    parser.add_argument('-rf', '--reference_files', nargs='+')
    parser.add_argument('-ss', '--source_sentences', nargs='+')
    parser.add_argument('-rs', '--reference_sentences', nargs='+')
    parser.add_argument('-p', "--parser_path", help="The path to the tupa model to be used, if no parameter is passed the hard-coded USim.PARSER_PATH path would be used")

    args, unknown = parser.parse_known_args()
    PARSER_PATH = args.parser_path
    if not((args.source_files is not None and args.reference_files is not None) or (args.source_sentences is not None and args.reference_sentences is not None)):
        print("please provide sources and references as files or as sentences.")
    main(args)
