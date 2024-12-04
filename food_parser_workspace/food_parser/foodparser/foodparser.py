import re  # regex
import nltk  # nlp stuff
import string  # string manipulation
import json  # json file storage
import pkg_resources  # file location
import numpy as np  # arrays and math
import wordsegment
import pandas as pd  # dataframe (table) manipulations
from tqdm import tqdm  # progress bars for operations

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords  # common stopwords (e.g. 'the')

wordsegment.load()
nltk.download("words")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")


class FoodParser:
    """
    Food parser handles taking unprocessed food log entries and adding relevant
    information from a pre-made dictionary (taken from a variety of Panda Lab
    studies) of food items. This includes matching unprocessed terms to
    their likely matches, adding food type and other identifying information.
    """

    def __init__(self):
        """Initializes food parser object."""
        # read in manually annotated file
        parser_keys_df = pd.read_csv(
            pkg_resources.resource_stream(__name__, "data/2024_12_parser_keys.csv")
        )
        all_gram_set, food_type_dict, food2tags = self.process_parser_keys_df(
            parser_keys_df
        )
        correction_dic = json.load(
            pkg_resources.resource_stream(__name__, "data/2024_12_correction_dict.json")
        )
        self.all_gram_set = all_gram_set
        self.food_type_dict = food_type_dict
        self._food_phrases = np.unique(
            [
                word
                for key in self.food_type_dict
                for word in key.split()
                if self.food_type_dict[key] in ["f", "b", "m"] and len(word) > 2
            ]
        )
        self.correction_dic = correction_dic
        self.food2tags = food2tags

        # Load common stop words and nlp type objects
        self.wnl = WordNetLemmatizer()

        self.stop_words = stopwords.words("english")
        self.stop_words.remove("out")  # since pre work out is a valid beverage name
        self.stop_words.remove("no")
        self.stop_words.remove("not")
        self.stop_words.remove("and")
        self.stop_words.remove("with")
        self.stop_words.remove("m")
        self.stop_words.remove("of")
        # preventing common vitamins from being removed
        self.stop_words.remove("d")

    @staticmethod
    def process_parser_keys_df(parser_keys_df):
        """Takes a dataframe of parser keys and processes gram sets and
        food type information.

        Parameters
        ----------
        parser_keys_df: pd.DataFrame
            Dataframe of parser keys.

        Returns
        -------
        all_gram_set: list
            List of sets where each set corresponds to all (n-grams for n in range 1-6)

        food_type_dict: dict
            Dictionary of gram key and food type pairings where gram key
            is the food (e.g. 'milk') and the value is the type (e.g. 'b')

        food2tags: dict
            Dictionary of gram key and corresponding tags. (e.g. key = 'vodka'
            with tags 'alcohol', 'liquor', etc..)
        """
        # f = food, b = beverage, m = medicine, w = water
        # below are the possible valid item types
        parser_keys_df = parser_keys_df.query(
            'food_type in ["f", "b", "m", "w", "modifier", "general", "stopword", "selfcare"]'
        ).reset_index(drop=True)

        # creates an all_gram_set to store n-grams (e.g. 1-grams, 2-grams, etc.)
        all_gram_set = []
        for i in range(1, 5 + 1):
            all_gram_set.append(
                set(parser_keys_df.query("gram_type == " + str(i))["gram_key"].values)
            )

        # pair gram key and corresponding food type for items as a dictionary
        food_type_dict = dict(
            zip(parser_keys_df["gram_key"].values, parser_keys_df["food_type"].values)
        )

        # find all relevant tags
        def find_all_tags(df):
            """Finds tags for each 'food' item.

            Parameters
            -----------
            df: DataFrame
                Dataframe with food tag columns (e.g. tag1)
            """
            tags = []
            for i in range(1, 8):
                # expects column names of the format 'tag#'
                if not isinstance(df["tag" + str(i)], str) and np.isnan(
                    df["tag" + str(i)]
                ):
                    # append empty tag if tag doesn't exist
                    # forced food2tags to have length 8 + can restore tags
                    # in order with a dictionary
                    tags.append("")
                    continue
                # append existing tags
                tags.append(df["tag" + str(i)])
            return tags

        # pair gram_keys (food names) and their corresponding list of informative tags
        food2tags = dict(
            zip(
                parser_keys_df["gram_key"].values,
                parser_keys_df.apply(find_all_tags, axis=1),
            )
        )
        return all_gram_set, food_type_dict, food2tags

    ########## Pre-Processing ##########

    # Function for removing numbers
    @staticmethod
    def handle_numbers(text):
        """Removes numeric characters from text.

        Parameters
        ----------
        text: str
            Text to be cleaned

        Returns
        -------
        text: str
            Text with numbers removed.
        """
        # regex pattern removes all decimal numbers
        text = re.sub("[0-9]+\.[0-9]+", "", text)
        # retaining common items (h2o, co2, v8, ag1)
        if any(sub in text for sub in ["v8", "h2", "ag1", "co2", "0z"]):
            # regex pattern strips any numbers not immediately preceded or followed by a letter
            text = re.sub(r"(?<![a-zA-Z])\d+(?![a-zA-Z])", "", text)
        else:
            # regex pattern strips all whole numbers
            text = re.sub("[0-9+]", "", text)
        # regex pattern truncates any duplicate whitespace (e.g. "  " becomes  " ")
        text = re.sub("\s\s+", " ", text)
        return text

    # Function for removing punctuation
    @staticmethod
    def drop_punc(text):
        """
        Removes punctuation from text.

         Parameters
        ----------
        text: str
            Text to be cleaned

        Returns
        -------
        text: str
            Text with punctuation removed.
        """
        return re.sub("[%s]" % re.escape(string.punctuation), " ", text)

    # Remove normal stopwords
    def remove_stop(self, text):
        """
        Removes english stop words defined by nltk (e.g. 'the') from text.

         Parameters
        ----------
        text: str
            Text to be cleaned

        Returns
        -------
        text: str
            Text with stop words removed.
        """
        return " ".join([word for word in text.split() if word not in self.stop_words])

    def pre_processing(self, text):
        """
        Handles text cleaning overall.

        Parameters
        ----------
        text: str
            Text to be pre-processed/cleaned.

        Returns
        --------
        text: str
            Cleaned text.
        """
        return self.remove_stop(
            self.handle_numbers(self.drop_punc(text.lower()))
        ).strip()

    ########## Handle Format ##########
    @staticmethod
    def handle_front_mixing(sent, front_mixed_tokens):
        """
        Handles cleaned front mixed tokens, which seem to be some sort of digit/numerical.
        Potentially dealing with cleaning out item quantities? (e.g. 12oz?)

        Parameters
        ----------
        sent: str
            Phrase/sentence describing a food item.

        front_mixed_tokens: list
            List of front mixed tokens (I think these are supposed to be amount tokens),
            e.g. 12oz

        Returns
        --------
        sent: str
            Original sentence with any front mixed tokens cleaned (e.g. 12oz --> 12 oz).
        """
        cleaned_tokens = []
        for t in sent.split():
            if t in front_mixed_tokens:
                number = re.findall("\d+[xX]*", t)[0]
                for t_0 in t.replace(number, number + " ").split():
                    cleaned_tokens.append(t_0)
            else:
                cleaned_tokens.append(t)
        return " ".join(cleaned_tokens)

    @staticmethod
    def handle_x2(sent, times_x_tokens):
        """
        Handles variations in spacing or capitalization for the same token.
        e.g. ('x2' vs 'X2' vs 'X 2' vs 'X 2')

        Parameters
        ----------
        sent: str
            Phrase/sentence describing a food item.

        times_x_tokens: list of 'times x' tokens to be dealt with.

        Returns
        --------
        sent: str
            Original sentence with any 'times x' variations cleaned.
        """
        for token in times_x_tokens:
            sent = sent.replace(token, " " + token.replace(" ", "").lower() + " ")

        return " ".join([s for s in sent.split() if s != ""]).strip()

    def clean_format(self, sent):
        """
        Cleans text of any amount indications. Specifically deals with
        front mixing (amount of an item noted at the front of the string)
        and 'times x' where multiple of an item is recorded in the style of
        (item x 2) or some variation of that.

        Parameters
        ----------
        sent: str
            Sentence to be cleaned.

        Returns
        -------
        sent: str
            Sentence after all amount/format style cleaning applied.
        """
        # Problem 1: 'front mixing'
        front_mixed_tokens = re.findall("\d+[^\sxX]+", sent)
        if len(front_mixed_tokens) != 0:
            sent = self.handle_front_mixing(sent, front_mixed_tokens)

        # Problem 2: 'x2', 'X2', 'X 2', 'x 2'
        times_x_tokens = re.findall("[xX]\s*?\d", sent)
        if len(times_x_tokens) != 0:
            sent = self.handle_x2(sent, times_x_tokens)
        return sent

    ########## Handle Typos ##########

    def fix_spelling(self, entry):
        """Corrects spelling mistakes.

        Parameters
        ----------
        entry: str
            String entry to be corrected.

        Returns
        -------
        entry: str
            Original entry with spelling corrections applied.
        """
        result = []

        # if entry is recognized keep it
        if entry in self.food_type_dict:
            return entry
        # try looking for entry in correction dictionary
        if entry in self.correction_dic:
            return " ".join(
                [self.wnl.lemmatize(i) for i in self.correction_dic[entry].split()]
            )
        # try lemmatized version of whole entry
        if self.wnl.lemmatize(entry) in self.food_type_dict:
            return self.wnl.lemmatize(entry)
        # try correcting individual tokens within entry phrase
        for token in entry.split():
            # try lemmatized version of the token
            lem_token = self.wnl.lemmatize(token)
            # try looking for token in correction dictionary
            if token in self.correction_dic:
                token = " ".join(
                    [self.wnl.lemmatize(i) for i in self.correction_dic[token].split()]
                )
            elif (lem_token in self._food_phrases) or (
                lem_token in self.food_type_dict
            ):
                token = lem_token
            # check if token is incorrectly joined
            # (e.g. blueberrymuffin instead of blueberry muffin)
            elif token not in self._food_phrases:
                temp = []
                token_alt = wordsegment.segment(token)
                for word in token_alt:
                    if self.wnl.lemmatize(word) in self._food_phrases:
                        temp += [word]
                    # check if split word needs spell correction
                    elif (
                        word in self.correction_dic
                        and self.correction_dic[word] in self._food_phrases
                    ):
                        temp += [self.correction_dic[word]]
                # if there are any newly corrected/unjoined tokens add them back to the result
                if len(temp) > 1:
                    token = " ".join([self.wnl.lemmatize(i) for i in temp])
            result.append(token.strip())
        return " ".join(result).strip()

    def handle_all_cleaning(self, entry):
        """
        Performs all types of text cleaning.

        Parameters
        ----------
        entry: str
            String entry to be corrected.

        Returns
        -------
        entry: str
            String entry with all forms of pre-processing and cleaning
            applied.
        """
        entry = self.pre_processing(entry)
        entry = self.clean_format(entry)
        entry = self.fix_spelling(entry)
        entry = re.sub("\s\s+", " ", entry)
        return entry

    ########## Handle Gram Matching ##########
    @staticmethod
    def parse_single_gram(gram_length, gram_set, gram_lst, sentence_tag):
        """
        Parses a single gram, combining words as necessary for the proper gram
        length (e.g. grape juice is a bigram) and then adding any new words
        to a food list.

        Parameters
        ----------
        gram_length: int
            Length of gram to look for (e.g. 2 --> bigram).

        gram_set: list
            Existing gram set.

        gram_list: list
            List of grams to check for entry into the gram set.

        sentence tag: list
            List of tags for the sentence?

        Returns
        -------
            food_lst: list
                All unique items for the given gram being parsed.
        """
        food_lst = []
        for i, gram in enumerate(gram_lst):
            # if not a unigram, combine the words from the gram list into a
            # gram string
            if gram_length > 1:
                curr_word = " ".join(gram)
            else:
                curr_word = gram
            if (
                curr_word in gram_set
                and sum([t != "Unknown" for t in sentence_tag[i : i + gram_length]])
                == 0
            ):
                # add any first occurance of a food to the food list
                sentence_tag[i : i + gram_length] = str(gram_length)
                food_lst.append(curr_word)
        return food_lst

    def parse_single_entry(self, entry, return_sentence_tag=False):
        """
        Handles pre-processing, cleaning, and gram processing for a single entry.

        Parameters
        ----------
        entry: str
            Entry to be processed

        Returns
        -------
        All grams of length 5 or under for the particular entry.
        """
        cleaned = self.handle_all_cleaning(entry)

        # Create tokens and n-grams
        tokens = nltk.word_tokenize(cleaned)
        bigram = list(nltk.ngrams(tokens, 2)) if len(tokens) > 1 else None
        trigram = list(nltk.ngrams(tokens, 3)) if len(tokens) > 2 else None
        quadgram = list(nltk.ngrams(tokens, 4)) if len(tokens) > 3 else None
        pentagram = list(nltk.ngrams(tokens, 5)) if len(tokens) > 4 else None
        all_gram_lst = [tokens, bigram, trigram, quadgram, pentagram]

        # Create an array of tags
        sentence_tag = np.array(["Unknown"] * len(tokens))

        all_food = []
        for gram_length in [5, 4, 3, 2, 1]:
            if len(tokens) < gram_length:
                continue
            tmp_food_lst = self.parse_single_gram(
                gram_length,
                self.all_gram_set[gram_length - 1],
                all_gram_lst[gram_length - 1],
                sentence_tag,
            )
            all_food += tmp_food_lst
        if return_sentence_tag:
            return all_food, sentence_tag
        return all_food

    def parse_food(self, series):
        """
        Parses a series of single food entries.

        Parameters
        ----------
        series: list or series of str
        Food entries to be parsed.
        """
        vec = np.vectorize(self._parse_food)
        return vec(series)

    def _parse_food(self, entry):
        """
        Parses a single food entry.

        Parameters
        ----------
        entry: str
            Food entry to be parsed.

        return_sentence_tag: bool
            If true, returns token information (number of tokens in the entry,
            number of unknown tokens in the entry, unknown tokens within
            the entry).
        """
        result = []
        unknown_tokens = []
        num_unknown = 0
        num_token = 0

        for word in entry.split(","):
            # previous versions of this if statement assume that
            # return_sentence_tag is True and will actually break
            # if the default parameters are used
            all_food, sentence_tag = self.parse_single_entry(word, True)
            result += all_food

            if sentence_tag is not None and len(sentence_tag) > 0:
                num_unknown += sum(np.array(sentence_tag) == "Unknown")
                num_token += len(sentence_tag)
                cleaned = nltk.word_tokenize(self.handle_all_cleaning(word))

                # Return uncaught tokens, grouped into sub-sections
                tmp_unknown = ""
                for i, tag in enumerate(sentence_tag):
                    if tag == "Unknown":
                        tmp_unknown += " " + cleaned[i]
                        if i == len(sentence_tag) - 1:
                            unknown_tokens.append(tmp_unknown.strip())
                    elif tmp_unknown != "":
                        unknown_tokens.append(tmp_unknown.strip())
                        tmp_unknown = ""
        return np.array(
            [result, num_token, num_unknown, unknown_tokens], dtype="object"
        )

    def find_food_type(self, food):
        """
        Finds corresponding food-item type for a given food. Types include
        beverage, water, food, medicine.

        Parameters
        ----------
        food: str
            Food to find type for

        Returns
        -------
        Corresponding food type abbrevation or 'u' if not found.
        """
        if food in self.food_type_dict:
            return self.food_type_dict[food]
        # shorthand for unknown
        return "u"

    ################# DataFrame Functions #################
    def expand_entries(self, df):
        """
        Creates an 'exploded' version of the given dataframe, expanding entries
        from within list-style 'desc_text' column entries such that each
        entry in a list has its own corresponding row.

        Parameters
        ----------
        df: pd.Dataframe
            Dataframe to be expanded.

        Returns
        -------
        df: pd.Dataframe
            Expanded dataframe.
        """
        assert "desc_text" in df.columns, '[ERROR] Required a column of "desc_text".'
        df = df.copy()
        df["desc_text"] = df["desc_text"].str.split(",")
        df = df.explode("desc_text")
        # remove entries that are only spaces or digits (entries with no alphabetical characters)
        df = df[~df["desc_text"].str.isspace()]
        df = df[~df["desc_text"].str.isdigit()]
        return df
