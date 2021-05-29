import string
import numpy as np

# default list of characters to encode and decode
default_vocabulary = list(string.ascii_lowercase) + list(string.digits) +\
    [' ', '-', '.', ':', '?', '!', '<', '>', '#', '@', '(', ')', '$', '%', '&'] +\
    ['١', '٢', '٣', '٤', '٥', '٦', '٧', '٨', '٩', '٠', 'ى', 'ا', 'أ', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د',
     'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي']


class Vocabulary:
    pad = '<PAD>'
    sos = '<SOS>'
    eos = '<EOS>'
    unk = '<UNK>'

    def __init__(self, vocabulary: list = default_vocabulary):
        '''
            Class wrapper to encode and decode the text before passing to the network to trained on
            It Perform one hot encodeing for supported character for the model to trained on
                vocabulary: the list of vocabulary that the model will trained on If not passed the model will only trained on the arabic and english language
        '''
        self._characters = [self.pad, self.sos,
                            self.eos, self.unk] + sorted(vocabulary)
        self._character_index = dict(
            [(char, i) for i, char in enumerate(self._characters)])
        self._character_reverse_index = dict(
            (i, char) for char, i in self._character_index.items())

    def is_special_character(self, char_idx) -> bool:
        '''
            Check if charcater is a special char (pad, sos, eos, unk)
        '''
        return self._characters[char_idx] in [self.pad, self.sos, self.eos]

    def one_hot_encode(self, txt: str, length: int, sos: bool = False, eos: bool = True) -> np.ndarray:
        '''
            Applay one hot encoding (Convert every chars in the text to number)
        '''
        txt = list(txt)
        txt = txt[:length - int(sos) - int(eos)]
        txt = [c if c in self._characters else self.unk for c in txt]

        if sos:
            txt = [self.sos] + txt
        if eos:
            txt = txt + [self.eos]

        txt += [self.pad] * (length - len(txt))
        encoding = np.zeros((length, len(self)), dtype='float32')

        for char_pos, char in enumerate(txt):
            if char in self._character_index:
                encoding[char_pos, self._character_index[char]] = 1.
            else:
                encoding[char_pos, self._character_index[self.unk]] = 1.
        return encoding

    def one_hot_decode(self, one_hot: np.ndarray, max_length: int) -> str:
        '''
            Applay one hot decoder (Convert the prediction back to text)
        '''
        text = ''
        for sample_index in np.argmax(one_hot, axis=-1):
            sample = self._character_reverse_index[sample_index]
            if sample == self.eos or sample == self.pad or len(text) > max_length:
                break
            text += sample
        return text

    def __len__(self):
        '''
            Return the length of the supported chars
        '''
        return len(self._characters)
