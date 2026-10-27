#### Tokenisation and byte-pair encoding
A language model does not read characters but a sequence of integer *token ids* from a fixed
vocabulary. The tokeniser defines this mapping, and the choice matters:

- **Word-level** vocabularies are huge and cannot represent unseen words (typos, names, new terms).
- **Character- or byte-level** vocabularies are tiny and never fail, but sequences become long, and the
  cost of a transformer grows quadratically with the sequence length.
- **Subword** tokenisers sit in between: frequent words are single tokens, rare words are composed of
  several pieces (for example `" decentralisation"` → `" de" "central" "isation"`).

**Byte-pair encoding (BPE)**, originally a compression algorithm, is the subword method used by the GPT
family, Llama and many others. In its byte-level variant it works as follows:

1. *Pre-tokenisation*: a regular expression splits the text into chunks - words with their leading
   space, groups of up to three digits, punctuation, whitespace. Merges never cross chunk boundaries,
   so `"dog."` and `"dog!"` share the token `" dog"`.
2. Each chunk is encoded as UTF-8 bytes. The initial vocabulary consists of the 256 possible bytes,
   so *every* string can be encoded; there is no unknown token.
3. *Training*: count all pairs of adjacent symbols in the corpus, merge the most frequent pair into a
   new symbol with the next free id, repeat until the vocabulary has the desired size. The list of
   merges, in order, *is* the tokeniser.
4. *Encoding* new text applies the merges in the order they were learned (always the pair with the
   lowest merge rank first); *decoding* concatenates the bytes of the tokens.

Special tokens such as `<|endoftext|>` are added on top of the learned vocabulary and are used to
separate documents when they are concatenated into one training stream. They are control signals, so
untrusted text (user input, scraped web pages) must never be able to produce them just by containing
the string `"<|endoftext|>"` - otherwise a user could end a document or inject a chat role marker
(*special token injection*). tiktoken therefore raises an error by default if the text contains a
special token string; `encode(text, allowed_special="all")` is for trusted text, and
`encode_ordinary` encodes the string as normal text. Our from-scratch tokeniser behaves the same way.

##### BPE from scratch (`main_bpe_tokeniser.py`, `utils/bpe.py`)
`BPETokeniser` implements the algorithm above in about 150 lines of plain Python, using the
pre-tokenisation pattern of OpenAI's `cl100k_base` (GPT-4) tokeniser, exactly as in tiktoken. It expects
valid Unicode text: a lone surrogate code point (which can appear in broken JSON) has no UTF-8 encoding
and raises a `UnicodeEncodeError`, as in tiktoken. For speed, training counts pairs
over the *distinct* chunks weighted by their frequency instead of over the raw text. The script trains
it on the small bundled corpus `data/corpus.txt` (an original text about personal finance, with some
Danish, German, French and Spanish sentences and a few lines of code), using 90 % of the lines for
training and 10 % as held-out text. The first merges are `" t"`, `" a"`, `"in"`, `"he"`, `"re"`,
and `" t"` + `"he"` → `" the"` follows quickly; after 256 merges, frequent domain words such as
`" interest"` and `" expected"` are single tokens.

The script then checks that `decode(encode(text)) == text`, shows how example sentences are split
(Danish text and numbers need many more tokens than English prose, because the corpus is mostly
English), saves and reloads the tokeniser as JSON, and exports the learned merges to
`tiktoken.Encoding`. Because tiktoken uses exactly the same algorithm, the Rust implementation must
produce identical ids - a strong test of the from-scratch code - and it is more than an order of
magnitude faster.

##### Comparison with production tokenisers
The Hugging Face `tokenizers` library trains a byte-level BPE in Rust with a few lines of code
(`models.BPE`, `pre_tokenizers.ByteLevel`, `trainers.BpeTrainer`). Its tokens are displayed with a
byte-to-unicode mapping, e.g. `Ġ` for a leading space. With the same vocabulary size it reaches almost
the same compression; small differences come from the pre-tokenisation rules and tie-breaking.

The quality of a tokeniser is often summarised by its *compression*: bytes (or characters) per token on
held-out text. On the held-out text, raw bytes give 1.0 bytes/token, 512-token BPE vocabularies about
2.1 and a 768-token vocabulary about 2.5. OpenAI's `cl100k_base` with its 100k vocabulary reaches roughly
4 characters per token on English text (it is included in the comparison when its vocabulary file can
be downloaded). A larger vocabulary shortens sequences but increases the size of the embedding and
output layers, and rare tokens receive few training updates; vocabularies of modern LLMs range from
32k to about 256k tokens.

##### Tokenising a real dataset (`main.py`)
`main.py` tokenises the instructions of the `gbharti/finance-alpaca` dataset from Hugging Face with
`cl100k_base`, appends the end-of-text token after each instruction (as when preparing pre-training
data) and prints statistics such as the average number of tokens per instruction. Note that the
end-of-text id is `encoding.eot_token`; encoding the *string* `"<|endoftext|>"` as ordinary text would
instead produce several normal tokens. This script needs internet access on its first run.

The tokeniser used to train the miniGPT model in part 7 (`p07_llms/c00_gpt_like_models`) is a
byte-level BPE trained with the `tokenizers` library on a much larger Wikipedia corpus.
