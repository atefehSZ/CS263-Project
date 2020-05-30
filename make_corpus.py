import sys
from gensim.corpora import WikiCorpus

## used enwiki-latest-pages-articles1.xml-p1p30303.bz2 as the input
## shouldn't decompress it and should pass it az bz2 file
## download from https://dumps.wikimedia.org/enwiki/latest/

def make_corpus(in_f, out_f):

    """Convert Wikipedia xml dump file to text corpus"""
    print(in_f)
    output = open(out_f, 'w')
    wiki = WikiCorpus(in_f)
    print("done")

    i = 0
    for text in wiki.get_texts():
        output.write(bytes(' '.join(text), 'utf-8').decode('utf-8') + '\n')
    output.close()
    print('Processing complete!')


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print('Usage: python make_wiki_corpus.py <wikipedia_dump_file> <processed_text_file>')
        sys.exit(1)
    in_f = sys.argv[1]
    out_f = sys.argv[2]
    make_corpus(in_f, out_f)
