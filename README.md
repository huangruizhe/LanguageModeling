# Language Modeling

This is an implementation of computing Kneser-Ney smoothed 
language model in the same way as 
[srilm](http://www.speech.sri.com/projects/srilm/manpages/ngram-discount.7.html)

This is a back-off, unmodified version of Kneser-Ney smoothing, 
which produces the same results as the following command 
(as an example):
```
ngram-count 
    -order 4
    -kn-modify-counts-at-end 
    -ukndiscount
    -gt1min 0 
    -gt2min 0
    -gt3min 0
    -gt4min 0
    -text corpus.txt 
    -lm lm.arpa
```

