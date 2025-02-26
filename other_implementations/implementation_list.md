## 01. DenseQMC: an efficient bit-slice implementation of the Quine-McCluskey algorithm
- Paper:            https://eprint.iacr.org/2023/201.pdf <b>(also explains and corrects a common implementation mistake for QMC)</b>
- Implementation:   https://github.com/hellman/Quine-McCluskey

## 02. Jiangbo Huang - Singapore (Department of Biological Sciences)
- Paper: https://arxiv.org/pdf/1410.1059
- Implementation extracted from paper into the .c file
- <i>I'm kinda doubtful since it's been cited 11 times, but by some random papers; the paper itself does not look too professional</i>

## 03. vj-ug implementation
- Implementation: https://github.com/vj-ug/Quine-McCluskey-algorithm
- <i>Looks like a personal project more than like an optimized implementation</i>

## 04. Barneji collection of implementations (regular + decimal representation) 
- Paper: https://arxiv.org/pdf/1404.3349 <b>(collects and presents multiple implementations of QMC)</b>
- <i>Explains and gives two implementations: the regular and a decimal representations of minterms. Gives a C and a C++ imlementation for each.</i>
- The first variant does NOT include don't care support, while variant 2 does support them.
- The second variant is RECURSIVE!
- The second variant uses inline(), which I'm not sure what it does

## 05. int-main Python implementation
- Implementation: https://github.com/int-main/Quine-McCluskey
- <i>Seems like another personal project</i>
