# Smith-Waterman Local Sequence Alignment

Sequence alignment is an important task in bioinformatics used to identify regions of conservation between two sequences. The Smith-Waterman local sequence alignment algorithm is one of the original sequence alignment algorithms, but its O(mn) complexity poses practical issues with larger sequences. Altering the Smith-Waterman algorithm, can allow it to be ran parallel utilizing modern GPU architecture resulting in increased performance over the original sequential CPU algorithm.
