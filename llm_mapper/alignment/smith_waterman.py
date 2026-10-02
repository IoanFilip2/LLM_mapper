"""Exact affine-gap local DNA alignment with a length-normalized similarity.

The score is a sequence-similarity statistic, not a proof of evolutionary homology.
Its complement is a dissimilarity, not a mathematical metric (no triangle guarantee).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class SWConfig:
    match: int = 2
    mismatch: int = -3
    gap_open: int = 5
    gap_extend: int = 1
    both_strands: bool = True
    algorithm: str = 'striped'

    def __post_init__(self):
        if self.match <= 0 or self.mismatch > 0 or self.gap_extend <= 0 or self.gap_open < self.gap_extend:
            raise ValueError('Require positive match, nonpositive mismatch, gap_open >= gap_extend > 0')
        if self.algorithm not in {'striped','scan'}: raise ValueError('Unknown SW algorithm')


def dna(sequence):
    value=sequence.upper()
    if not value or set(value)-set('ACGTN'): raise ValueError('Expected nonempty A/C/G/T/N sequence')
    if not any(b!='N' for b in value): raise ValueError('All-N sequence has undefined normalized similarity')
    return value


def reverse_complement(sequence):
    return sequence.translate(str.maketrans('ACGTN','TGCAN'))[::-1]


def reference_sw_score(a,b,config=None):
    """Small independent Gotoh DP oracle: O(len(a)*len(b)) time, O(len(b)) memory.

    A gap of length k costs gap_open + (k-1)*gap_extend. N comparisons score zero.
    Single orientation only; intended for tests and small inputs.
    """
    c=config or SWConfig();a=dna(a);b=dna(b)
    previous=[0]*(len(b)+1);vertical=[-10**30]*(len(b)+1);best=0
    for x in a:
        current=[0]*(len(b)+1);horizontal=-10**30
        for j,y in enumerate(b,1):
            vertical[j]=max(previous[j]-c.gap_open,vertical[j]-c.gap_extend)
            horizontal=max(current[j-1]-c.gap_open,horizontal-c.gap_extend)
            substitution=0 if 'N' in (x,y) else c.match if x==y else c.mismatch
            current[j]=max(0,previous[j-1]+substitution,vertical[j],horizontal)
            best=max(best,current[j])
        previous=current
    return best


class SmithWaterman:
    """Parasail SIMD score-only SW; immutable matrix can be shared across threads."""
    def __init__(self,config=None):
        import parasail
        self.parasail=parasail;self.config=config or SWConfig()
        self.matrix=parasail.matrix_create('ACGTN',self.config.match,self.config.mismatch)
        # Unknown bases do not contribute evidence, including N:N comparisons.
        for i in range(5):
            self.matrix[4,i]=0;self.matrix[i,4]=0

    def raw_score(self,a,b):
        c=self.config
        width=16 if min(len(a),len(b))*c.match < 30000 else 32
        result=getattr(self.parasail,f'sw_{c.algorithm}_{width}')(a,b,c.gap_open,c.gap_extend,self.matrix)
        if result.saturated:
            result=getattr(self.parasail,f'sw_{c.algorithm}_64')(a,b,c.gap_open,c.gap_extend,self.matrix)
        if result.saturated:raise OverflowError('SW score saturated even with 64-bit arithmetic')
        return int(result.score)

    def compare(self,a,b):
        a=dna(a);b=dna(b)
        score=self.raw_score(a,b);orientation='forward'
        if self.config.both_strands:
            reverse=self.raw_score(a,reverse_complement(b))
            if reverse>score:score=reverse;orientation='reverse_complement'
        self_a=self.config.match*(len(a)-a.count('N'))
        self_b=self.config.match*(len(b)-b.count('N'))
        similarity=2*score/(self_a+self_b)
        if not 0<=similarity<=1+1e-12:raise ValueError('Normalized SW score outside bounds')
        return {'score':score,'similarity':min(1.,similarity),'distance':max(0.,1-similarity),
                'shorter_normalized_similarity':min(1.,score/min(self_a,self_b)),
                'orientation':orientation}
