# Clusters 3 and 4: transcript choice, CDS length and biological interpretation

Both clusters are enriched for very long CDS sequences, but neither has a significant excess of non-MANE selections after correction. The top-ranked genes distinguish a zinc-finger-dominated cluster 3 from a more heterogeneous structural/motor/centrosomal cluster 4.

## Methods and scope

Uses the completed resolution-0.3, k=30 Louvain partition and the same HGNC snapshot used in reference preparation. Background: all 19,217 embedded genes. Each full cluster is compared with its complement, not with its own top 50. One-sided Fisher tests assess enrichment. Benjamini–Hochberg correction covers 12 annotation/length tests jointly; 2,114 HGNC gene-group tests are corrected separately (all groups with at least five background members, in both clusters). Gene-group annotations overlap. These are exploratory, unadjusted-for-GC/length associations, not causal or functional validation.

The top 50 are selected by the existing internal cosine-weighted graph centrality, not re-ranked by biology. Their annotation counts are descriptive. CDS length thresholds of 10 kb and 20 kb operationalize “very long”; genomic span ≥1 Mb is tested separately because introns were not embedded.

“Selected non-MANE” means an Ensembl canonical fallback in the Ensembl 110 GTF. It does not establish that the gene lacks MANE in newer annotation. All genes have an annotated selected CDS. The newer HGNC snapshot has only 65 genes with an empty MANE field, versus 236 GTF selections made via the canonical fallback.

## Annotation and length enrichment

| Cluster | Feature | Cluster n/N | Rest n/N | Odds ratio | BH q |
|---|---|---:|---:|---:|---:|
| 3 | selected_non_MANE | 14/643 | 222/18574 | 1.84 | 0.056 |
| 3 | no_MANE_in_HGNC_snapshot | 5/643 | 60/18574 | 2.42 | 0.113 |
| 3 | CDS_over_3998_bp | 44/643 | 1222/18574 | 1.04 | 0.628 |
| 3 | CDS_at_least_10000_bp | 17/643 | 104/18574 | 4.82 | 3.48e-06 |
| 3 | CDS_at_least_20000_bp | 3/643 | 8/18574 | 10.88 | 0.0121 |
| 3 | genomic_span_at_least_1Mb | 0/643 | 69/18574 | 0.00 | 1 |
| 4 | selected_non_MANE | 2/303 | 234/18914 | 0.53 | 0.97 |
| 4 | no_MANE_in_HGNC_snapshot | 1/303 | 64/18914 | 0.98 | 0.8 |
| 4 | CDS_over_3998_bp | 53/303 | 1213/18914 | 3.09 | 6.7e-10 |
| 4 | CDS_at_least_10000_bp | 8/303 | 113/18914 | 4.51 | 0.00202 |
| 4 | CDS_at_least_20000_bp | 4/303 | 7/18914 | 36.13 | 7.33e-05 |
| 4 | genomic_span_at_least_1Mb | 1/303 | 68/18914 | 0.92 | 0.8 |

Cluster 3 has 14/643 canonical fallbacks (2.18%; nominal p=0.028, BH q=0.056), so this is a weak trend rather than corrected evidence of enrichment. Cluster 4 has only 2/303 (0.66%).

Cluster 3 has a small long-CDS tail despite no excess of multi-context genes overall. Its 17 CDS sequences ≥10 kb include mucins (MUC4, MUC5AC, MUC5B, MUC12, MUC16, MUC17, MUC19), NBPF10/19/20, AHNAK/AHNAK2, FLG, NEB, EPPK1, SPATA31H1 and TNXB. None is among its top 50.

Cluster 4 has 53/303 multi-context genes (17.5%), versus 6.4% in its complement. Its eight CDS sequences ≥10 kb are DST, AKAP9, SYNE2, MACF1, UTRN, DMD, SYNE1 and PCNT. Five of these occur in its top 50.

## What the top 50 suggest

**Cluster 3:** all 50 use MANE Select; lengths are 1,266–3,129 bp (median 1,917), so every representative fits in one context. HGNC annotates 48/50 as C2H2 zinc-finger genes, 44/50 as KRAB-domain-containing, and 5/50 as SCAN-domain-containing (overlapping classes). In the whole cluster, 400/643 are C2H2 and 328/643 are KRAB genes. This is a strong gene-family/domain signal. For example, [ZNF235](https://www.ncbi.nlm.nih.gov/gene/9310) encodes KRAB domains and tandem zinc fingers. Work on [KRAB zinc-finger regulation](https://www.nature.com/articles/nature21683) provides a biological context, but the present analysis does not identify their targets or demonstrate a shared regulatory program.

The whole cluster also enriches keratin-associated, NBPF, mucin and metallothionein groups. Therefore the zinc-finger representatives do not describe every member. A plausible explanation is shared/repetitive coding-sequence architecture and paralog similarity; this remains a hypothesis until sequence identity, repeat content and GC-matched controls are evaluated.

**Cluster 4:** 49/50 representatives use MANE Select; GOLGA8N is the sole canonical fallback, and it does have a MANE entry in the HGNC snapshot. Median top-50 CDS length is 2,868 bp; 16 exceed 3,998 bp, five exceed 10 kb, and three exceed 20 kb. The list combines centrosomal/ciliary proteins (CDK5RAP2, CNTLN, CEP152, CEP112, CEP89, OFD1, IFT74), kinesins (KIF15, KIF4A/B, KIF27, KIF5C), Golgin A8 members, and large structural proteins (DST, SYNE2, MACF1, UTRN, DSP). HGNC top-50 annotations include five kinesins, five golgin-related genes, five centriolar satellite proteins, four spectrin-repeat-containing genes and three plakins.

Centrosomal interaction evidence connects [CEP152 and CDK5RAP2](https://pmc.ncbi.nlm.nih.gov/articles/PMC4004176/). A potentially broader link is elongated/coiled-coil or repeat-rich protein architecture: [KIF15](https://pmc.ncbi.nlm.nih.gov/articles/PMC3919264/) has a coiled-coil stalk; [RAD50](https://pubmed.ncbi.nlm.nih.gov/12152085/) has coiled-coil domains despite its DNA-repair role; and [golgins](https://pmc.ncbi.nlm.nih.gov/articles/PMC4254398/) include coiled-coil tethers. This cross-pathway pattern supports an architecture/composition hypothesis more directly than a single pathway label. It does not establish that GB.DNA explicitly encodes protein secondary structure; no per-gene coiled-coil annotation or prediction was run.

## Strongest HGNC family associations

| Cluster | HGNC group | Cluster members | Rest members | BH q |
|---|---|---:|---:|---:|
| 3 | KRAB domain containing | 328/643 | 45/18574 | <1e-300 |
| 3 | Zinc fingers C2H2-type | 400/643 | 363/18574 | <1e-300 |
| 3 | Keratin associated proteins | 74/643 | 20/18574 | 4.99e-89 |
| 3 | SCAN domain containing | 40/643 | 17/18574 | 1.26e-43 |
| 3 | NBPF members | 12/643 | 2/18574 | 4.61e-14 |
| 3 | Metallothioneins | 10/643 | 2/18574 | 2.4e-11 |
| 3 | Mucins | 11/643 | 8/18574 | 6.16e-10 |
| 3 | Speedy/RINGO cell cycle regulator family | 9/643 | 12/18574 | 1.35e-06 |
| 4 | Golgin related | 19/303 | 7/18914 | 8.17e-27 |
| 4 | Centriolar satellite proteins | 23/303 | 45/18914 | 2.28e-22 |
| 4 | Golgins | 9/303 | 2/18914 | 7.57e-13 |
| 4 | Structural maintenance of chromosomes proteins | 7/303 | 0/18914 | 4.78e-11 |
| 4 | Myosin heavy chains, class II | 8/303 | 6/18914 | 1.7e-09 |
| 4 | N-Ank proteins | 7/303 | 6/18914 | 5.82e-08 |
| 4 | CTAGE family | 6/303 | 2/18914 | 6.02e-08 |
| 4 | Keratins, type I | 8/303 | 20/18914 | 1.16e-06 |

The machine-readable results retain log10 p and q for extreme enrichments whose ordinary floating-point values underflow to zero.

## Implications for zero-shot mapping

Family/domain organization is encouraging evidence that the embedding retains sequence structure. It can also impede locus-specific retrieval: paralogous genes may cluster together even when the experiment requires distinguishing individual loci. Separate same-family retrieval from exact-gene accuracy, and stratify by GC, length and transcript-selection status. For long-CDS effects, compare single-context genes with multi-context centroids, or use length-matched exon-block references. GC/length-matched enrichment and coiled-coil/repeat annotation are sensible next controls; they have not been performed here.

## Reproduce

```sh
.venv/bin/python -m scripts.characterize_gene_clusters \
  results/gb_dna_300m_cds_2026-09-28/vector_db \
  results/gb_dna_300m_cds_2026-09-28/louvain_r0.3_k30/gene_clusters.csv \
  data/reference/GRCh38-ensembl-110/sources/hgnc_complete_set.tsv \
  results/gb_dna_300m_cds_2026-09-28/louvain_r0.3_k30/cluster_3_4_new_review \
  --clusters 3 4 --top-n 50
```

Current artifacts are in `results/gb_dna_300m_cds_2026-09-28/louvain_r0.3_k30/cluster_3_4_characterization/`: `analysis.json`, `annotation_length_enrichment.csv`, `hgnc_family_enrichment.csv`, and `top50_annotated.csv`.

## Cluster 3: all 50 representatives

| Rank | Gene | HGNC name | CDS bp | Selection |
|---:|---|---|---:|---|
| 1 | ZNF235 | zinc finger protein 235 | 2214 | MANE_Select |
| 2 | ZNF439 | zinc finger protein 439 | 1512 | MANE_Select |
| 3 | ZNF615 | zinc finger protein 615 | 2226 | MANE_Select |
| 4 | ZNF12 | zinc finger protein 12 | 2091 | MANE_Select |
| 5 | ZNF311 | zinc finger protein 311 | 1998 | MANE_Select |
| 6 | ZNF227 | zinc finger protein 227 | 2397 | MANE_Select |
| 7 | ZNF426 | zinc finger protein 426 | 1662 | MANE_Select |
| 8 | ZNF528 | zinc finger protein 528 | 1884 | MANE_Select |
| 9 | ZNF432 | zinc finger protein 432 | 1956 | MANE_Select |
| 10 | ZNF157 | zinc finger protein 157 | 1518 | MANE_Select |
| 11 | ZNF490 | zinc finger protein 490 | 1587 | MANE_Select |
| 12 | ZNF81 | zinc finger protein 81 | 1983 | MANE_Select |
| 13 | ZNF442 | zinc finger protein 442 | 1881 | MANE_Select |
| 14 | ZNF600 | zinc finger protein 600 | 2373 | MANE_Select |
| 15 | ZNF805 | zinc finger protein 805 | 1881 | MANE_Select |
| 16 | ZNF699 | zinc finger protein 699 | 1926 | MANE_Select |
| 17 | ZNF700 | zinc finger protein 700 | 2226 | MANE_Select |
| 18 | ZNF892 | zinc finger protein 892 | 1566 | MANE_Select |
| 19 | ZNF527 | zinc finger protein 527 | 1827 | MANE_Select |
| 20 | ZNF570 | zinc finger protein 570 | 1608 | MANE_Select |
| 21 | ZNF3 | zinc finger protein 3 | 1338 | MANE_Select |
| 22 | ZNF879 | zinc finger protein 879 | 1689 | MANE_Select |
| 23 | ZNF44 | zinc finger protein 44 | 1845 | MANE_Select |
| 24 | ZNF816 | zinc finger protein 816 | 1953 | MANE_Select |
| 25 | ZNF98 | zinc finger protein 98 | 1716 | MANE_Select |
| 26 | ZNF264 | zinc finger protein 264 | 1881 | MANE_Select |
| 27 | ZNF616 | zinc finger protein 616 | 2343 | MANE_Select |
| 28 | ZNF599 | zinc finger protein 599 | 1764 | MANE_Select |
| 29 | ZFP90 | ZFP90 zinc finger protein | 1908 | MANE_Select |
| 30 | ZNF7 | zinc finger protein 7 | 2058 | MANE_Select |
| 31 | ZSCAN12 | zinc finger and SCAN domain containing 12 | 1833 | MANE_Select |
| 32 | ZKSCAN7 | zinc finger with KRAB and SCAN domains 7 | 2262 | MANE_Select |
| 33 | ZNF761 | zinc finger protein 761 | 2238 | MANE_Select |
| 34 | ZNF69 | zinc finger protein 69 | 1698 | MANE_Select |
| 35 | ZNF620 | zinc finger protein 620 | 1266 | MANE_Select |
| 36 | ZNF347 | zinc finger protein 347 | 2517 | MANE_Select |
| 37 | ZSCAN29 | zinc finger and SCAN domain containing 29 | 2556 | MANE_Select |
| 38 | ZNF606 | zinc finger protein 606 | 2376 | MANE_Select |
| 39 | ZNF726 | zinc finger protein 726 | 1848 | MANE_Select |
| 40 | ZKSCAN5 | zinc finger with KRAB and SCAN domains 5 | 2517 | MANE_Select |
| 41 | ZNF20 | zinc finger protein 20 | 1596 | MANE_Select |
| 42 | ZNF175 | zinc finger protein 175 | 2133 | MANE_Select |
| 43 | TEX13C | TEX13 family member C | 2979 | MANE_Select |
| 44 | ZSCAN20 | zinc finger and SCAN domain containing 20 | 3129 | MANE_Select |
| 45 | NBPF15 | NBPF member 15 | 2010 | MANE_Select |
| 46 | ZNF221 | zinc finger protein 221 | 1851 | MANE_Select |
| 47 | ZNF880 | zinc finger protein 880 | 1731 | MANE_Select |
| 48 | ZNF836 | zinc finger protein 836 | 2808 | MANE_Select |
| 49 | ZNF624 | zinc finger protein 624 | 2595 | MANE_Select |
| 50 | ZNF30 | zinc finger protein 30 | 1869 | MANE_Select |

## Cluster 4: all 50 representatives

| Rank | Gene | HGNC name | CDS bp | Selection |
|---:|---|---|---:|---|
| 1 | KIF15 | kinesin family member 15 | 4164 | MANE_Select |
| 2 | CDK5RAP2 | CDK5 regulatory subunit associated protein 2 | 5679 | MANE_Select |
| 3 | CNTLN | centlein | 4218 | MANE_Select |
| 4 | CEP152 | centrosomal protein 152 | 5130 | MANE_Select |
| 5 | CCDC144A | coiled-coil domain containing 144A | 4275 | MANE_Select |
| 6 | BBOF1 | basal body orientation factor 1 | 1587 | MANE_Select |
| 7 | CEP112 | centrosomal protein 112 | 2865 | MANE_Select |
| 8 | RAD50 | RAD50 double strand break repair protein | 3936 | MANE_Select |
| 9 | DST | dystonin | 23454 | MANE_Select |
| 10 | CEP89 | centrosomal protein 89 | 2349 | MANE_Select |
| 11 | AKAP9 | A-kinase anchoring protein 9 | 11721 | MANE_Select |
| 12 | CCDC102B | coiled-coil domain containing 102B | 1539 | MANE_Select |
| 13 | ANKRD26 | ankyrin repeat domain 26 | 5130 | MANE_Select |
| 14 | BICD1 | BICD cargo adaptor 1 | 2925 | MANE_Select |
| 15 | FHAD1 | forkhead associated phosphopeptide binding domain 1 | 4437 | MANE_Select |
| 16 | KIF4B | kinesin family member 4B | 3702 | MANE_Select |
| 17 | KIF27 | kinesin family member 27 | 4203 | MANE_Select |
| 18 | EVI5 | ecotropic viral integration site 5 | 2478 | MANE_Select |
| 19 | SPDL1 | spindle apparatus coiled-coil protein 1 | 1815 | MANE_Select |
| 20 | CGNL1 | cingulin like 1 | 3906 | MANE_Select |
| 21 | CCDC30 | coiled-coil domain containing 30 | 2814 | MANE_Select |
| 22 | KIF4A | kinesin family member 4A | 3696 | MANE_Select |
| 23 | CNTRL | centriolin | 6975 | MANE_Select |
| 24 | GOLGA8N | golgin A8 family member N | 1896 | Ensembl_canonical |
| 25 | OFD1 | OFD1 centriole and centriolar satellite protein | 3036 | MANE_Select |
| 26 | ODF2 | outer dense fiber of sperm tails 2 | 2736 | MANE_Select |
| 27 | GOLGA8O | golgin A8 family member O | 1896 | MANE_Select |
| 28 | GOLGA8Q | golgin A8 family member Q | 1896 | MANE_Select |
| 29 | CCDC88A | coiled-coil and HOOK domain protein 88A | 5613 | MANE_Select |
| 30 | SYNE2 | spectrin repeat containing nuclear envelope protein 2 | 20721 | MANE_Select |
| 31 | CEP55 | centrosomal protein 55 | 1392 | MANE_Select |
| 32 | JAKMIP2 | janus kinase and microtubule interacting protein 2 | 2460 | MANE_Select |
| 33 | HOOK3 | hook microtubule tethering protein 3 | 2154 | MANE_Select |
| 34 | NDC80 | NDC80 kinetochore complex component | 1926 | MANE_Select |
| 35 | KIF5C | kinesin family member 5C | 2871 | MANE_Select |
| 36 | TAX1BP1 | Tax1 binding protein 1 | 2367 | MANE_Select |
| 37 | IFT74 | intraflagellar transport 74 | 1800 | MANE_Select |
| 38 | CCDC149 | coiled-coil domain containing 149 | 1605 | MANE_Select |
| 39 | GOLGA8R | golgin A8 family member R | 1893 | MANE_Select |
| 40 | MIPOL1 | mirror-image polydactyly 1 | 1326 | MANE_Select |
| 41 | MACF1 | microtubule actin crosslinking factor 1 | 22665 | MANE_Select |
| 42 | CCDC171 | coiled-coil domain containing 171 | 3978 | MANE_Select |
| 43 | UTRN | utrophin | 10299 | MANE_Select |
| 44 | CCDC146 | coiled-coil domain containing 146 | 2865 | MANE_Select |
| 45 | SPECC1 | sperm antigen with calponin homology and coiled-coil domains 1 | 3204 | MANE_Select |
| 46 | GOLGA8M | golgin A8 family member M | 1896 | MANE_Select |
| 47 | RUFY2 | RUN and FYVE domain containing 2 | 1818 | MANE_Select |
| 48 | KRT23 | keratin 23 | 1266 | MANE_Select |
| 49 | DSP | desmoplakin | 8613 | MANE_Select |
| 50 | RUFY3 | RUN and FYVE domain containing 3 | 1860 | MANE_Select |
