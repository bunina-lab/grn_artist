## This script will overlap the nodes (which would be TFs and genes) to decoupler databases with the decoupler tool

"""
decouplR paper

https://academic.oup.com/bioinformaticsadvances/article/2/1/vbac016/6544613?login=true
"""

# using msigdb
#msigdb = dc.op.get_resource("MSigDB")

## from version 2.0.0
### https://decoupler.readthedocs.io/en/latest/notebooks/scell/rna_sc.html

msigdb = dc.op.resource("MSigDB", organism="human")
msigdb

def calc_db_score(db, adata, tmin=5, verbose=False):
    """
    Calculates scores with different methods:
    ora, ulm, mlm
    Adds score_* to .obsm inplace
    """
    try:
        dc.mt.mlm(adata, net=db, verbose=verbose, tmin=tmin)
    except Exception as e:
        print("multivariate fitting does not work:\n{error}".format(error=e))
    dc.mt.ulm(adata, net=db, verbose=verbose, tmin=tmin)
    dc.mt.ora(adata, net=db, verbose=verbose, tmin=tmin)

## input: msigdb pandas dataframe
#	genesymbol	collection	geneset
#0	A1BG	reactome_pathways	REACTOME_HEMOSTASIS
#1	A1BG	go_cellular_component	GOCC_PLATELET_ALPHA_GRANULE_LUMEN
#2	A1BG	chemical_and_genetic_perturbations	CHENG_IMPRINTED_BY_ESTRADIOL
#3	A1BG	immunesigdb	GSE13522_WT_VS_IFNG_KO_SKING_T_CRUZI_Y_STRAIN_...
#4	A1BG	immunesigdb	GSE25088_CTRL_VS_IL4_AND_ROSIGLITAZONE_STIM_MA...
#...	...	...	...
#5895457	ZZZ3	mirna_targets_mirdb	MIR656_3P
#5895458	ZZZ3	mirna_targets_mirdb	MIR513B_5P
#5895459	ZZZ3	mirna_targets_mirdb	MIR449C_5P
#5895460	ZZZ3	tf_targets_gtrf	SETD7_TARGET_GENES
#5895461	ZZZ3	mirna_targets_mirdb	MIR4719
#5895462 rows × 3 columns


## input: genes with graph statistics
#       degree_centrality	is_in_dominating_set    leiden_cluster
#ESRRG	0.133315	True    1
#A2M	0.000557	False   2
#ZNF331	0.107988	True    1
#PPARG	0.193710	True    3
#AADACL3	0.000278	False   1
#...	...	... 
#ZNF563	0.000278	False   3
#ZNF641	0.001113	False   2
#ZNF765	0.000557	False   2
#ZNF792	0.004453	False   2
#ZNF821	0.002783	False   1



### output:
## leiden_dict = { 1 : {"reactome_pathways" : set(["go_cellular_component", ...]), "REACTOME_HEMOSTASIS": set([...])} }



import decoupler as dc

ct = dc.op.collectri()
ct

"""
Wrapper to access CollecTRI gene regulatory network. CollecTRI is a comprehensive resource containing a curated collection of transcription factors (TFs) and their target genes. It is an expansion of DoRothEA. Each interaction is weighted by its mode of regulation (either positive or negative).

ex:
source	target	weight	resources	references	sign_decision
0	MYC	TERT	1.0	DoRothEA-A;ExTRI;HTRI;NTNU.Curated;Pavlidis202...	10022128;10491298;10606235;10637317;10723141;1...	PMID
1	SPI1	BGLAP	1.0	ExTRI	10022617	default activation
2	SMAD3	JUN	1.0	ExTRI;NTNU.Curated;TFactS;TRRUST	10022869;12374795	PMID
3	SMAD4	JUN	1.0	ExTRI;NTNU.Curated;TFactS;TRRUST	10022869;12374795	PMID
4	STAT5A	IL2	1.0	ExTRI	10022878;11435608;17182565;17911616;22854263;2...	default activation
...	...	...	...	...	...	...
42985	NFKB	hsa-miR-143-3p	1.0	ExTRI	19472311	default activation
42986	AP1	hsa-miR-206	1.0	ExTRI;GEREDB;NTNU.Curated	19721712	PMID
42987	NFKB	hsa-miR-21	1.0	ExTRI	20813833;22387281	default activation
42988	NFKB	hsa-miR-224-5p	1.0	ExTRI	23474441;23988648	default activation
42989	AP1	hsa-miR-144-3p	1.0	ExTRI	23546882	default activation
"""



import decoupler as dc

do = dc.op.dorothea()
do

"""
DoRothEA gene regulatory network [GAHI+19].

Wrapper to access DoRothEA gene regulatory network. DoRothEA is a comprehensive resource containing a curated collection of transcription factors (TFs) and their target genes. Each interaction is weighted by its mode of regulation (either positive or negative) and by its confidence level.
"""


import decoupler as dc

hm = dc.op.hallmark()
hm

"""
Hallmark gene sets [LSP+11].

Hallmark gene sets summarize and represent specific well-defined biological states or processes and display coherent expression.
"""