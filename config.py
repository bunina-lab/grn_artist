MSIG_DATABASE_KEYS = [
    'reactome_pathways', 
    #'go_cellular_component',
    #'chemical_and_genetic_perturbations', 
    #'immunesigdb',
    #'tf_targets_legacy',
    #'cell_type_signatures',
    'go_biological_process',
    #'positional', 
    #'mirna_targets_mirdb',
    #'tf_targets_gtrf', 
    #'cancer_modules', 
    #'vaccine_response',
    'go_molecular_function',
    #'oncogenic_signatures',
    #'cancer_cell_atlas',
    'hallmark',
    #'pid_pathways', 
    'kegg_pathways',
    #'human_phenotype_ontology',
    #'wikipathways',
    #'cancer_gene_neighborhoods', 
    #'mirna_targets_legacy',
    #'biocarta_pathways', 
    #'kegg_medicus_pathways'
    ]

import os

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

ORCA_BIN = os.path.join(BASE_DIR, "bin", "orca")