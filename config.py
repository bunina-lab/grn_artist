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

DECOUPLER_RESOURCE_DIR = os.path.join(BASE_DIR, "resources", "decoupler")
REACTOME_RESOURCE_DIR = os.path.join(BASE_DIR, "resources", "reactome")
GO_RESOURCE_DIR = os.path.join(BASE_DIR, "resources", "go")
GO_OBO_FILE=os.path.join(GO_RESOURCE_DIR,"go-basic.obo")

CENTRALITY_COLOURS = {
    "betweenness_centrality": "#FF0000",      # Pure Red
    "eigenvector_centrality": "darkmagenta",     # 
    "pagerank": "gold",                   #
    "degree_centrality": "#8B0000",          # Dark Red
    "in_degree_centrality": "#FF1493",       # Deep Pink (red-pink)
    "out_degree_centrality": "#C71585",      # Medium Violet Red (red-violet)
    "harmonic_centrality": "darkslategrey",        # 
    "closeness_centrality": "#E53935",       # Material Red 600
    "katz_centrality": "slateblue",
    #"local_efficiency": "#EF5350",           # Material Red 400
}


COMMUNITY_COLOURS = [
    '#FF6B6B',  # Red
    '#4ECDC4',  # Teal
    'ivory', 
    "Cyan",
    "Yellow",
    "Royalblue",
    "plum",
    "seashell",
    "lightsteelblue",
    "lightgreen",
    "violet"
]