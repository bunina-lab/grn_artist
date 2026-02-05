"""
Hierarchical grouping for enrichment terms from multiple databases
Supports: GO, Reactome, KEGG, Hallmark, MSigDB
"""

from goatools import obo_parser
from collections import defaultdict
import requests
import re
from typing import Dict, List, Optional, Set


class GOTermGrouper:
    """Group GO terms using the GO ontology hierarchy"""
    
    def __init__(self):
        """Initialize with GO ontology file"""
        print("Loading GO ontology...")
        from config import GO_OBO_FILE
        self.go_dag = obo_parser.GODag(GO_OBO_FILE)
        
    def get_term_id_from_name(self, term_name: str) -> Optional[str]:
        """Convert term name to GO ID"""
        clean_name = term_name.replace('_', ' ').lower().strip()
        
        for go_id, go_term in self.go_dag.items():
            if go_term.name.lower() == clean_name:
                return go_id
        return None
    
    def get_parent_at_level(self, go_id: str, level: int = 3):
        """Get parent term at specific level from root"""
        if go_id not in self.go_dag:
            return None
            
        go_term = self.go_dag[go_id]
        paths = self._get_all_paths_to_root(go_id)
        
        if not paths:
            return go_term
        
        longest_path = max(paths, key=len)
        
        if len(longest_path) > level:
            return self.go_dag[longest_path[-(level+1)]]
        else:
            return self.go_dag[longest_path[0]]
    
    def get_specific_parent(self, go_id: str) -> Optional[str]:
        """
        Get a specific, meaningful parent term using smart heuristics.
        Aims for terms like 'lipid metabolism' rather than generic 'metabolism'.
        """
        if go_id not in self.go_dag:
            return None
        
        go_term = self.go_dag[go_id]
        
        # Try different levels and pick the most specific meaningful one
        for level in range(5, 2, -1):  # Try levels 5, 4, 3
            parent = self.get_parent_at_level(go_id, level)
            if parent and parent.name != go_term.name:
                parent_name = parent.name.lower()
                # Avoid overly generic terms
                generic_terms = ['biological process', 'molecular function', 'cellular component',
                               'metabolic process', 'cellular process', 'biological regulation']
                if not any(generic in parent_name for generic in generic_terms):
                    return parent.name.title()
        
        # Fallback to level 4 if all else fails
        parent = self.get_parent_at_level(go_id, level=4)
        return parent.name.title() if parent else go_term.name.title()
    
    def _get_all_paths_to_root(self, go_id: str) -> List[List[str]]:
        """Get all paths from term to root"""
        paths = []
        
        def dfs(current_id, path):
            current_term = self.go_dag[current_id]
            
            if not current_term.parents:
                paths.append(path[:])
                return
            
            for parent in current_term.parents:
                path.append(parent.id)
                dfs(parent.id, path)
                path.pop()
        
        dfs(go_id, [go_id])
        return paths
    
    def _get_ancestors(self, go_id: str, ancestors_set: Set[str]):
        """Recursively get all ancestors of a term"""
        if go_id not in self.go_dag:
            return
        
        term = self.go_dag[go_id]
        ancestors_set.add(go_id)
        
        for parent in term.parents:
            if parent.id not in ancestors_set:
                self._get_ancestors(parent.id, ancestors_set)
    
    def group_terms(self, term_names: List[str], level: int = 3) -> Dict[str, str]:
        """Group GO terms under parent terms at specified level"""
        mapping = {}
        
        for term_name in term_names:
            go_id = self.get_term_id_from_name(term_name)
            if go_id:
                # Use smart parent selection
                parent_name = self.get_specific_parent(go_id)
                if parent_name:
                    mapping[term_name] = parent_name
                else:
                    mapping[term_name] = term_name.replace('_', ' ').title()
            else:
                print(f"Warning: Could not find GO ID for '{term_name}'")
                mapping[term_name] = term_name.replace('_', ' ').title()
        
        return mapping


class ReactomeGrouper:
    """Group Reactome pathways using hierarchy"""
    
    def __init__(self):
        self.base_url = "https://reactome.org/ContentService"
        self.hierarchy_cache = {}
        # More specific Reactome categories
        self.specific_categories = {
            'Lipid Metabolism': ['LIPID', 'FATTY_ACID', 'CHOLESTEROL', 'STEROID', 'PHOSPHOLIPID', 'SPHINGOLIPID'],
            'Carbohydrate Metabolism': ['GLUCOSE', 'GLYCOLYSIS', 'GLUCONEOGENESIS', 'GLYCOGEN', 'PENTOSE'],
            'Amino Acid Metabolism': ['AMINO_ACID', 'PROTEIN_DEGRADATION', 'UREA'],
            'Nucleotide Metabolism': ['NUCLEOTIDE', 'PURINE', 'PYRIMIDINE'],
            'Energy Metabolism': ['OXIDATIVE_PHOSPHORYLATION', 'ELECTRON_TRANSPORT', 'ATP', 'CITRATE'],
            'Epigenetic Regulation': ['EPIGENETIC', 'HISTONE', 'CHROMATIN', 'METHYLATION', 'ACETYLATION', 'MLL', 'WDR5'],
            'DNA Repair': ['DNA_REPAIR', 'TP53', 'P53', 'DNA_DAMAGE', 'BASE_EXCISION', 'NUCLEOTIDE_EXCISION'],
            'Transcription Regulation': ['TRANSCRIPTION', 'TRANSCRIPTIONAL', 'AP_1', 'TRANSCRIPTION_FACTOR', 'RNA_POL'],
            'Gene Expression': ['GENE_EXPRESSION', 'RNA_PROCESSING', 'SPLICING'],
            'Translation': ['TRANSLATION', 'RIBOSOME', 'PROTEIN_SYNTHESIS'],
            'Cell Cycle': ['CELL_CYCLE', 'MITOSIS', 'G1', 'G2', 'S_PHASE', 'M_PHASE'],
            'Apoptosis': ['APOPTOSIS', 'PROGRAMMED_CELL_DEATH', 'CASPASE'],
            'MAPK Signaling': ['MAPK', 'ERK', 'JNK', 'P38'],
            'PI3K-AKT Signaling': ['PI3K', 'AKT', 'MTOR', 'PTEN'],
            'Wnt Signaling': ['WNT', 'BETA_CATENIN'],
            'Notch Signaling': ['NOTCH'],
            'TGF-beta Signaling': ['TGF', 'SMAD'],
            'NF-kB Signaling': ['NFKB', 'NF_KAPPA_B', 'TNFA', 'TNF_ALPHA'],
            'JAK-STAT Signaling': ['JAK', 'STAT'],
            'GPCR Signaling': ['GPCR', 'G_PROTEIN'],
            'Receptor Signaling': ['RECEPTOR', 'LIGAND'],
            'Immune Response': ['IMMUNE', 'INTERFERON', 'INTERLEUKIN', 'CYTOKINE', 'CHEMOKINE'],
            'Innate Immunity': ['INNATE', 'TOLL_LIKE', 'TLR'],
            'Adaptive Immunity': ['ADAPTIVE', 'T_CELL', 'B_CELL', 'ANTIBODY', 'LYMPHOCYTE'],
            'Inflammation': ['INFLAMMATION', 'INFLAMMATORY'],
            'Complement System': ['COMPLEMENT'],
            'Hemostasis': ['HEMOSTASIS', 'COAGULATION', 'PLATELET', 'BLOOD_CLOTTING'],
            'Cardiovascular System': ['CARDIAC', 'HEART', 'BLOOD_CIRCULATION', 'VASCULAR'],
            'Neuronal System': ['NEURONAL', 'NEUROTRANSMITTER', 'SYNAPSE', 'NERVE', 'NGF'],
            'Hormone Signaling': ['HORMONE', 'INSULIN', 'GLUCAGON', 'ESTROGEN', 'ANDROGEN'],
            'Cell Adhesion': ['ADHESION', 'INTEGRIN', 'CELL_JUNCTION', 'FOCAL_ADHESION'],
            'ECM Organization': ['EXTRACELLULAR_MATRIX', 'ECM', 'COLLAGEN', 'LAMININ'],
            'Vesicle Transport': ['VESICLE', 'ENDOCYTOSIS', 'EXOCYTOSIS', 'GOLGI'],
            'Autophagy': ['AUTOPHAGY', 'LYSOSOME'],
            'Developmental Biology': ['DEVELOPMENT', 'DIFFERENTIATION', 'MORPHOGENESIS', 'EMBRYO'],
            'Muscle Biology': ['MUSCLE', 'MYOGENESIS', 'CONTRACTION'],
            'Metabolism of Cofactors': ['VITAMIN', 'COFACTOR', 'HEME'],
            'Xenobiotic Metabolism': ['XENOBIOTIC', 'DRUG_METABOLISM', 'CYTOCHROME_P450', 'DETOXIFICATION'],
        }
        
    def get_pathway_hierarchy(self, pathway_id: str) -> Optional[List[Dict]]:
        """Get the hierarchy for a Reactome pathway"""
        if pathway_id in self.hierarchy_cache:
            return self.hierarchy_cache[pathway_id]
        
        try:
            url = f"{self.base_url}/data/pathway/{pathway_id}/containedEvents"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                self.hierarchy_cache[pathway_id] = data
                return data
        except Exception as e:
            print(f"Error fetching Reactome hierarchy for {pathway_id}: {e}")
        
        return None
    
    def extract_pathway_id(self, pathway_name: str) -> Optional[str]:
        """Extract Reactome ID from pathway name (e.g., R-HSA-1234567)"""
        match = re.search(r'R-[A-Z]{3}-\d+', pathway_name)
        return match.group(0) if match else None
    
    def get_specific_category(self, pathway_name: str) -> str:
        """Get specific pathway category using keyword matching"""
        pathway_upper = pathway_name.upper()
        
        # Try to find the most specific match
        best_match = None
        max_keyword_length = 0
        
        for category, keywords in self.specific_categories.items():
            for keyword in keywords:
                if keyword in pathway_upper:
                    # Prefer longer, more specific keywords
                    if len(keyword) > max_keyword_length:
                        max_keyword_length = len(keyword)
                        best_match = category
        
        return best_match if best_match else "Other Pathways"
    
    def group_terms(self, term_names: List[str]) -> Dict[str, str]:
        """Group Reactome pathways into specific categories"""
        mapping = {}
        
        for term_name in term_names:
            category = self.get_specific_category(term_name)
            mapping[term_name] = category
        import os
        import json
        from config import REACTOME_RESOURCE_DIR

        # Path for caching Reactome term->category mapping
        resources_dir = REACTOME_RESOURCE_DIR
        os.makedirs(resources_dir, exist_ok=True)
        cache_file = os.path.join(resources_dir, "reactome_pathway_term_mapping.json")
        
        # Try to load mapping from cache if it exists
        cache_loaded = False
        mapping = {}
        if os.path.isfile(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                cache_mapping = json.load(f)
            # Only add mappings that are needed
            needed_terms = set(term_names)
            for term in needed_terms:
                if term in cache_mapping:
                    mapping[term] = cache_mapping[term]
            # If all terms are present in cache, use it
            if len(mapping) == len(term_names):
                cache_loaded = True

        if not cache_loaded:
            # Build mapping and update cache if necessary
            # If loading partially from cache, only map missing terms
            terms_to_map = [term for term in term_names if term not in mapping]
            ##freshly_mapped = {}

            for term_name in terms_to_map:
                category = self.get_specific_category(term_name)
                mapping[term_name] = category
            
            # Save updated mapping to cache
                # If file exists, update with newly mapped terms, else create fresh mapping
        if os.path.isfile(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                current_cache = json.load(f)
        else:
            current_cache = {}
        current_cache.update(mapping)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(current_cache, f, indent=4, sort_keys=True, ensure_ascii=False)
        return mapping


class KEGGGrouper:
    """Group KEGG pathways into functional categories"""
    
    def __init__(self):
        # More specific KEGG pathway categories
        self.specific_categories = {
            'Lipid Metabolism': ['FATTY_ACID', 'LIPID', 'STEROID', 'CHOLESTEROL', 'SPHINGOLIPID', 'GLYCEROPHOSPHOLIPID'],
            'Carbohydrate Metabolism': ['GLYCOLYSIS', 'GLUCONEOGENESIS', 'CITRATE_CYCLE', 'TCA', 'PENTOSE_PHOSPHATE', 'PYRUVATE', 'GLYCOGEN'],
            'Amino Acid Metabolism': ['AMINO_ACID', 'ALANINE', 'ASPARTATE', 'GLUTAMATE', 'VALINE', 'LEUCINE', 'ISOLEUCINE'],
            'Nucleotide Metabolism': ['NUCLEOTIDE', 'PURINE', 'PYRIMIDINE'],
            'Energy Metabolism': ['OXIDATIVE_PHOSPHORYLATION', 'ELECTRON_TRANSPORT', 'ATP'],
            'Cofactor/Vitamin Metabolism': ['VITAMIN', 'COFACTOR', 'FOLATE', 'THIAMINE', 'RIBOFLAVIN'],
            'Xenobiotic Metabolism': ['XENOBIOTIC', 'DRUG_METABOLISM', 'CYTOCHROME_P450'],
            'DNA Replication/Repair': ['DNA_REPLICATION', 'DNA_REPAIR', 'MISMATCH_REPAIR', 'BASE_EXCISION', 'NUCLEOTIDE_EXCISION'],
            'Transcription': ['TRANSCRIPTION', 'RNA_POLYMERASE', 'BASAL_TRANSCRIPTION'],
            'Translation': ['RIBOSOME', 'TRANSLATION', 'AMINOACYL_TRNA'],
            'RNA Processing': ['SPLICEOSOME', 'RNA_DEGRADATION', 'RNA_TRANSPORT'],
            'Protein Processing': ['PROTEIN_PROCESSING', 'PROTEIN_EXPORT', 'FOLDING_SORTING'],
            'Proteasome': ['PROTEASOME', 'UBIQUITIN'],
            'MAPK Signaling': ['MAPK'],
            'PI3K-AKT Signaling': ['PI3K_AKT'],
            'Wnt Signaling': ['WNT'],
            'Notch Signaling': ['NOTCH'],
            'Hedgehog Signaling': ['HEDGEHOG'],
            'TGF-beta Signaling': ['TGF_BETA'],
            'Hippo Signaling': ['HIPPO'],
            'JAK-STAT Signaling': ['JAK_STAT'],
            'NF-kB Signaling': ['NF_KAPPA_B', 'NFKB'],
            'TNF Signaling': ['TNF'],
            'mTOR Signaling': ['MTOR'],
            'VEGF Signaling': ['VEGF'],
            'Calcium Signaling': ['CALCIUM_SIGNALING'],
            'cAMP Signaling': ['CAMP_SIGNALING'],
            'Phosphatidylinositol Signaling': ['PHOSPHATIDYLINOSITOL'],
            'Receptor Signaling': ['RECEPTOR_INTERACTION', 'CYTOKINE_RECEPTOR'],
            'Cell Cycle': ['CELL_CYCLE', 'OOCYTE_MEIOSIS'],
            'Apoptosis': ['APOPTOSIS'],
            'Autophagy': ['AUTOPHAGY'],
            'Ferroptosis': ['FERROPTOSIS'],
            'Necroptosis': ['NECROPTOSIS'],
            'Cell Adhesion': ['FOCAL_ADHESION', 'ADHERENS_JUNCTION', 'TIGHT_JUNCTION', 'GAP_JUNCTION', 'CELL_ADHESION_MOLECULES'],
            'ECM Interaction': ['ECM_RECEPTOR'],
            'Endocytosis': ['ENDOCYTOSIS'],
            'Phagosome': ['PHAGOSOME'],
            'Peroxisome': ['PEROXISOME'],
            'Lysosome': ['LYSOSOME'],
            'Immune Response': ['IMMUNE', 'COMPLEMENT', 'TOLL_LIKE', 'NOD_LIKE', 'RIG_I'],
            'T Cell Signaling': ['T_CELL_RECEPTOR'],
            'B Cell Signaling': ['B_CELL_RECEPTOR'],
            'Chemokine Signaling': ['CHEMOKINE'],
            'Natural Killer Cell': ['NATURAL_KILLER'],
            'Leukocyte Migration': ['LEUKOCYTE_TRANSENDOTHELIAL'],
            'Platelet Activation': ['PLATELET_ACTIVATION'],
            'Hematopoiesis': ['HEMATOPOIETIC'],
            'Insulin Signaling': ['INSULIN'],
            'Adipocytokine Signaling': ['ADIPOCYTOKINE'],
            'PPAR Signaling': ['PPAR'],
            'AMPK Signaling': ['AMPK'],
            'Thyroid Signaling': ['THYROID'],
            'Estrogen Signaling': ['ESTROGEN'],
            'Circadian Rhythm': ['CIRCADIAN'],
            'Thermogenesis': ['THERMOGENESIS'],
            'Neuronal Signaling': ['NEUROACTIVE_LIGAND', 'NEUROTRANSMITTER'],
            'Synaptic Function': ['SYNAPSE', 'SYNAPTIC'],
            'Axon Guidance': ['AXON_GUIDANCE'],
            'Cardiac Function': ['CARDIAC_MUSCLE', 'ARRHYTHMOGENIC'],
            'Vascular Function': ['VASCULAR_SMOOTH_MUSCLE'],
            'Cancer Pathways': ['CANCER', 'CARCINOMA', 'GLIOMA', 'MELANOMA', 'LEUKEMIA'],
            'Neurodegenerative Disease': ['ALZHEIMER', 'PARKINSON', 'HUNTINGTON'],
            'Metabolic Disease': ['DIABETES', 'NON_ALCOHOLIC_FATTY_LIVER'],
            'Infectious Disease': ['INFECTION', 'VIRAL', 'BACTERIAL', 'PATHOGENIC'],
        }
    
    def group_terms(self, term_names: List[str]) -> Dict[str, str]:
        """Group KEGG pathways into specific functional categories"""
        mapping = {}
        
        for term_name in term_names:
            term_upper = term_name.upper()
            best_match = None
            max_keyword_length = 0
            
            for category, keywords in self.specific_categories.items():
                for keyword in keywords:
                    if keyword in term_upper:
                        # Prefer longer, more specific keywords
                        if len(keyword) > max_keyword_length:
                            max_keyword_length = len(keyword)
                            best_match = category
            
            mapping[term_name] = best_match if best_match else "Other Pathways"
        
        return mapping


class HallmarkGrouper:
    """Group Hallmark gene sets into functional categories"""
    
    def __init__(self):
        # Hallmark categories based on MSigDB groupings
        self.categories = {
            'Metabolism': [
                'ADIPOGENESIS', 'BILE_ACID_METABOLISM', 'CHOLESTEROL_HOMEOSTASIS',
                'FATTY_ACID_METABOLISM', 'GLYCOLYSIS', 'HEME_METABOLISM',
                'OXIDATIVE_PHOSPHORYLATION', 'XENOBIOTIC_METABOLISM'
            ],
            'Immune/Inflammatory': [
                'ALLOGRAFT_REJECTION', 'COMPLEMENT', 'INFLAMMATORY_RESPONSE',
                'INTERFERON_ALPHA_RESPONSE', 'INTERFERON_GAMMA_RESPONSE',
                'IL2_STAT5_SIGNALING', 'IL6_JAK_STAT3_SIGNALING', 'TNFA_SIGNALING_VIA_NFKB'
            ],
            'Signaling': [
                'ANDROGEN_RESPONSE', 'ESTROGEN_RESPONSE_EARLY', 'ESTROGEN_RESPONSE_LATE',
                'HEDGEHOG_SIGNALING', 'KRAS_SIGNALING_DN', 'KRAS_SIGNALING_UP',
                'NOTCH_SIGNALING', 'PI3K_AKT_MTOR_SIGNALING', 'TGF_BETA_SIGNALING',
                'WNT_BETA_CATENIN_SIGNALING'
            ],
            'Proliferation/Cell Cycle': [
                'E2F_TARGETS', 'G2M_CHECKPOINT', 'MITOTIC_SPINDLE', 'MYC_TARGETS_V1',
                'MYC_TARGETS_V2', 'SPERMATOGENESIS'
            ],
            'DNA Damage/Stress': [
                'APOPTOSIS', 'DNA_REPAIR', 'P53_PATHWAY', 'REACTIVE_OXYGEN_SPECIES_PATHWAY',
                'UNFOLDED_PROTEIN_RESPONSE', 'UV_RESPONSE_DN', 'UV_RESPONSE_UP'
            ],
            'Development/Differentiation': [
                'ANGIOGENESIS', 'APICAL_JUNCTION', 'APICAL_SURFACE', 'COAGULATION',
                'EPITHELIAL_MESENCHYMAL_TRANSITION', 'MYOGENESIS', 'PANCREAS_BETA_CELLS'
            ],
            'Cellular Structure': [
                'APICAL_JUNCTION', 'APICAL_SURFACE', 'MITOTIC_SPINDLE'
            ],
            'General Stress Response': [
                'HYPOXIA', 'PROTEIN_SECRETION', 'PEROXISOME', 'MTORC1_SIGNALING'
            ]
        }
    
    def group_terms(self, term_names: List[str]) -> Dict[str, str]:
        """Group Hallmark gene sets into functional categories"""
        mapping = {}
        
        for term_name in term_names:
            # Remove HALLMARK_ prefix if present
            clean_name = term_name.replace('HALLMARK_', '')
            matched = False
            
            for category, hallmarks in self.categories.items():
                if clean_name in hallmarks:
                    mapping[term_name] = category
                    matched = True
                    break
            
            if not matched:
                # Try partial matching
                for category, hallmarks in self.categories.items():
                    if any(h in clean_name for h in hallmarks):
                        mapping[term_name] = category
                        matched = True
                        break
            
            if not matched:
                mapping[term_name] = "Other Hallmarks"
        
        return mapping


class EnrichmentTermGrouper:
    """
    Unified interface for grouping enrichment terms from multiple databases
    """
    
    def __init__(self):
        """
        Initialize groupers for all supported databases
        
        Args:
            go_obo_file: Path to GO ontology file --> takes from config
        """

        self.groupers = {
            'go_biological_process': GOTermGrouper(),
            'go_molecular_function': GOTermGrouper(),
            'go_cellular_component': GOTermGrouper(),
            'reactome_pathways': ReactomeGrouper(),
            'kegg_pathways': KEGGGrouper(),
            'hallmark': HallmarkGrouper(),
        }
    
    def group_terms(self, 
                    terms: List[str], 
                    database: str, 
                    level: int = 3) -> Dict[str, str]:
        """
        Group terms from a specific database
        
        Args:
            terms: List of enrichment terms
            database: Database name (e.g., 'go_biological_process', 'reactome_pathways')
            level: Hierarchy level for GO terms (ignored for other databases)
        
        Returns:
            Dictionary mapping original term -> parent/group term
        """
        if database not in self.groupers:
            raise ValueError(f"Unknown database: {database}. Supported: {list(self.groupers.keys())}")
        
        grouper = self.groupers[database]
        
        # GO terms support level parameter
        if isinstance(grouper, GOTermGrouper):
            return grouper.group_terms(terms, level=level)
        else:
            return grouper.group_terms(terms)
    
    def group_mixed_terms(self, 
                         term_database_pairs: List[tuple],
                         go_level: int = 3) -> Dict[str, str]:
        """
        Group terms from multiple databases at once
        
        Args:
            term_database_pairs: List of (term, database) tuples
            go_level: Hierarchy level for GO terms
        
        Returns:
            Dictionary mapping original term -> parent/group term
        """
        mapping = {}
        
        for term, database in term_database_pairs:
            grouped = self.group_terms([term], database, level=go_level)
            mapping.update(grouped)
        
        return mapping
    
    def normalize_category_name(self, category: str) -> str:
        """
        Normalize category names to avoid duplicates with minor differences.
        E.g., 'G Protein Coupled Receptor Binding' and 'G Protein-Coupled Receptor Binding'
        should be treated as the same category.
        """
        # Convert to lowercase for comparison
        normalized = category.lower()
        
        # Remove hyphens and extra spaces
        normalized = re.sub(r'[-–—]', ' ', normalized)  # Replace all types of dashes
        normalized = re.sub(r'\s+', ' ', normalized)  # Normalize whitespace
        normalized = normalized.strip()
        
        # Title case for consistent formatting
        return ' '.join(word.capitalize() for word in normalized.split())
    
    def group_terms_from_prefixed_list(self, 
                                       term_list: List[str],
                                       go_level: int = 3) -> Dict[str, List[str]]:
        """
        Group enrichment terms that are prefixed with their database name.
        
        Args:
            term_list: List of terms prefixed with database (e.g., 'GO_LIPID_METABOLIC_PROCESS', 
                      'HALLMARK_GLYCOLYSIS', 'REACTOME_METABOLISM_OF_LIPIDS', 'KEGG_FATTY_ACID_METABOLISM')
            go_level: Hierarchy level for GO terms (default: 3)
        
        Returns:
            Dictionary mapping normalized parent term -> list of original child terms
        
        Example:
            >>> terms = ['GO_LIPID_METABOLIC_PROCESS', 'GO_FATTY_ACID_METABOLIC_PROCESS', 
                        'HALLMARK_GLYCOLYSIS']
            >>> grouped = grouper.group_terms_from_prefixed_list(terms)
            >>> # Returns: {'Lipid Metabolism': ['GO_LIPID_METABOLIC_PROCESS', ...], ...}
        """
        # Detect database and extract clean term name
        database_terms = {
            'go_biological_process': [],
            'go_molecular_function': [],
            'go_cellular_component': [],
            'reactome_pathways': [],
            'kegg_pathways': [],
            'hallmark': []
        }
        
        # Map original prefixed terms to database and clean names
        term_mapping = {}
        
        for prefixed_term in term_list:
            # Determine database from prefix
            if prefixed_term.startswith('GO_'):
                clean_term = prefixed_term[3:]  # Remove 'GO_' prefix
                # Default to biological_process, but could be enhanced to detect others
                database = 'go_biological_process'
                database_terms[database].append(clean_term)
                term_mapping[clean_term] = prefixed_term
                
            elif prefixed_term.startswith('GOBP_'):
                clean_term = prefixed_term[5:]  # Remove 'GOBP_' prefix
                database = 'go_biological_process'
                database_terms[database].append(clean_term)
                term_mapping[clean_term] = prefixed_term
                
            elif prefixed_term.startswith('GOMF_'):
                clean_term = prefixed_term[5:]  # Remove 'GOMF_' prefix
                database = 'go_molecular_function'
                database_terms[database].append(clean_term)
                term_mapping[clean_term] = prefixed_term
                
            elif prefixed_term.startswith('GOCC_'):
                clean_term = prefixed_term[5:]  # Remove 'GOCC_' prefix
                database = 'go_cellular_component'
                database_terms[database].append(clean_term)
                term_mapping[clean_term] = prefixed_term
                
            elif prefixed_term.startswith('HALLMARK_'):
                clean_term = prefixed_term  # Keep HALLMARK_ prefix for grouper
                database = 'hallmark'
                database_terms[database].append(clean_term)
                term_mapping[clean_term] = prefixed_term
                
            elif prefixed_term.startswith('REACTOME_'):
                clean_term = prefixed_term[9:]  # Remove 'REACTOME_' prefix
                database = 'reactome_pathways'
                database_terms[database].append(clean_term)
                term_mapping[clean_term] = prefixed_term
                
            elif prefixed_term.startswith('KEGG_'):
                clean_term = prefixed_term[5:]  # Remove 'KEGG_' prefix
                database = 'kegg_pathways'
                database_terms[database].append(clean_term)
                term_mapping[clean_term] = prefixed_term
                
            else:
                # No recognized prefix, try to detect database from term content
                if prefixed_term.startswith('HALLMARK_'):
                    database = 'hallmark'
                    clean_term = prefixed_term
                elif any(x in prefixed_term for x in ['METABOLISM_OF', 'SIGNALING_BY', 'ACTIVATION_OF_THE']):
                    database = 'reactome_pathways'
                    clean_term = prefixed_term
                else:
                    # Default to GO biological process
                    database = 'go_biological_process'
                    clean_term = prefixed_term
                    
                database_terms[database].append(clean_term)
                term_mapping[clean_term] = prefixed_term
        
        # Group terms by database
        all_parent_to_children = defaultdict(list)
        
        for database, terms in database_terms.items():
            if not terms:
                continue
            
            # Get parent mapping for this database
            if database.startswith('go_'):
                parent_mapping = self.group_terms(terms, database, level=go_level)
            else:
                parent_mapping = self.group_terms(terms, database)
            
            # Map back to original prefixed terms and normalize parent names
            for clean_term, parent in parent_mapping.items():
                normalized_parent = self.normalize_category_name(parent)
                original_term = term_mapping.get(clean_term, clean_term)
                all_parent_to_children[normalized_parent].append(original_term)
        
        # Sort children within each parent group
        result = {parent: sorted(children) for parent, children in all_parent_to_children.items()}
        
        return dict(result)