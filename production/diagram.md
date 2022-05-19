```mermaid
graph TD
A[L2G Model] -->B(Classifier) --> C[XGBoost]
A --> D[Genetic and functional <br> genomics features]
D --> E[Naive closest gene <br> footprint]
D --> F[Naive closest gene <br> TSS]
D --> |LOGO/LOGI|H[Chromatin interaction]
D --> |LOGO/LOGI|I[Colocalization <br>of molQTLs]
D --> |LOGO/LOGI|J[Pathogenicity prediction <br> from VEP and PolyPhen]
A --> L[GS Confidence]
L --> M[High]
L --> N[Medium]
L -.-> |Filtered out| O[Low]
A --> P[Cross validation set]
P --> Q["[0,1,2,3,4]"]
E --> |consists of| GA[dist_foot_min,<br> dist_foot_min_nbh,<br> dist_foot_ave,<br> dist_foot_ave_nbh]
F --> |consists of| GAB[dist_tss_min,<br> dist_tss_min_nbh,<br> dist_tss_ave,<br> dist_tss_ave_nbh]
GA --> |extracted in| GB[distance.py]
GAB --> |extracted in| GB[distance.py]:::module
GB --- R[toploci]:::input
GB --- S[credsets_v2d]:::input
GB --- T[Gene index]:::input
I --> |GWAS without sumstats| IA[PICS]
IA --> IB[eqtl_pics_clpp_max,<br> eqtl_pics_clpp_max_neglogp,<br> eqtl_pics_clpp_max_nhb,<br> pqtl_pics_clpp_max,<br> pqtl_pics_clpp_max_neglogp,<br> pqtl_pics_clpp_max_nhb]
IB --> |extracted in| IC[pics_coloc.py]:::module
IC --- S[credsets_v2d]:::input
IC --- U[credsets_qtl]:::input
I --> |GWAS with sumstats| ID[GCTA- COJO]
ID --> IE[eqtl_coloc_llr_max,<br> eqtl_coloc_llr_max_neglogp,<br> eqtl_coloc_llr_max_nbh,<br> pqtl_coloc_llr_max,<br> pqtl_coloc_llr_max_neglogp,<br> pqtl_coloc_llr_max_nbh]
IE --> |extracted in| IF[fm_coloc.py]:::module
IF --- V[coloc]:::input
IF --- U[credsets_qtl]:::input
H --> HA[DHS-gene promoter correlation]
H --> HB[Enhancer-TSS analysis of gene expression correlation]
H --> HC[PCHI-C from 27 different cell types]
HA --> HD[dhs_prmtr_max,<br> dhs_prmtr_max_nbh,<br> dhs_prmtr_ave,<br> dhs_prmtr_ave_nbh]
HD --> |extracted in| HE[dhs_prmtr.py]:::module
HE --- S
HE --- W[interval]:::input
HB --> |consists of| HF[enhc_tss_max,<br> enhc_tss_max_nbh,<br> enhc_tss_ave,<br> enhc_tss_ave_nbh]
HF --> |extracted in| HG[enhc_tss.py]:::module
HG --- S
HG --- W
HC --> |consists of| HH[pchicJung_max,<br> pchicJung_max_nbh,<br> pchicJung_ave,<br> pchicJung_ave_nbh]
HH --> |extracted in| HI[pchiJung.py]:::module
HI --- S
HI --- X[pchicJung]:::input
J --> JA[VEP]
JA --> |consists of| JC[vep_credset_max,<br> vep_credset_max_nbh,<br> vep_ave,<br> vep_ave_nhb]
JC --> |extracted in| JD[vep.py]:::module
JD --- S
JD --- Y[VEP]:::input
J --> JB[Polyphen]
JB --> |consists of| JE[polyphen_credset_max,<br> polyphen_credset_max_nbh,<br> polyphen_ave,<br> polyphen_ave_nhb]
JE --> |extracted in| JF[polyphen.py]:::module
JF --- S
JF --- Z[Polyphen]:::input

D --> DA[Other]
DA --> |consists of| DB[count_credset_95,<br> has_sumstats,<br> gene_count_lte_50k,<br> gene_count_lte_100k,<br> gene_count_lte_250k,<br> gene_count_lte_500k]
DB --> |extracted in| DC[others.py]:::module
DC --- S

classDef input fill:#F1AB86;
classDef module fill:#CCFBFE;
```
