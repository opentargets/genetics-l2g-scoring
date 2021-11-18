#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Ed Mountjoy
#

import argparse
import os
import sys
import pyspark.sql
from pyspark.sql.types import *
from pyspark.sql.functions import *
from datetime import date
from pyspark.sql.window import Window
import networkx as nx

def main():

    # Parse args
    args = parse_args()

    # File paths
    out_gs = 'gs://genetics-portal-dev-staging/l2g/{v}/features/inputs/'.format(v=args.version)
    manifest = {
        'study': {
            'in': 'gs://genetics-portal-dev-staging/v2d/210922/studies.parquet',
            'out': out_gs + 'studies.parquet'
        },
        'toploci': {
            'in': 'gs://genetics-portal-dev-staging/v2d/210922/toploci.parquet',
            'out': out_gs + 'toploci.parquet'
        },
        'credsets_v2d': {
            'in_fm': 'gs://genetics-portal-dev-staging/finemapping/210923/credset/part-*.json.gz',
            'in_pics': 'gs://genetics-portal-dev-staging/v2d/210922/ld.parquet',
            'out': out_gs + 'credsets_v2d.parquet'
        },
        'credsets_qtl': {
            'in': 'gs://genetics-portal-dev-staging/finemapping/210923/credset/part-*.json.gz',
            'in_lut': 'gs://genetics-portal-input/luts/phenotype_id_lut.190629.json', # This is a lut to map from non-gene phenotype ID to Ensembl ID
            'out': out_gs + 'credsets_qtl.parquet'
        },
        'coloc': {
            'in': 'gs://genetics-portal-dev-staging/coloc/210927/coloc_processed_w_betas.parquet',
            'out': out_gs + 'coloc.parquet'
        },
        'v2g': {
            'in': 'gs://genetics-portal-dev-data/21.10/outputs/v2g/part-*.json',
            'out_interval': out_gs + 'interval.parquet',
            'out_vep': out_gs + 'vep.parquet'
        },
        'genes': {
            'in': 'gs://genetics-portal-dev-data/21.10/inputs/lut/homo_sapiens_core_104_38_genes.json.gz',
            'out': out_gs + 'genes.parquet'
        },
        'polyphen': {
            'in': 'gs://genetics-portal-staging/variant-annotation/190129/variant-annotation.parquet',
            'out': out_gs + 'polyphen.parquet'
        },
        'string': {
            'in_links': 'gs://genetics-portal-input/string/9606.protein.links.v11.0.txt',
            'in_aliases': 'gs://genetics-portal-input/string/9606.protein.aliases.v11.0.txt',
            'in_genes': out_gs + 'genes.parquet',
            'out': out_gs + 'string.parquet'
        },
        'clusters': {
            'in_toploci': out_gs + 'toploci.parquet',
            'in_credset': out_gs + 'credsets_v2d.parquet',
            'out': out_gs + 'clusters.parquet'
        },
        'pchicJung': {
            'in': 'gs://genetics-portal-dev-staging/v2g/interval/pchic/jung2019/190920/data.parquet',
            'out': out_gs + 'pchicJung.parquet'
        },
        'protein_attenuation': {
            'in': 'gs://genetics-portal-input/protein_attenuation/Sousa2019_142565_1_supp_344512_psqnlv.csv',
            'in_genes': 'gs://genetics-portal-input/luts/19.06_gene_symbol_synonym_map.json',
            'out': out_gs + 'proteinAttenuation.parquet'
        },
        'variant_index': {
            'in': 'gs://genetics-portal-staging/variant-annotation/190129/variant-annotation.parquet',
            'out': out_gs + 'variant-annotation.parquet'
        },
    }

    # Make spark session
    global spark
    spark = (
        pyspark.sql.SparkSession.builder
        .getOrCreate()
    )
    print('Spark version: ', spark.version)

    # Process datasets
    process_study_table(
        in_path=manifest['study']['in'],
        out_path=manifest['study']['out']
    )
    process_toploci_table(
        in_path=manifest['toploci']['in'],
        out_path=manifest['toploci']['out']
    )
    process_credsets_v2d_table(
        in_fm=manifest['credsets_v2d']['in_fm'],
        in_pics=manifest['credsets_v2d']['in_pics'],
        out_path=manifest['credsets_v2d']['out']
    )
    process_credsets_qtl_table(
        in_path=manifest['credsets_qtl']['in'],
        in_lut=manifest['credsets_qtl']['in_lut'],
        out_path=manifest['credsets_qtl']['out']
    )
    process_gene_table(
        in_path=manifest['genes']['in'],
        out_path=manifest['genes']['out']
    )
    process_coloc_table(
        in_path=manifest['coloc']['in'],
        in_genes=manifest['genes']['out'],
        out_path=manifest['coloc']['out']
    )
    process_v2g_table(
        in_path=manifest['v2g']['in'],
        out_interval=manifest['v2g']['out_interval'],
        out_vep=manifest['v2g']['out_vep']
    )
    process_polyphen(
        in_path=manifest['polyphen']['in'],
        out_path=manifest['polyphen']['out']
    )
    process_string(
        in_links=manifest['string']['in_links'],
        in_aliases=manifest['string']['in_aliases'],
        min_score=0,
        gene_info=manifest['string']['in_genes'],
        out_path=manifest['string']['out']
    )
    process_clusters(
        in_top_loci=manifest['clusters']['in_toploci'],
        in_credible_sets=manifest['clusters']['in_credset'],
        min_pp_clustering=0.001,
        out_path=manifest['clusters']['out']
    )
    process_pchicJung(
        in_path=manifest['pchicJung']['in'],
        out_path=manifest['pchicJung']['out']
    )
    process_protein_attenuation(
        in_path=manifest['protein_attenuation']['in'],
        in_genes=manifest['protein_attenuation']['in_genes'],
        out_path=manifest['protein_attenuation']['out']
    )
    process_variant_index(
        in_path=manifest['variant_index']['in'],
        in_credsets=manifest['credsets_v2d']['in_fm'],
        out_path=manifest['variant_index']['out']
    )

    return 0

def process_clusters(in_top_loci, in_credible_sets, min_pp_clustering,
                    out_path):
    ''' Clusters the top loci based on credible set information and adds
        cluster label
    Args:
        df (spark.df): data to add cluster labels to
        in_credible_sets (parquet): datasets containing credible set
            information
        min_pp_clustering (float): only tag variants with a PP > this will
            be used for clustering.
    '''
    # Get top loci
    toploci = (
        spark.read.parquet(in_top_loci)
        .select('study_id', 'chrom', 'pos', 'ref', 'alt')
        .drop_duplicates()
    )

    # Load credible set information
    credset = (
        spark.read.parquet(in_credible_sets)
        .filter(col('combined_postprob') >= min_pp_clustering)
        .select(
            col('study_id'),
            col('lead_chrom').alias('chrom'),
            col('lead_pos').alias('pos'),
            col('lead_ref').alias('ref'),
            col('lead_alt').alias('alt'),
            col('tag_chrom'),
            col('tag_pos'),
            col('tag_ref'),
            col('tag_alt')
        )
    )

    # Inner merge with toploci in order to drop none needed columns
    credset = credset.join(
        broadcast(toploci),
        on=['study_id', 'chrom', 'pos', 'ref', 'alt'],
        how='inner'
    )

    # # Debug
    # print('Warning, only using 1000 rows for locus overlap')
    # credset = credset.limit(1000)

    # Join the credible set table to itself to find overlapping loci
    overlaps = (
        credset.alias('left').join(
            credset.alias('right'),
            on=['tag_chrom', 'tag_pos', 'tag_ref', 'tag_alt'],
            how='inner'
        )
        # Drop the tag variants and deduplicate (1 row per overlap)
        .drop('tag_chrom', 'tag_pos', 'tag_ref', 'tag_alt')
        .drop_duplicates()
    )

    # Cluster loci
    cluster_df = build_and_cluster_network(overlaps)

    # Join cluster labels back to toploci
    toploci = toploci.join(
        cluster_df,
        on=['study_id', 'chrom', 'pos', 'ref', 'alt'],
        how='left'
    )

    # Add unknown cluster label
    toploci = toploci.withColumn('cluster_label',
        when(col('cluster_label').isNull(), 'cluster_unknown')
        .otherwise(col('cluster_label'))
    )

    # Write
    (
        toploci
        .repartitionByRange('study_id', 'chrom', 'pos')
        .write
        .parquet(
            out_path,
            mode='ignore'
        )
    )

    return 0

def build_and_cluster_network(df):
    ''' Build a network of nodes and edges then finds all connected
        subgraphs
    Args:
        df (spark df): dataframe listing overlapping loci
    Returns:
        df of study_id, chrom, pos, ref, alt, cluster_label
    '''
    # Spark to pandas
    pddf = df.toPandas()

    # Create edges list
    edges = zip(
        [tuple(x) for x in pddf.iloc[:, 0:5].values],
        [tuple(x) for x in pddf.iloc[:, 5:10].values]
    )

    # Build graph using networkx
    G = nx.Graph()
    G.add_edges_from(edges)

    # Find connect subgraphs
    cluster_data = []
    for i, subgraph in enumerate(nx.connected_component_subgraphs(G)):
        for node in subgraph.nodes:
            cluster_data.append(
                list(node) + ['cluster_{}'.format(i)]
            )

    # Make spark df
    resdf = (
        spark.createDataFrame(
            cluster_data,
            ['study_id', 'chrom', 'pos', 'ref', 'alt',
             'cluster_label']
        )
    )

    return resdf

def process_string(in_links, in_aliases, min_score, gene_info, out_path):
    ''' Loads the string DB dataset
    Args:
        in_links (file): STRING DB file
        in_aliases (file): STRING DB aliases file
        min_score (float): minimum STRING posterior probability
        gene_info (file): gene LUT in parquet format
        out_path (file)
    '''

    #
    # Load string data --------------------------------------------------------
    #

    # Load string data
    import_schema = (
        StructType()
        .add('p1', StringType())
        .add('p2', StringType())
        .add('string_score', LongType())
    )
    string = (
        spark.read.csv(
            in_links,
            schema=import_schema,
            header=True,
            sep=' '
        )
        .withColumn('string_score', col('string_score') / 1000)
    )

    # Only keep connections above certain score
    string = string.filter(col('string_score') >= min_score)

    # Load aliases
    import_schema = (
        StructType()
        .add('string_id', StringType())
        .add('gene_id', StringType())
        .add('source', StringType())
    )
    aliases = (
        spark.read.csv(
            in_aliases,
            schema=import_schema,
            header=True,
            sep='\t'
        )
        .filter(col('gene_id').startswith('ENSG'))
        .drop('source')
        .drop_duplicates(subset=['string_id'])
    )

    # Merge aliases to data
    string = (
        string.join(
            broadcast(aliases.withColumnRenamed('gene_id', 'g1').alias('p1')),
            col('p1') == col('p1.string_id')
        )
        .drop('string_id')
    )
    string = (
        string.join(
            broadcast(aliases.withColumnRenamed('gene_id', 'g2').alias('p2')),
            col('p2') == col('p2.string_id')
        )
        .drop('string_id')
    )

    #
    # Load and merge gene information -----------------------------------------
    #

    # Load
    gene = (
        spark.read.parquet(gene_info)
        .select(
            'gene_id', 'chrom', 'start', 'end'
        )
    )

    # Merge g1
    string = (
        string.join(
            broadcast(gene.alias('g1')),
            col('g1') == col('g1.gene_id'),
            how='left'
        )
        .drop('gene_id')
    )

    # Merge g2
    string = (
        string.join(
            broadcast(gene.alias('g2')),
            col('g2') == col('g2.gene_id')
        )
        .drop('gene_id')
    )

    #
    # Write -------------------------------------------------------------------
    #

    outdata = (
        string
        .select(
            col('g1').alias('gene_id_1'),
            col('p1').alias('string_id_1'),
            col('g1.chrom').alias('gene_1_chrom'),
            col('g1.start').alias('gene_1_start'),
            col('g1.end').alias('gene_1_end'),
            col('g2').alias('gene_id_2'),
            col('p2').alias('string_id_2'),
            col('g2.chrom').alias('gene_2_chrom'),
            col('g2.start').alias('gene_2_start'),
            col('g2.end').alias('gene_2_end'),
            'string_score'
        )
    )

    # Write
    (
        outdata
        .repartitionByRange('gene_id_1', 'gene_id_2')
        .write
        .parquet(
            out_path,
            mode='ignore'
        )
    )

    return 0

def process_polyphen(in_path, out_path):
    ''' Extacts (variant, gene, polyphen) from our variant index
    '''

    # Load
    data = (
        spark.read.parquet(in_path)
        .select(
            col('chrom_b38').alias('chrom'),
            col('pos_b38').alias('pos'),
            'ref',
            'alt',
            col('vep.transcript_consequences').alias('csq')
        )
    )

    # Explode transcripts, then extract required scores
    data = (
        data
        .withColumn('csq',
            explode(col('csq'))
        )
        .select(
            'chrom',
            'pos',
            'ref',
            'alt',
            col('csq.gene_id').alias('gene_id'),
            col('csq.transcript_id').alias('transcript_id'),
            col('csq.polyphen_score').alias('polyphen_score'),
        )
    )

    # Remove null score rows
    data = data.filter(
        col('polyphen_score').isNotNull()
    )

    # Keep the highest score per (variant, gene)
    window_spec = (
        Window
        .partitionBy('chrom', 'pos', 'ref', 'alt', 'gene_id')
        .orderBy(desc('polyphen_score'))
    )
    data = (
        data
        .withColumn('rn', row_number().over(window_spec))
        .filter(col('rn') == 1)
        .drop('rn')
    )

    # Save
    (
        data
        .repartitionByRange('chrom', 'pos', 'ref', 'alt')
        .write
        .parquet(
            out_path,
            mode='ignore'
        )
    )

    return 0

def process_study_table(in_path, out_path):
    ''' Copy study table
    '''
    # Load
    data = spark.read.parquet(in_path)
    
    # Fix Neale standing height, which has incorrect EFO
    data = (
        data
        .withColumn('trait_efos',
            when(
                col('study_id') == 'NEALE2_50_raw',
                array([lit('EFO_0004339')])
            ).otherwise(col('trait_efos'))
        )
        .withColumn('trait_category',
            when(
                col('study_id') == 'NEALE2_50_raw',
                lit('anthropometric measurement')
            ).otherwise(col('trait_category'))
        )
    )

    # Write
    (
        data
        .repartition(10)
        .write.parquet(
            out_path,
            mode='ignore'
        )
    )

    return 0

def process_pchicJung(in_path, out_path):
    ''' Copy Jung 2019 pchic data
    '''
    # Copy
    (
        spark.read.parquet(in_path)
        .write.parquet(
            out_path,
            mode='ignore'
        )
    )

    return 0

def process_toploci_table(in_path, out_path):
    ''' Process toploci table
    '''
    (
        spark.read.parquet(in_path)
        .withColumn('neglog_p', 
            -1 * (log10(col('pval_mantissa')) + col('pval_exponent')))
        .select('study_id', 'chrom', 'pos', 'ref', 'alt', 'neglog_p')
        .repartitionByRange('study_id', 'chrom', 'pos')
        .write.parquet(
            out_path,
            mode='ignore'
        )
    )

    return 0

def process_credsets_v2d_table(in_fm, in_pics, out_path):
    ''' Process the finemapping and pics credible set info
        Steps:
            - Only keep type = 'gwas' in fm table
            - filter both to pp > 0.0001
            - drop unneeded columns
            - merged fm and pics
            - create combined column that use fm if available, otherwise
              use pics
        Args:
            in_fm (json): credible set results from fine-mapping pipeline
            in_pics (parquet): credible set information from the PICS LD method
            out_path (parquet)
    '''
    # Load
    fm = spark.read.json(in_fm).persist()
    pics = spark.read.parquet(in_pics)

    # Create table showing whether each study has fine mapping results
    has_fm = (
        fm.select('study_id')
        .distinct()
        .withColumn('has_fm', lit(True))
    )

    # Prepare fm
    fm = (
        fm
        .filter(col('type') == 'gwas')
        .filter(col('postprob') >= 0.001)
        .select([
            'study_id',
            'lead_chrom',
            'lead_pos',
            'lead_ref',
            'lead_alt',
            'tag_chrom',
            'tag_pos',
            'tag_ref',
            'tag_alt',
            col('postprob').alias('fm_postprob'),
            col('is95_credset').alias('fm_is95')
        ])
    )

    # Prepare pics
    pics = (
        pics
        .filter(col('pics_postprob') >= 0.001)
        .select([
            'study_id',
            'lead_chrom',
            'lead_pos',
            'lead_ref',
            'lead_alt',
            'tag_chrom',
            'tag_pos',
            'tag_ref',
            'tag_alt',
            'pics_postprob',
            col('pics_95perc_credset').alias('pics_is95')
        ])
    )

    # Join fine mapping and PICs credible sets
    combined = fm.join(
        pics,
        on=['study_id',
            'lead_chrom',
            'lead_pos',
            'lead_ref',
            'lead_alt',
            'tag_chrom',
            'tag_pos',
            'tag_ref',
            'tag_alt'],
        how='outer'
    )

    # Join table showing whether each study has fm
    combined = combined.join(has_fm,
        on='study_id',
        how='left'
    )

    # Create new column using fm if available, otherwise pics
    combined = (
        combined
        .withColumn('combined_postprob',
            when(col('has_fm').isNotNull(),
                 col('fm_postprob'))
                 .otherwise(col('pics_postprob'))
        )
        .withColumn('combined_is95',
            when(col('has_fm').isNotNull(),
                 col('fm_is95'))
                 .otherwise(col('pics_is95'))
        )
        # Drop has_fm column as no longer needed
        .drop('has_fm')
    )

    # Repartition
    combined = combined.repartitionByRange(
        'study_id', 'lead_chrom', 'lead_pos'
    )

    # Write
    combined.write.parquet(
        out_path,
        mode='ignore'
    )

    return 0

def process_credsets_qtl_table(in_path, in_lut, out_path):
    ''' Extract QTL credible set info
        Steps:
            - Only keep type != 'gwas'
            - filter to pp > 0.0001
            - Add bool whether tag is sentinel
            - drop unneeded columns
        Args:
            in_path (json): credible set results from fine-mapping pipeline
            in_lut (json): phenotype ID to gene ID lookup table
            out_path (parquet)
    '''
    # Load
    fm = spark.read.json(in_path)

    #Filter qtl
    qtl = (
        fm
        .filter(col('type') != 'gwas')
        .filter(col('postprob') >= 0.001)
    )

    # Create column showing if the tag is the sentinel
    qtl = (
        qtl
        .withColumn('is_sentinel',
            when((
                (col('lead_chrom') == col('tag_chrom')) &
                (col('lead_pos') == col('tag_pos')) &
                (col('lead_ref') == col('tag_ref')) &
                (col('lead_alt') == col('tag_alt'))
            ), lit(True)).otherwise(lit(False))
        )
        .withColumn('qtl_neglog_p',
            -log10(col('tag_pval_cond'))
        )
    )

    # Load phenotype id lut and join
    lut = spark.read.json(in_lut)
    qtl = (
        # Add gene_id from lut for none ENSG phenotypes
        qtl.join(
            broadcast(lut),
            on='phenotype_id',
            how='left'
        )
        # Add gene_id for ENSG phenotypes
        .withColumn('gene_id',
            when(col('phenotype_id').startswith('ENSG'),
                 col('phenotype_id'))
            .otherwise(col('gene_id'))
        )
        # Filter rows without a gene_id
        .filter(col('gene_id').isNotNull())

    )

    # Drop unneeded cols
    qtl = qtl.select(
        'type',
        'study_id',
        'bio_feature',
        'phenotype_id',
        'gene_id',
        'lead_chrom',
        'lead_pos',
        'lead_ref',
        'lead_alt',
        'tag_chrom',
        'tag_pos',
        'tag_ref',
        'tag_alt',
        'postprob',
        'qtl_neglog_p',
        'is_sentinel'
    )

    qtl = qtl.repartitionByRange(
        'study_id', 'lead_chrom', 'lead_pos'
    )

    # Write
    qtl.write.parquet(
        out_path,
        mode='ignore'
    )

    return 0

def process_coloc_table(in_path, in_genes, out_path):
    ''' Process coloc info
        Steps:
            - Only keep right_type != 'gwas'
            - Filter on coloc_n_vars >= 250
            - drop unneeded columns
        Args:
            in_path (parquet): coloc table
            out_path (parquet)
    '''
    # Load
    coloc = spark.read.parquet(in_path)

    # Filter and select cols
    coloc = (
        coloc
        .filter(col('right_type') != 'gwas')
        .filter(col('coloc_n_vars') >= 250)
        .filter(~isnan(col('coloc_log2_h4_h3')))
        .select(
            'left_study',
            'left_chrom',
            'left_pos',
            'left_ref',
            'left_alt',
            'right_type',
            'right_study',
            'right_bio_feature',
            'right_phenotype',
            'right_chrom',
            'right_pos',
            'right_ref',
            'right_alt',
            col('right_gene_id').alias('gene_id'),
            'coloc_log2_h4_h3',
            'coloc_h4'
        )
    )

    # Keep only protein-coding genes by inner joining with filtered gene table
    genes = (
        spark.read.parquet(in_genes)
        .select('gene_id')
        .distinct()
    )
    coloc = coloc.join(
        genes,
        on='gene_id',
        how='inner'
    )

    # Repartition
    coloc = coloc.repartitionByRange(
        'left_study',
        'left_chrom',
        'left_pos'
    )

    # Write
    coloc.write.parquet(
        out_path,
        mode='ignore'
    )

    return 0

def process_v2g_table(in_path, out_interval, out_vep):
    ''' Extract interval and vep data from V2G
        Steps:
            - Only keep interval and vep columns
            - Output separately
        Args:
            in_path (json): v2g
            out_interval (parquet)
            out_vep (parquet)
    '''
    # Load
    v2g = spark.read.json(in_path)

    # Process interval
    (
        v2g
        .select(
            col('chr_id').alias('chrom'),
            col('position').alias('pos'),
            col('ref_allele').alias('ref'),
            col('alt_allele').alias('alt'),
            col('type_id').alias('interval_type'),
            col('source_id').alias('interval_source'),
            col('feature').alias('interval_feature'),
            'interval_score',
            'gene_id'
        )
        .drop_duplicates()
        .dropna()
        .repartitionByRange('chrom', 'pos')
        .write.parquet(
            out_interval,
            mode='ignore'
        )
    )

    # Process vep
    (
        v2g
        .select(
            col('chr_id').alias('chrom'),
            col('position').alias('pos'),
            col('ref_allele').alias('ref'),
            col('alt_allele').alias('alt'),
            col('fpred_max_label').alias('csq'),
            col('fpred_max_score').alias('csq_score'),
            'gene_id'
        )
        .drop_duplicates()
        .dropna()
        .repartitionByRange('chrom', 'pos')
        .write.parquet(
            out_vep,
            mode='ignore'
        )
    )

    return 0

def process_gene_table(in_path, out_path):
    ''' Process gene table
        Steps:
            - Keep protein coding only
        Args:
            in_path (json): v2g
            out_path (parquet)
    '''
    # Load
    genes = spark.read.json(in_path)

    # Filter and select
    genes = (
        genes
        .filter(col('biotype') == 'protein_coding')
        .select(
            'gene_id',
            col('chr').alias('chrom'),
            'start',
            'end',
            'tss',
            'fwdstrand',
            'gene_name',
            col('description').alias('gene_description'),
            col('biotype').alias('gene_biotype')
        )
        .repartition(10)
    )

    # Write
    genes.write.parquet(
        out_path,
        mode='ignore'
    )

    return 0

def process_protein_attenuation(in_path, in_genes, out_path):
    ''' Process protein attenuation table
        Args:
            in_path (csv): protein attenuation data
            in_genes (json): gene synonymn and name mapped to ensembl IDs
            out_path (parquet)
    '''

    # Load protein attenuation data
    import_schema = (
        StructType()
        .add('gene', StringType())
        .add('r_cnv_rna', DoubleType())
        .add('p_cnv_rna', DoubleType())
        .add('r_cnv_prot', DoubleType())
        .add('p_cnv_prot', DoubleType())
        .add('attenuation', DoubleType())
        .add('class', StringType())
    )
    data = spark.read.csv(
        in_path,
        schema=import_schema,
        header=True,
        sep=','
    )
    
    # Load gene data
    genes = (
        spark.read.json(in_genes)
    )

    # Make gene name map
    name_map = (
        genes
        .select(
            'gene_name',
            'gene_id',
        )
    )
    # Make synonym map
    synonym_map = (
        genes
        # Explode synonyms
        .select(
            explode(col('gene_synonyms')).alias('gene_name'),
            'gene_id'            
        )
        # Left anti join to remove names that exist in name_map
        .join(
            name_map,
            on='gene_name',
            how='left_anti'
        )
    )
    # Concatenate name and synonym maps
    gene_map = name_map.unionByName(synonym_map)
    
    # Map protein names to ensembl IDs
    data = (
        data
        .withColumnRenamed('gene', 'gene_name')
        .join(
            gene_map,
            on='gene_name',
            how='left'
        )
    ).cache()

    # Print warning that missing gene IDs
    n_missing = data.filter(col('gene_id').isNull()).count()
    if n_missing > 0:
        print('Warning: {0} rows with missing gene IDs in protein attenuation file'.format(
            n_missing))
        data.filter(col('gene_id').isNull()).show()
        # Drop missing rows
        data = data.filter(col('gene_id').isNotNull())

    # Warn if there are duplicated gene_ids
    n_duplicated = data.count() - data.drop_duplicates(subset=['gene_id']).count()
    if n_duplicated > 0:
        print('Warning: {0} gene_ids are duplicated in protein attenuation file. Dropping'.format(
            n_duplicated))
        # Drop duplicates randomly
        data = data.drop_duplicates(subset=['gene_id'])

    # Write
    (
        data
        .select(
            'gene_id',
            col('attenuation').alias('proteinAttenuation')
        )
        .write.parquet(
            out_path,
            mode='ignore'
        )

    )

    return 0

def process_variant_index(in_path, in_credsets, out_path):
    ''' Process variant index
        Args:
            in_path (csv): full variant index
            in_credsets (json): credible sets to take variants from
            out_path (parquet)
    '''

    # Load index
    varindex = (
        spark.read.parquet(in_path)
        .drop('locus', 'alleles', 'locus_GRCh38')
        .withColumnRenamed('chrom_b38', 'chrom')
        .withColumnRenamed('pos_b38', 'pos')
    )
    
    # Load variants from credset file
    variants = (
        spark.read.json(in_credsets)
        .select(
            col('tag_chrom').alias('chrom'),
            col('tag_pos').alias('pos'),
            col('tag_ref').alias('ref'),
            col('tag_alt').alias('alt')
        )
        .drop_duplicates()
    )

    # Inner merge to filter index
    varindex_filt = varindex.join(
        variants,
        on=['chrom', 'pos', 'ref', 'alt'],
        how='inner'
    )

    # Write filtered
    (
        varindex_filt
        .repartitionByRange('chrom', 'pos')
        .write.parquet(
            out_path,
            mode='ignore'
        )
    )

    return 0

def parse_args():
    """ Load command line args """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--version',
        metavar="<str>",
        help="Version number (default: todays date)",
        type=str,
        default=date.today().strftime("%y%m%d")
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    main()
