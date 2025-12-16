# Phase 5 Results: Integration and Synthesis

## Integrated Findings and Synthesis Results

This document provides a structured template for documenting Phase 6 (integration and synthesis) results for the AMR thesis project.

---

## Table of Contents

1. [Resistance Archetype Summary](#1-resistance-archetype-summary)
2. [Species-Environment Associations](#2-species-environment-associations)
3. [MDR Enrichment Analysis](#3-mdr-enrichment-analysis)
4. [Cross-Phase Coherence](#4-cross-phase-coherence)
5. [Final Documentation Checklist](#5-final-documentation-checklist)

---

## 1. Resistance Archetype Summary

> **Building on the resistance-based clusters identified in Phase 3**, this section characterizes the dominant resistance archetypes discovered through unsupervised structure identification.

### 1.1 Archetype Overview Table

| Cluster | N | MDR Rate | MAR Index | Archetype Name | Key Characteristics |
|---------|---|----------|-----------|----------------|---------------------|
| 1 | [n] | [%] | [mean] | [Name] | [Key antibiotics, resistance level] |
| 2 | [n] | [%] | [mean] | [Name] | [Key antibiotics, resistance level] |
| 3 | [n] | [%] | [mean] | [Name] | [Key antibiotics, resistance level] |
| ... | ... | ... | ... | ... | ... |

### 1.2 Detailed Archetype Profiles

#### Archetype 1: [Descriptive Name]

| Attribute | Value |
|-----------|-------|
| **Cluster** | 1 |
| **Size** | [N] isolates ([%] of dataset) |
| **MDR Rate** | [%] |
| **Mean MAR** | [value] |
| **Resistance Level** | High/Moderate/Low |

**High-Resistance Antibiotics** (mean > 1.5):
- [AB1]: [mean] — [class]
- [AB2]: [mean] — [class]

**Low-Resistance Antibiotics** (mean < 0.5):
- [AB1]: [mean] — [class]
- [AB2]: [mean] — [class]

**Characterization**: [Narrative description of this resistance pattern]

[Repeat for each archetype]

### 1.3 Archetype Comparison

| Characteristic | Archetype 1 | Archetype 2 | Archetype 3 |
|----------------|-------------|-------------|-------------|
| Resistance Level | [H/M/L] | [H/M/L] | [H/M/L] |
| MDR Rate | [%] | [%] | [%] |
| Primary Resistance Class | [class] | [class] | [class] |
| Dominant Species | [sp] | [sp] | [sp] |
| Primary Environment | [env] | [env] | [env] |

---

## 2. Species-Environment Associations

### 2.1 Species Distribution by Environment

| Species | Water | Fish | Hospital | Total | Primary Env |
|---------|-------|------|----------|-------|-------------|
| E. coli | [n] | [n] | [n] | [N] | [env] |
| K. pneumoniae | [n] | [n] | [n] | [N] | [env] |
| [Other] | [n] | [n] | [n] | [N] | [env] |
| **Total** | **[N]** | **[N]** | **[N]** | **[Total]** | - |

**Chi-Square Test**: χ² = [value], df = [df], p = [p-value]

### 2.2 Species-Environment Association Summary

| Species | Primary Environment | Secondary Environment | Ecological Notes |
|---------|--------------------|-----------------------|------------------|
| E. coli | [env] ([%]) | [env] ([%]) | [Notes] |
| K. pneumoniae | [env] ([%]) | [env] ([%]) | [Notes] |
| [Other species] | [env] ([%]) | [env] ([%]) | [Notes] |

### 2.3 Species MDR Rates

| Species | Total | MDR Count | MDR Rate | Fold Enrichment |
|---------|-------|-----------|----------|-----------------|
| [Species 1] | [N] | [n] | [%] | [×] |
| [Species 2] | [N] | [n] | [%] | [×] |
| [Species 3] | [N] | [n] | [%] | [×] |
| **Overall** | **[N]** | **[n]** | **[%]** | **1.0×** |

---

## 3. MDR Enrichment Analysis

### 3.1 MDR Enrichment by Category

#### By Cluster

| Cluster | N | MDR Rate | Fold Enrichment | Enrichment Level |
|---------|---|----------|-----------------|------------------|
| [Most enriched] | [N] | [%] | [×] | High |
| [Second] | [N] | [%] | [×] | Moderate/High |
| ... | ... | ... | ... | ... |
| [Least enriched] | [N] | [%] | [×] | Low |

#### By Region

| Region | N | MDR Rate | Fold Enrichment | Enrichment Level |
|--------|---|----------|-----------------|------------------|
| [Most enriched] | [N] | [%] | [×] | High/Moderate/Low |
| [Second] | [N] | [%] | [×] | High/Moderate/Low |
| [Third] | [N] | [%] | [×] | High/Moderate/Low |

#### By Environment

| Environment | N | MDR Rate | Fold Enrichment | Enrichment Level |
|-------------|---|----------|-----------------|------------------|
| [Most enriched] | [N] | [%] | [×] | High/Moderate/Low |
| [Second] | [N] | [%] | [×] | High/Moderate/Low |
| [Third] | [N] | [%] | [×] | High/Moderate/Low |

### 3.2 MDR Resistance Signature

**Antibiotics Most Associated with MDR**:

| Antibiotic | MDR Mean | Non-MDR Mean | Difference | MDR-Associated? |
|------------|----------|--------------|------------|-----------------|
| [AB1] | [value] | [value] | [diff] | Yes |
| [AB2] | [value] | [value] | [diff] | Yes |
| [AB3] | [value] | [value] | [diff] | Yes |
| ... | ... | ... | ... | ... |

**MDR Signature Pattern**: [Description of which antibiotics characterize MDR isolates]

### 3.3 MDR Hotspot Summary

| Hotspot Type | Category | MDR Rate | Fold Enrichment |
|--------------|----------|----------|-----------------|
| Cluster | [Cluster X] | [%] | [×] |
| Region | [Region] | [%] | [×] |
| Environment | [Environment] | [%] | [×] |
| Species | [Species] | [%] | [×] |

---

## 4. Cross-Phase Coherence

### 4.1 Phase Integration Summary

| Finding Type | Phase 3 (Clustering) | Phase 4 (Supervised) | Phase 5 (Regional) | Coherent? |
|--------------|---------------------|---------------------|-------------------|-----------|
| MDR patterns | Cluster [X] highest MDR | [Model] F1=[value] | [Region/Env] enriched | Yes/No |
| Species patterns | Cluster [X] = [Sp] dominant | Species discrimination F1=[value] | [Sp] in [Env] | Yes/No |
| Resistance breadth | Clusters vary in MAR | Feature importance: [ABs] | Regional variation | Yes/No |

### 4.2 Model Agreement with Clustering

| Comparison | Agreement Level | Evidence |
|------------|-----------------|----------|
| High-importance antibiotics vs. cluster-defining antibiotics | High/Moderate/Low | [Overlap description] |
| MDR discrimination vs. cluster MDR enrichment | High/Moderate/Low | [Correlation description] |
| Species discrimination vs. cluster species purity | High/Moderate/Low | [Alignment description] |

### 4.3 Coherence Statement

> "The integration of unsupervised clustering (Phase 3), supervised discrimination (Phase 4), and multivariate analysis (Phase 5) reveals [consistent/partially consistent/inconsistent] patterns: [summary of key coherent findings]. These findings support [main conclusion] while acknowledging [key limitations including no temporal inference, dataset dependency, and no clinical decision support applicability]."

---

## 5. Final Documentation Checklist

### Phase 8 Audit Checklist

| Item | Status | Location | Notes |
|------|--------|----------|-------|
| ✅ Preprocessing decision log | Complete | docs/methods/preprocessing.md | Decision table with justifications |
| ✅ Clustering parameter table | Complete | docs/methods/clustering.md | Ward linkage, Euclidean distance |
| ✅ Supervised evaluation framing | Complete | docs/methods/supervised_models.md | Discrimination terminology |
| ✅ Terminology enforcement | Complete | docs/methods/supervised_models.md | Language control table |
| ✅ Figure caption discipline | Complete | All results docs | Data source, method, scope |
| ✅ Limitations section | Complete | docs/limitations.md | No temporal, dataset-dependent |
| ✅ Reproducibility statement | Complete | docs/methods/deployment.md | Seeds, versions, environment |

### Results Documentation Completeness

| Results Section | Documented | Template Location |
|-----------------|------------|-------------------|
| Preprocessing results | Yes/No | docs/results/phase2_clusters.md |
| Clustering results | Yes/No | docs/results/phase3_discrimination.md |
| Supervised results | Yes/No | docs/results/phase3_discrimination.md |
| Regional/Environmental | Yes/No | docs/results/phase4_environment.md |
| Integration/Synthesis | Yes/No | docs/results/phase5_synthesis.md |

---

## Key Findings Summary

### 1. Primary Findings

1. **[Finding 1]**: [Description with supporting evidence from multiple phases]

2. **[Finding 2]**: [Description with supporting evidence from multiple phases]

3. **[Finding 3]**: [Description with supporting evidence from multiple phases]

### 2. Limitations

1. **No temporal inference**: Cross-sectional design; cannot establish temporal trends
2. **Dataset dependency**: Findings reflect this specific dataset and may not generalize
3. **No clinical decision support**: Results are for exploratory analysis only

### 3. Implications

[Discussion of what these findings mean for AMR surveillance, with appropriate caveats]

---

## Figure Captions

### Figure 6.1: Archetype Summary Visualization

> "Summary of resistance archetypes identified through hierarchical clustering (n=[N] isolates). Archetypes characterized by mean resistance profiles; MDR rates and environmental associations shown for post-hoc interpretation."

### Figure 6.2: MDR Enrichment Comparison

> "Multi-Drug Resistance enrichment by cluster, region, and environment. Fold enrichment calculated relative to overall MDR rate ([%]). Enrichment values > 1.0 indicate higher-than-expected MDR prevalence."

### Figure 6.3: Integration Summary

> "Cross-phase integration summary showing alignment between clustering results (Phase 3), supervised discrimination (Phase 4), and regional analysis (Phase 5). Consistent patterns across phases strengthen confidence in identified resistance archetypes."

---

## Document Version

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024 | Initial results template |

---

*This document is part of Phase 8 — Documentation & Reporting for the AMR Thesis Project.*

**See also**: [integration.md](../methods/integration.md) | [limitations.md](../limitations.md) | [DOCUMENTATION.md](../DOCUMENTATION.md)
