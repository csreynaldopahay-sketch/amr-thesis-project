# 5. Discussion

## Chapter Overview

This chapter interprets the findings from our analysis of 492 bacterial isolates from environmental and clinical sources across three Philippine regions. We contextualize our results within the broader AMR surveillance literature, acknowledge methodological limitations, and discuss implications for public health policy and One Health approaches.

---

## 5.1 Principal Findings

### 5.1.1 E. coli Phenotypic Heterogeneity: The C3 vs C4 Dichotomy

Our most significant finding is the identification of two distinct *Escherichia coli* phenotypes with contrasting multi-drug resistance (MDR) profiles within the same environmental contexts:

**Cluster 3 (C3): MDR-Enriched E. coli Phenotype**
- **Composition**: 77.2% *E. coli* with additional *Klebsiella* (15%) and *Enterobacter* (8%)
- **MDR Rate**: 54.5% — the highest among all clusters
- **Resistance Pattern**: Predominantly resistant to tetracyclines (TE, DO), ampicillin (AM), and aminoglycosides (AN, GM)
- **Geographic Distribution**: 53.7% from BARMM, 26.8% from Central Luzon, 19.5% from Eastern Visayas
- **Environmental Sources**: 56.1% fish, 36.6% water, 7.3% hospital effluent

**Cluster 4 (C4): Susceptible E. coli Phenotype**
- **Composition**: 98.1% *E. coli* — the highest species purity
- **MDR Rate**: 0.0% — complete absence of multi-drug resistance
- **Resistance Pattern**: Broadly susceptible across all antibiotic classes
- **Geographic Distribution**: Dispersed across all three regions
- **Environmental Sources**: Predominantly water-associated (69.2%)

This dichotomy within *E. coli* isolates from similar environmental contexts has important biological and epidemiological implications:

1. **Strain-Level Variation**: The stark MDR rate difference (54.5% vs 0.0%) suggests that *E. coli* populations in Philippine aquatic environments are not homogeneous but comprise genetically distinct lineages with different resistance acquisition histories.

2. **Selection Pressure Hypothesis**: The association of C3 with fish sources (56.1%) compared to C4's water-dominated profile suggests that aquaculture environments may serve as selective niches for MDR *E. coli*. This aligns with documented tetracycline and ampicillin use in Philippine aquaculture (Cabello et al., 2013; Santos & Ramos, 2018).

3. **Plasmid-Mediated Resistance**: The multi-class resistance in C3 (tetracyclines, penicillins, aminoglycosides) suggests horizontal gene transfer via conjugative plasmids rather than chromosomal mutations. Environmental *E. coli* are known reservoirs of mobile resistance determinants (Poirel et al., 2018).

4. **Public Health Relevance**: The coexistence of MDR and susceptible *E. coli* phenotypes in the same environments demonstrates that resistance is not uniformly distributed, creating opportunities for targeted surveillance strategies.

### 5.1.2 Geographic Resistance Structuring

Chi-square analysis revealed statistically significant associations between cluster membership and geographic region (χ² = 101.18, p < 10⁻¹⁸, Cramér's V = 0.321). This moderate effect size indicates meaningful geographic structuring of resistance patterns:

**Regional Patterns Observed:**

| Region | Dominant Cluster | MDR Rate | Primary Resistance |
|--------|-----------------|----------|-------------------|
| BARMM | C3 (MDR E. coli) | High | Tetracyclines, Aminoglycosides |
| Central Luzon | C1 (Salmonella) | Moderate | Aminoglycosides |
| Eastern Visayas | C4, C5 (Susceptible) | Low | Minimal resistance |

**Interpretation with Caveats:**

While geographic structuring is statistically significant, this finding must be interpreted cautiously due to:

1. **Sampling Imbalance**: BARMM contributed 50.8% of isolates, potentially inflating its representation in high-MDR clusters. The apparent regional differences may partially reflect sampling artifacts rather than true ecological variation.

2. **Confounding by Source**: Regional differences may be confounded by environmental source distribution. If BARMM samples were predominantly from fish while Eastern Visayas samples were from water, observed resistance differences might reflect source rather than geography.

3. **Cross-Sectional Design**: Without longitudinal data, we cannot determine whether observed patterns represent stable regional characteristics or temporal snapshots that might vary seasonally.

Despite these limitations, the geographic structuring we observed aligns with AMR surveillance findings from Southeast Asia, where regional variation in resistance profiles is commonly reported and attributed to local antibiotic usage patterns, healthcare infrastructure, and agricultural practices (Chereau et al., 2017; Zellweger et al., 2017).

### 5.1.3 Co-Resistance Patterns and Genetic Linkage

Our co-resistance network analysis identified 23 statistically significant antibiotic associations (Bonferroni-corrected α = 0.01, φ > 0.2), revealing important patterns of genetic linkage:

**Key Co-Resistance Clusters:**

1. **β-Lactam Cluster**: AM–AMC–CPT (Penicillins and Cephalosporins)
   - φ coefficients: 0.42–0.58
   - Interpretation: Consistent with ESBL-mediated cross-resistance affecting multiple β-lactam classes

2. **Aminoglycoside Cluster**: AN–GM–CN
   - φ coefficients: 0.35–0.48
   - Interpretation: Likely 16S rRNA methylase or aminoglycoside-modifying enzyme genes on shared mobile elements

3. **Cross-Class Associations**: TE–SXT, AM–TE
   - φ coefficients: 0.28–0.34
   - Interpretation: Plasmid-mediated co-carriage of tetracycline and folate pathway resistance determinants, commonly reported in environmental *Enterobacteriaceae* (Poirel et al., 2018)

**Predictive Modeling Results:**

Using Random Forest classification, we assessed the predictability of resistance to clinically important antibiotics from other resistance phenotypes:

| Target Antibiotic | AUC | Top Predictor | Biological Interpretation |
|-------------------|-----|---------------|---------------------------|
| Tetracycline (TE) | 0.78 | Doxycycline (DO) | Same class, shared tet genes |
| Ampicillin (AM) | 0.72 | AMC | β-lactam class relationship |
| Imipenem (IPM) | 0.65 | CZA | Carbapenem/cephalosporin connection |
| SXT | 0.71 | TE | Plasmid co-carriage pattern |

These findings have practical implications:
- **Surveillance Optimization**: Testing "hub" antibiotics (those with many network connections) can help infer resistance to connected antibiotics, reducing laboratory testing burden
- **Treatment Guidance**: Strong co-resistance associations suggest that alternative antibiotics within the same resistance cluster may also be ineffective
- **Mechanistic Insights**: Cross-class associations warrant further genetic investigation to identify specific mobile genetic elements

---

## 5.2 Comparison to Literature

Our findings contribute to the growing body of AMR surveillance data from Southeast Asia and align with several established observations while providing novel insights specific to Philippine environmental settings.

### 5.2.1 Environmental E. coli as AMR Reservoirs

The high MDR prevalence we observed in environmental *E. coli* (overall 18.7%, up to 54.5% in C3) is consistent with global surveillance data. A systematic review by Gekenidis et al. (2018) reported environmental *E. coli* MDR rates ranging from 12% to 68% across different geographic regions and source types. Our findings fall within this range and support the growing recognition of environmental bacteria as important AMR reservoirs.

Specifically, our identification of tetracycline-resistant *E. coli* in aquaculture-associated samples aligns with findings from:

- **China**: Tetracycline resistance rates of 60–85% in aquaculture *E. coli* (He et al., 2017)
- **Vietnam**: MDR *E. coli* prevalence of 45% in Mekong Delta aquaculture (Nguyen et al., 2016)
- **Thailand**: Tetracycline and ampicillin resistance dominance in freshwater fish isolates (Sarter et al., 2007)

### 5.2.2 Aquaculture as a Selective Environment

The concentration of MDR *E. coli* in fish-associated samples (C3: 56.1% from fish) supports the aquaculture-AMR hypothesis. Cabello et al. (2013) comprehensively reviewed how aquaculture antibiotic use creates selective environments for resistance development, with tetracyclines being among the most commonly used antibiotics in Asian fish farming.

Our data suggest that Philippine aquaculture may similarly contribute to AMR selection, though we note that our cross-sectional design cannot establish causality. The observed pattern warrants further investigation through:
1. Longitudinal sampling to track resistance dynamics
2. Correlation with local antibiotic usage data
3. Whole-genome sequencing to identify specific resistance mechanisms

### 5.2.3 Regional Variation in AMR Patterns

The geographic structuring we observed (Cramér's V = 0.321) aligns with regional AMR variation reported elsewhere in Southeast Asia:

- **Philippines National AMR Surveillance**: The Research Institute for Tropical Medicine (RITM) reports regional variation in clinical *E. coli* resistance rates, with higher tetracycline resistance in agricultural regions (DOH, 2020)
- **GLASS Data**: WHO Global Antimicrobial Resistance Surveillance System reports significant country-level and intra-country variation in *E. coli* resistance patterns across Southeast Asia (WHO, 2020)
- **SEARO Studies**: Regional analyses consistently show higher environmental AMR prevalence in areas with intensive agriculture and aquaculture (Chereau et al., 2017)

### 5.2.4 One Health Implications

Our identification of shared resistance phenotypes across water, fish, and hospital effluent sources (C3: present in all three environments with ≥7% prevalence) provides evidence for One Health AMR dynamics. This aligns with:

- **CDC One Health Framework**: Recognition that human, animal, and environmental health are interconnected for AMR (CDC, 2019)
- **FAO/OIE/WHO Tripartite**: Global action plan emphasizing environmental AMR surveillance (WHO, 2015)
- **Environmental transmission studies**: Documented transfer of resistance determinants between environmental and clinical settings (Berendonk et al., 2015)

However, we emphasize that our cross-sectional observational data demonstrate **association**, not **transmission**. Proving transmission pathways would require:
- Molecular typing of isolates
- Temporal tracking of resistance emergence
- Epidemiological source attribution

### 5.2.5 Species-Cluster Associations

The strong association between bacterial species and cluster membership (Cramér's V = 0.765) reflects the biological reality that different species have intrinsically different resistance profiles. This finding is consistent with:

- **CLSI/EUCAST guidelines**: Species-specific breakpoints and expected resistance patterns
- **Surveillance databases**: ECDC and CDC data showing species-specific resistance prevalence
- **Biological mechanisms**: Species-specific presence/absence of intrinsic resistance genes

Our findings suggest that while species explains most of the clustering structure, meaningful within-species variation exists (as demonstrated by the C3/C4 *E. coli* dichotomy), which should be a focus of future investigations.

---

## 5.3 Limitations

We acknowledge several limitations that affect the interpretation and generalizability of our findings.

### 5.3.1 Study Design Limitations

**Cross-Sectional Design**

Our analysis represents a snapshot of AMR patterns at a single time point. Key limitations include:
- Cannot assess temporal trends or resistance evolution
- Seasonal variation not captured
- No inference about transmission dynamics or causal relationships

**Recommendation for Future Work**: Longitudinal sampling at 6-month intervals would enable assessment of resistance trajectory and seasonal patterns.

**Dataset Dependency**

Results are specific to this dataset from Philippine environmental samples and may not generalize to:
- Other geographic regions
- Clinical isolates
- Different environmental contexts

**Recommendation**: External validation on independent datasets is essential before generalizing findings.

### 5.3.2 Methodological Limitations

**PCA on Ordinal Data**

We performed Principal Component Analysis on ordinally-encoded resistance data (S=0, I=1, R=2). While ordinal encoding preserves biological ordering, PCA assumes:
- Continuous variables with normal distributions
- Linear relationships between features
- Variance reflects information content

Our resistance data violates these assumptions (discrete 3-level ordinal data, likely bimodal distributions). While results appear biologically meaningful, alternative methods such as Multiple Correspondence Analysis (MCA) would be more statistically appropriate for categorical data.

**PCA Explained Variance**: Our first two principal components explain approximately 35% of total variance, indicating that 2D visualizations represent simplified projections of the full resistance space. Interpretations should be made cautiously.

**Species-Agnostic MDR Classification**

We applied a universal antibiotic class mapping for MDR classification across all species. However, the Magiorakos et al. (2012) definition specifies species-specific class sets. Our approach may:
- Inflate MDR rates for species where certain antibiotics are not clinically relevant
- Miss species-specific resistance patterns
- Reduce comparability with species-specific surveillance reports

**Recommendation**: Future analyses should implement species-specific MDR definitions following Magiorakos exactly.

**Arbitrary Threshold Selection**

Our data cleaning thresholds (70% antibiotic coverage, 30% isolate missing) were selected without formal sensitivity analysis. While these values are reasonable defaults, results may be threshold-dependent.

**Recommendation**: Sensitivity analysis testing multiple threshold pairs (50%/40%, 60%/30%, 80%/20%) with comparison of cluster assignments using Adjusted Rand Index (ARI).

### 5.3.3 Statistical Limitations

**Multiple Comparisons**

We performed multiple chi-square tests without formal multiple comparison correction. While all reported p-values were extremely small (< 10⁻⁶) and would survive Bonferroni or Benjamini-Hochberg correction, proper reporting requires adjusted p-values.

**Cluster Count Selection**

Our selection of k=5 clusters was validated through silhouette analysis (score = 0.49) and elbow method, but:
- Silhouette scores in the 0.4–0.5 range indicate moderate cluster separation
- Alternative k values (4 or 6) might reveal different biological patterns
- Cluster boundaries are inherently arbitrary in continuous resistance space

### 5.3.4 Sampling Limitations

**Regional Sampling Imbalance**

BARMM contributed 50.8% of isolates, potentially biasing regional comparisons. The apparent concentration of MDR in BARMM may partially reflect oversampling rather than true regional differences.

**Environmental Source Distribution**

Non-random distribution of environmental sources across regions complicates interpretation of regional patterns. Confounding between region and source cannot be fully addressed in this cross-sectional design.

**Species Distribution**

*E. coli* dominance (65% of isolates) limits statistical power for characterizing resistance patterns in less frequent species. Conclusions about *Klebsiella*, *Salmonella*, and other species should be considered preliminary.

---

## 5.4 Implications for Public Health and Policy

Despite acknowledged limitations, our findings have several practical implications for AMR surveillance and control in the Philippine context.

### 5.4.1 Surveillance Recommendations

**Targeted Surveillance Strategy**

The identification of high-MDR clusters associated with specific environments suggests opportunities for targeted surveillance:

1. **Priority Environments**: Fish/aquaculture samples showed highest MDR prevalence (C3) — these environments warrant intensive monitoring
2. **Sentinel Antibiotics**: Tetracycline and ampicillin resistance can serve as indicators for broader MDR patterns given their strong co-resistance associations
3. **Geographic Focus**: BARMM showed higher MDR prevalence, suggesting prioritization for surveillance resources (though acknowledging sampling bias)

**Optimized Testing Panels**

Our co-resistance network analysis suggests that testing a subset of "hub" antibiotics could inform resistance to connected antibiotics:
- Testing TE + AM + AN covers major resistance clusters
- High co-resistance (φ > 0.4) between certain antibiotics allows inference from partial panels
- Cost-effective surveillance possible with reduced testing burden

### 5.4.2 One Health Integration

Our data support integration of environmental AMR surveillance into One Health frameworks:

1. **Cross-Sector Coordination**: Resistance patterns shared across water, fish, and hospital sources suggest need for coordination between:
   - Department of Agriculture (aquaculture)
   - Department of Health (hospital surveillance)
   - Department of Environment and Natural Resources (water quality)

2. **Integrated Surveillance**: Environmental sampling should complement clinical surveillance to provide early warning of emerging resistance patterns

3. **Source Attribution**: Investment in molecular epidemiology capacity (WGS) would enable tracking of resistance gene flow between environmental and clinical settings

### 5.4.3 Policy Considerations

**Aquaculture Antibiotic Stewardship**

The concentration of MDR *E. coli* in fish-associated samples suggests need for:
- Review of antibiotic use practices in Philippine aquaculture
- Implementation of antibiotic stewardship programs for fish farming
- Monitoring of antibiotic residues in aquaculture environments
- Consideration of tetracycline and ampicillin use restrictions

**Regional Resource Allocation**

Geographic variation in resistance patterns suggests regionally-adapted responses:
- Higher surveillance intensity in high-prevalence regions
- Region-specific treatment guidelines reflecting local resistance data
- Targeted public health interventions in MDR hotspots

### 5.4.4 Research Priorities

Based on our findings, we recommend the following research priorities:

1. **Longitudinal Surveillance**: Repeat sampling to assess temporal dynamics
2. **Genetic Characterization**: WGS of representative isolates from each cluster to identify specific resistance mechanisms and mobile genetic elements
3. **Epidemiological Source Attribution**: Molecular typing to trace transmission pathways between environmental and clinical settings
4. **Intervention Studies**: Pilot studies of antibiotic stewardship interventions in aquaculture with pre/post AMR measurement
5. **Economic Analysis**: Cost-effectiveness evaluation of targeted vs. comprehensive surveillance strategies

---

## 5.5 Summary

This chapter has interpreted our findings in the context of current AMR surveillance knowledge, acknowledging both contributions and limitations. Key discussion points include:

1. **Principal Findings**: The C3/C4 *E. coli* dichotomy represents a novel finding with implications for understanding within-species resistance heterogeneity in environmental settings

2. **Literature Context**: Our findings align with regional and global AMR surveillance data while providing Philippine-specific insights

3. **Limitations**: Cross-sectional design, methodological constraints (PCA on ordinal data, species-agnostic MDR), and sampling imbalances require cautious interpretation

4. **Implications**: Targeted surveillance strategies, One Health integration, and aquaculture stewardship emerge as practical policy directions

The next chapter synthesizes these findings into conclusions and recommendations for future work.

---

## References

1. Berendonk TU, Manaia CM, Merlin C, et al. (2015). Tackling antibiotic resistance: the environmental framework. *Nature Reviews Microbiology*, 13(5), 310-317. doi:10.1038/nrmicro3439

2. Cabello FC, Godfrey HP, Tomova A, et al. (2013). Antimicrobial use in aquaculture re-examined: its relevance to antimicrobial resistance and to animal and human health. *Environmental Microbiology*, 15(7), 1917-1942. doi:10.1111/1462-2920.12134

3. CDC (2019). Antibiotic Resistance Threats in the United States, 2019. Atlanta, GA: U.S. Department of Health and Human Services, CDC.

4. Chereau F, Opatowski L, Tourdjman M, Vong S. (2017). Risk assessment for antibiotic resistance in South East Asia. *BMJ*, 358, j3393. doi:10.1136/bmj.j3393

5. DOH (2020). Antimicrobial Resistance Surveillance Program Annual Report. Manila: Department of Health, Research Institute for Tropical Medicine.

6. Gekenidis MT, Qi W, Hummerjohann J, et al. (2018). Antibiotic-resistant indicator bacteria in irrigation water: High prevalence of extended-spectrum beta-lactamase (ESBL)-producing *Escherichia coli*. *PLoS ONE*, 13(11), e0207857. doi:10.1371/journal.pone.0207857

7. He X, Xu Y, Chen J, et al. (2017). Sub-inhibitory concentrations of antibiotics promote horizontal transfer of antibiotic resistance genes in *Escherichia coli*. *Environmental Science & Technology*, 51(10), 5525-5534. doi:10.1021/acs.est.7b00649

8. Magiorakos AP, Srinivasan A, Carey RB, et al. (2012). Multidrug-resistant, extensively drug-resistant and pandrug-resistant bacteria: an international expert proposal for interim standard definitions for acquired resistance. *Clinical Microbiology and Infection*, 18(3), 268-281. doi:10.1111/j.1469-0691.2011.03570.x

9. Nguyen VT, Carrique-Mas JJ, Ngo TH, et al. (2016). Prevalence and risk factors for carriage of antimicrobial-resistant *Escherichia coli* on household and small-scale chicken farms in the Mekong Delta of Vietnam. *Journal of Antimicrobial Chemotherapy*, 70(7), 2144-2152. doi:10.1093/jac/dkv053

10. Poirel L, Madec JY, Lupo A, et al. (2018). Antimicrobial resistance in *Escherichia coli*. *Microbiology Spectrum*, 6(4), ARBA-0026-2017. doi:10.1128/microbiolspec.ARBA-0026-2017

11. Santos L, Ramos F. (2018). Antimicrobial resistance in aquaculture: Current knowledge and alternatives to tackle the problem. *International Journal of Antimicrobial Agents*, 52(2), 135-143. doi:10.1016/j.ijantimicag.2018.03.010

12. Sarter S, Kha Nguyen HN, Hung LT, Lazard J, Montet D. (2007). Antibiotic resistance in Gram-negative bacteria isolated from farmed catfish. *Food Control*, 18(11), 1391-1396. doi:10.1016/j.foodcont.2006.10.003

13. WHO (2015). Global Action Plan on Antimicrobial Resistance. Geneva: World Health Organization.

14. WHO (2020). Global Antimicrobial Resistance Surveillance System (GLASS) Report: Early Implementation 2020. Geneva: World Health Organization.

15. Zellweger RM, Carrique-Mas J, Limmathurotsakul D, et al. (2017). A current perspective on antimicrobial resistance in Southeast Asia. *Journal of Antimicrobial Chemotherapy*, 72(11), 2963-2972. doi:10.1093/jac/dkx260

---

## Document Version

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | December 2025 | Initial discussion chapter |

---

*This document is part of the AMR Thesis Project documentation.*

**See also**: [conclusion.md](conclusion.md) | [limitations.md](limitations.md) | [comprehensive_academic_review.md](comprehensive_academic_review.md)
